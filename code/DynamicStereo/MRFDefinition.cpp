//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"
#include "external/MRF2.2/GCoptimization.h"
#include "local_matcher.h"
#include "../base/thread_guard.h"
#ifdef USE_CUDA
#include "cudaWrapper.h"
#endif

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo {

    void DynamicStereo::initMRF() {
        CHECK(model.get());
        model->allocate();
        double min_depth, max_depth;
        utility::computeMinMaxDepth(sfmModel, anchor, min_depth, max_depth);
        CHECK_GT(min_depth, 0.0);
        CHECK_GT(max_depth, 0.0);
        model->min_disp = 1.0 / max_depth;
        model->max_disp = 1.0 / min_depth;
        cout << "Assigning data term..." << endl << flush;
        assignDataTerm();
        //computeFrequencyConfidence();
        assignSmoothWeight();
    }

    void DynamicStereo::getPatchArray(const double x, const double y, const int d, const int r, const theia::Camera& refCam, vector<vector<double> >& patches) const {
        double depth = model->dispToDepth(d);
        //sample in 3D space
        vector<Vector4d> sptBase;
        const double scale = 2.0;
        const double dr = (double)r / (double)downsample * scale;
        for (double dy = -1 * dr; dy <= dr; dy += scale  / (double)downsample) {
            for (double dx = -1 * dr; dx <= dr; dx += scale / (double)downsample) {
                Vector2d pt(x + dx, y + dy);
                if (pt[0] < 0 || pt[1] < 0 || pt[0] > width - 1 || pt[1] > height - 1) {
                    sptBase.push_back(Vector4d(0, 0, 0, 0));
                    continue;
                }
                Vector3d ray = refCam.PixelToUnitDepthRay(pt * (double)downsample);
                Vector3d spt = refCam.GetPosition() + ray * depth;
                Vector4d spt_homo(spt[0], spt[1], spt[2], 1.0);
                sptBase.push_back(spt_homo);
            }
        }

        //project onto other views and compute matching cost
        patches.resize((size_t)(images.size()/stereo_stride));
        int index = 0;
        for (auto v = 0; v < images.size(); v += stereo_stride, ++index) {
            //const theia::Camera& cam2 = reconstruction.View(orderedId[v + offset].second)->Camera();
            const theia::Camera& cam2 = sfmModel.getCamera(v+offset);

            //printf("--------------Sample from frame %d\n", v+offset);
            for (const auto &spt: sptBase) {
                if (spt[3] == 0) {
                    patches[index].push_back(-1);
                    patches[index].push_back(-1);
                    patches[index].push_back(-1);
                } else {
                    Vector2d imgpt;
                    cam2.ProjectPoint(spt, &imgpt);
                    //printf("image pt: (%.2f,%.2f)\t", imgpt[0], imgpt[1]);
                    imgpt = imgpt / (double) downsample;

                    if (imgpt[0] < 0 || imgpt[1] < 0 || imgpt[0] > width - 1 || imgpt[1] > height - 1) {
                        patches[index].push_back(-1);
                        patches[index].push_back(-1);
                        patches[index].push_back(-1);
                    } else {
                        Vector3d c = interpolation_util::bilinear<uchar, 3>(images[v].data, width,
                                                                            height, imgpt);
                        patches[index].push_back(c[0]);
                        patches[index].push_back(c[1]);
                        patches[index].push_back(c[2]);
                        //	printf("Color: (%.2f,%.2f,%.2f)\n", c[0], c[1], c[2]);
                    }
                }
            }
        }
    }

    void DynamicStereo::computeMatchingCostCPU(){
        LOG(INFO) << "Computing matching cost on CPU";
        const theia::Camera& cam1 = sfmModel.getCamera(anchor);
        int unit = height / 10;
        //compute matching cost in multiple threads
        //Be careful about down sample ratio!!!!!
        auto threadFun = [&](int offset, const int N){
            for(auto y=offset; y<height; y+=N) {
                for (auto x = 0; x < width; ++x) {
                    for (auto d = 0; d < dispResolution; ++d) {
                        vector<vector<double> > patches;
                        getPatchArray((double) x, (double) y, d, pR, cam1, patches);
                        //assume that the patch of reference view is in the middle
                        double mCost = local_matcher::sumMatchingCost(patches, patches.size() / 2);
                        //double mCost = local_matcher::medianMatchingCost(patches, (int)patches.size() / 2);
                        //model->operator()(y*width+x, d) = (EnergyType) ((1 + mCost) * model->MRFRatio);
                        model->operator()(y * width + x, d) = (EnergyType)((1 - mCost) * model->MRFRatio);
                    }
                }
            }
        };

        const int num_threads = 6;
        vector<thread_guard> threads((size_t)num_threads);
        for(auto tid=0; tid<num_threads; ++tid){
            std::thread t(threadFun,tid,num_threads);
            threads[tid].bind(t);
        }
        for(auto& t: threads)
            t.join();
    }

#ifdef USE_CUDA
    void DynamicStereo::computeMatchingCostGPU() {
        LOG(INFO) << "Computing matching cost on GPU...";
        if (!checkDevice()) {
            printf("The GPU doesn't meet the requirement, switch to CPU mode...\n");
            computeMatchingCostCPU();
            return;
        }
        const int N = (int) images.size() / stereo_stride;
        vector<unsigned char> images_data(N * width * height * 3);
        vector<unsigned char> refImage_data(width * height * 3);

        const int extSize = 6;
        const int intSize = 7;

        vector<TCam> extrinsics(extSize * N, 0.0f);
        vector<TCam> intrinsics(intSize * N, 0.0f);

        auto copyCamera = [](const theia::Camera &cam, TCam *intrinsic, TCam *extrinsic) {
            Vector3d pos = cam.GetPosition();
            Vector3d ax = cam.GetOrientationAsAngleAxis();
            for (auto i = 0; i < 3; ++i) {
                extrinsic[i] = pos[i];
                extrinsic[i + 3] = ax[i];
            }

//            CHECK_EQ(cam.GetCameraIntrinsicsModelType(), theia::CameraIntrinsicsModelType::PINHOLE);
            const double* intParam = cam.intrinsics();
            intrinsic[0] = (TCam) intParam[theia::PinholeCameraModel::FOCAL_LENGTH];
            intrinsic[1] = (TCam) intParam[theia::PinholeCameraModel::ASPECT_RATIO];
            intrinsic[2] = (TCam) intParam[theia::PinholeCameraModel::SKEW];
            intrinsic[3] = (TCam) intParam[theia::PinholeCameraModel::PRINCIPAL_POINT_X];
            intrinsic[4] = (TCam) intParam[theia::PinholeCameraModel::PRINCIPAL_POINT_Y];
            intrinsic[5] = (TCam) intParam[theia::PinholeCameraModel::RADIAL_DISTORTION_1];
            intrinsic[6] = (TCam) intParam[theia::PinholeCameraModel::RADIAL_DISTORTION_2];
        };

        //copy data
        for (auto v = 0; v < N; ++v) {
            for (auto i = 0; i < width * height * 3; ++i)
                images_data[v * width * height * 3 + i] = images[v * stereo_stride].data[i];
            copyCamera(sfmModel.getCamera(v * stereo_stride + offset), intrinsics.data() + intSize * v,
                       extrinsics.data() + extSize * v);
        }

        vector<TCam> refIntrinsic(intSize, 0.0f);
        vector<TCam> refExtrinsic(extSize, 0.0f);
        copyCamera(sfmModel.getCamera(anchor), refIntrinsic.data(), refExtrinsic.data());
        for (auto i = 0; i < width * height * 3; ++i)
            refImage_data[i] = images[anchor - offset].data[i];

        //compute space point coordinate
        vector<TCam> spts(width * height * 3);
        const theia::Camera& refCam = sfmModel.getCamera(anchor);
#pragma omp parallel for
        for (auto y = 0; y < height; ++y) {
            for (auto x = 0; x < width; ++x) {
                Vector3d ray = refCam.PixelToUnitDepthRay(Vector2d(x, y) * downsample);
                spts[(y * width + x) * 3] = (TCam)ray[0];
                spts[(y * width + x) * 3 + 1] = (TCam)ray[1];
                spts[(y * width + x) * 3 + 2] = (TCam)ray[2];
            }
        };

        //allocate space for result
        vector<TOut> result(width * height * dispResolution);
        callStereoMatching(images_data, refImage_data, width, height, N,
                           model->min_disp, model->max_disp, model->downsample,
                           intrinsics, extrinsics, refIntrinsic, refExtrinsic, spts, model->nLabel, pR, result);

        for (auto i = 0; i < result.size(); ++i)
            model->unary[i] = (double) result[i] * model->MRFRatio;
    }
#endif

    void DynamicStereo::assignDataTerm() {
        CHECK_GT(model->min_disp, 0);
        CHECK_GT(model->max_disp, 0);
        //read from cache
        char buffer[1024] = {};
        sprintf(buffer, "%s/midres/matching%05dR%dD%d", file_io.getDirectory().c_str(), anchor, dispResolution, downsample);
        ifstream fin(buffer, ios::binary);
        bool recompute = true;
        if (fin.is_open()) {
            int frame, resolution, st, ds, type;
            double mindisp, maxdisp;
            fin.read((char *) &frame, sizeof(int));
            fin.read((char *) &resolution, sizeof(int));
            fin.read((char *) &st, sizeof(int));
            fin.read((char *) &ds, sizeof(int));
            fin.read((char *) &type, sizeof(int));
            fin.read((char *) &mindisp, sizeof(double));
            fin.read((char *) &maxdisp, sizeof(double));
            printf("Cached data: anchor:%d, resolution:%d, twindow:%d, downsample:%d, Energytype:%d, min_disp:%.15f, max_disp:%.15f\n",
                   frame, resolution, st, ds, type, mindisp, maxdisp);
            printf("Current config:: anchor:%d, resolution:%d, twindow:%d, downsample:%d, Energytype:%d, min_disp:%.15f, max_disp:%.15f\n",
                   anchor, dispResolution, stereo_stride, downsample, (int)sizeof(EnergyType), model->min_disp, model->max_disp);
            if (frame == anchor && resolution == dispResolution && st == stereo_stride &&
                type == sizeof(EnergyType) && ds == downsample) {
                printf("Reading unary term from cache...\n");
                fin.read((char *) model->unary.data(), model->unary.size() * sizeof(EnergyType));
                recompute = false;
            }
        }
        if(recompute) {
            float start_t = getTickCount();
#if USE_CUDA
            computeMatchingCostGPU();
#else
            computeMatchingCostCPU();
#endif

            printf("Time usage for stereo matching: %.2fs\n", ((float)getTickCount() - start_t) / (float)getTickFrequency());
            //caching
            ofstream fout(buffer, ios::binary);
            if (!fout.is_open()) {
                printf("Can not open cache file to write: %s\n", buffer);
                return;
            }
            printf("Writing unary term to cache...\n");
            int sz = sizeof(EnergyType);
            fout.write((char *) &anchor, sizeof(int));
            fout.write((char *) &dispResolution, sizeof(int));
            fout.write((char *) &stereo_stride, sizeof(int));
            fout.write((char *) &downsample, sizeof(int));
            fout.write((char *) &sz, sizeof(int));
            fout.write((char *) &model->min_disp, sizeof(double));
            fout.write((char *) &model->max_disp, sizeof(double));
            fout.write((char *) model->unary.data(), model->unary.size() * sizeof(EnergyType));

            fout.close();
        }

    }

    void DynamicStereo::assignSmoothWeight() {
        LOG(INFO) << "Assining smoothness weight";
        vector<EnergyType> &vCue = model->vCue;
        vector<EnergyType> &hCue = model->hCue;
        const double t = 80;
        const Mat &img = model->image;
        for (auto y = 0; y < height; ++y) {
            for (auto x = 0; x < width; ++x) {
                Vec3b pix1 = img.at<Vec3b>(y, x);
                //pixel value range from 0 to 1, not 255!
                Vector3d dpix1 = Vector3d(pix1[0], pix1[1], pix1[2]);
                if (y < height - 1) {
                    Vec3b pix2 = img.at<Vec3b>(y + 1, x);
                    Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]);
                    double diff = (dpix1 - dpix2).squaredNorm();
                    vCue[y*width+x] = (EnergyType) std::max(0.15, std::log(1+std::exp(-1*diff/(t*t))));
                }
                if (x < width - 1) {
                    Vec3b pix2 = img.at<Vec3b>(y, x + 1);
                    Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]);
                    double diff = (dpix1 - dpix2).squaredNorm();
                    hCue[y*width+x] = (EnergyType) std::max(0.15, std::log(1+std::exp(-1*diff/(t*t))));
                }
            }
        }

//		char buffer[1024] ={};
//		Mat outimgv(height, width, CV_8UC1);
//		uchar* pOutimgv = outimgv.data;
//		for(auto i=0; i<width * height; ++i) {
//			pOutimgv[i] = (uchar) (256 * vCue[i]);
//		}
//
//		sprintf(buffer, "%s/temp/cueV.png", file_io.getDirectory().c_str());
//		imwrite(buffer, outimgv);
//
//		Mat outimgh(height, width, CV_8UC1);
//		uchar* pOutimgh = outimgh.data;
//		for(auto i=0; i<width * height; ++i){
//			pOutimgh[i] = (uchar)(256 * hCue[i]);
//		}
//		sprintf(buffer, "%s/temp/cueH.png", file_io.getDirectory().c_str());
//		imwrite(buffer, outimgh);
    }
}//namespace dynamic_stereo
