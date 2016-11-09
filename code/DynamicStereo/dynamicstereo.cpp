//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"
#include "optimization.h"
#include "local_matcher.h"

using namespace std;
using namespace cv;
using namespace Eigen;
namespace dynamic_stereo{


    DynamicStereo::DynamicStereo(const dynamic_stereo::FileIO &file_io_, const int anchor_,
                                 const int tWindow_, const int stereo_stride_, const int downsample_, const double weight_smooth_, const int dispResolution_,
                                 const double min_disp_, const double max_disp_):
            file_io(file_io_), anchor(anchor_), tWindow(tWindow_), stereo_stride(stereo_stride_), downsample(downsample_), dispResolution(dispResolution_),
            pR(2), dbtx(-1), dbty(-1){
        CHECK_GE(stereo_stride, 1);
        LOG(INFO) << "Reading...";
        offset = anchor >= tWindow / 2 ? anchor - tWindow / 2 : 0;
        CHECK_GE(file_io.getTotalNum(), offset + tWindow);
        LOG(INFO) << "Reading reconstruction" << endl;
        sfmModel.init(file_io.getReconstruction());

        CHECK(downsample == 1 || downsample == 2 || downsample == 4 || downsample == 8) << "Invalid downsample ratio!";
        images.resize((size_t)tWindow);

        LOG(INFO) << "Reading images";
        const int nLevel = (int)std::log2((double)downsample) + 1;
        for(auto i=0; i<tWindow; ++i){
            vector<Mat> pyramid(nLevel);
            Mat tempMat = imread(file_io.getImage(i + offset));
            pyramid[0] = tempMat;
            for(auto k=1; k<nLevel; ++k)
                pyrDown(pyramid[k-1], pyramid[k]);
            images[i] = pyramid.back().clone();
        }
        CHECK_GT(images.size(), 2) << "Too few images";
        width = images.front().cols;
        height = images.front().rows;
        dispUnary.initialize(width, height, 0.0);

        model = shared_ptr<StereoModel<EnergyType> >(new StereoModel<EnergyType>(images[anchor-offset], (double)downsample, dispResolution, 1, weight_smooth_));
    }


    void DynamicStereo::runStereo(Depth& depth_firstOrder, cv::Mat& depthMask, const bool dryrun) {

        if(dryrun && dbtx < 0 && dbty < 0)
            return;

        char buffer[1024] = {};
        initMRF();
        for(auto y=0; y<height; ++y){
            for(auto x=0; x<width; ++x){
                EnergyType min_energy = numeric_limits<EnergyType>::max();
                for (int d = 0; d < dispResolution; ++d) {
                    const EnergyType curEnergy = model->operator()(y*width+x, d);
                    if ((double) curEnergy < min_energy) {
                        dispUnary.setDepthAtInt(x, y, (double) d);
                        min_energy = curEnergy;
                    }
                }
            }
        }

        sprintf(buffer, "%s/temp/unary%05d.png", file_io.getDirectory().c_str(), anchor);
        dispUnary.saveImage(buffer, -1);

        depthMask = Mat(height, width, CV_8UC1, Scalar(255));
        if(dbtx >= 0 && dbty >= 0){
            //debug: inspect unary term
            int dtx = (int)dbtx / downsample;
            int dty = (int)dbty / downsample;

            //const theia::Camera &cam = reconstruction.View(orderedId[anchor].second)->Camera();
            const theia::Camera &cam = sfmModel.getCamera(anchor);
            Vector3d ray = cam.PixelToUnitDepthRay(Vector2d(dbtx, dbty));

            int tdisp = (int) dispUnary(dtx, dty);
//			int tdisp = 142;
            double td = model->dispToDepth(tdisp);
            for(auto d=0; d<dispResolution; ++d)
                printf("%.3f ", model->operator()(dty*width + dtx, d));
            cout<< endl;
            cout << "Cost at d=" << tdisp << ": " << model->operator()(dty * width + dtx, tdisp) << endl;

            printf("CPU ray: (%.3f,%.3f,%.3f)\n", ray[0], ray[1], ray[2]);
            Vector3d spt = cam.GetPosition() + ray * td;
            printf("CPU camera position:(%.3f,%.3f,%.3f)\n", cam.GetPosition()[0], cam.GetPosition()[1], cam.GetPosition()[2]);
            printf("CPU camera axis:(%.3f,%.3f,%.3f)\n", cam.GetOrientationAsAngleAxis()[0], cam.GetOrientationAsAngleAxis()[1], cam.GetOrientationAsAngleAxis()[2]);

            printf("CPU 3d point: (%.3f,%.3f,%.3f)\n", spt[0], spt[1], spt[2]);
            for (auto v = 0; v < images.size(); ++v) {
                Mat curimg = imread(file_io.getImage(v + offset));
                Vector2d imgpt;
                const theia::Camera& tgtCam = sfmModel.getCamera(v+offset);
                double curdepth = tgtCam.ProjectPoint(
                        Vector4d(spt[0], spt[1], spt[2], 1.0), &imgpt);
                if(v == 0){
                    printf("Target camera:\n");
                    printf("CPU camera position:(%.3f,%.3f,%.3f)\n", tgtCam.GetPosition()[0], tgtCam.GetPosition()[1], tgtCam.GetPosition()[2]);
                    printf("CPU camera axis:(%.3f,%.3f,%.3f)\n", tgtCam.GetOrientationAsAngleAxis()[0], tgtCam.GetOrientationAsAngleAxis()[1], tgtCam.GetOrientationAsAngleAxis()[2]);
                    printf("CPU projected point: (%.3f,%.3f)\n", imgpt[0], imgpt[1]);
                }
                if (imgpt[0] >= 0 && imgpt[1] >= 0 && imgpt[0] < curimg.cols && imgpt[1] < curimg.rows)
                    cv::circle(curimg, cv::Point(imgpt[0], imgpt[1]), 1, cv::Scalar(255, 0, 0), 2);
                sprintf(buffer, "%s/temp/project_b%05d_v%05d.jpg", file_io.getDirectory().c_str(), anchor,
                        v + offset);

                imgpt = imgpt / (double)downsample;
                imwrite(buffer, curimg);
            }
        }


        if(dryrun)
            return;

        LOG(INFO) << "Solving with first order smoothness...";
        FirstOrderOptimize optimizer_firstorder(file_io, (int)images.size(), model);
        Depth result_firstOrder;
        optimizer_firstorder.optimize(result_firstOrder, 100);


//		Depth depth_firstOrder;
        //masking out invalid region
        //remove pixel where half disparity project outof half frames
//		const theia::Camera& refCam = sfmModel.getCamera(anchor);
//		const double invisThreshold = 0.3;
//
//		for(auto y=0; y<height; ++y){
//			for(auto x=0; x<width; ++x){
//				Vector3d ray = refCam.PixelToUnitDepthRay(Vector2d(x*downsample, y*downsample));
//				Vector3d spt = refCam.GetPosition() + depth_firstOrder(x,y) * ray;
//				double invisCount = 0.0;
//				for(auto v=0; v < images.size(); ++v){
//					Vector2d imgpt;
//					sfmModel.getCamera(v+offset).ProjectPoint(spt.homogeneous(), &imgpt);
//					if(imgpt[0] < 0 || imgpt[1] < 0 || imgpt[0] >= width * downsample || imgpt[1] >= height * downsample)
//						invisCount += 1.0;
//				}
//				if(invisCount / (double)images.size()> invisThreshold)
//					depthMask.at<uchar>(y,x) = (uchar)0;
//			}
//		}

        sprintf(buffer, "%s/temp/mesh_firstorder_b%05d.ply", file_io.getDirectory().c_str(), anchor);
        utility::saveDepthAsPly(string(buffer), depth_firstOrder, images[anchor - offset],
                                sfmModel.getCamera(anchor), downsample);

        if(dbtx >=0 && dbty >= 0){
            printf("Result disparity for (%d,%d): %d\n", (int)dbtx, (int)dbty, (int)result_firstOrder((int)dbtx/downsample, (int)dbty/downsample));
        }
    }

    void DynamicStereo::disparityToDepth(const Depth& disp, Depth& depth){
        depth.initialize(disp.getWidth(), disp.getHeight(), -1);
        for(auto i=0; i<disp.getWidth() * disp.getHeight(); ++i) {
            if(disp[i] < 0) {
                depth[i] = -1;
                continue;
            }
            depth[i] = model->dispToDepth(disp[i]);
        }
    }

    void DynamicStereo::bilateralFilter(const Depth &input, const cv::Mat &inputImg, Depth &output,
                                        const int size, const double sigmas, const double sigmar, const double sigmau) {
        CHECK_EQ(input.getWidth(), inputImg.cols);
        CHECK_EQ(input.getHeight(), inputImg.rows);
        CHECK_EQ(size % 2, 1);
        CHECK_GT(size, 2);
        CHECK_EQ(inputImg.type(), CV_8UC3) << "Guided image should be 3 channel uchar type.";
        const int R = (size - 1) / 2;
        const int width = input.getWidth();
        const int height = input.getHeight();
        output.initialize(width, height, -1);
        const uchar *pImg = inputImg.data;

        //kerner weight the computed from:
        //1. distance
        //2. color consistancy
        //3. unary confidence

        double conf_thres = 10;
        vector<double> wunary((size_t)width * height);
        for(auto i=0; i<width * height; ++i){
            vector<double> unary(dispResolution);
            for(auto j=0; j<dispResolution; ++j)
                unary[j] = model->operator()(i,j);
            auto min_pos = min_element(unary.begin(), unary.end());
            double minu = *min_pos;
            if(minu == 0){
//				printf("(%d,%d)\n", i/width, i%width);
//				for(auto j=0; j<dispResolution; ++j)
//					cout << model->operator()(i,j) << ' ' << unary[j]<<endl;
//				CHECK_GT(minu, 0);
                wunary[i] = 0.001;
                continue;
            }
            *min_pos = std::numeric_limits<double>::max();
            double seminu = *max_element(unary.begin(), unary.end());
            double conf = seminu / minu;
            if(conf > conf_thres)
                conf = conf_thres;
            wunary[i] = math_util::gaussian(conf_thres, sigmau, conf);
        }

        const double max_disp_diff = 10;
        //apply bilateral filter
        for (auto y = 0; y < height; ++y) {
#pragma omp parallel for
            for (auto x = 0; x < width; ++x) {
                int cidx = y * width + x;
                double m = 0.0;
                double acc = 0.0;
                Vector3d pc(pImg[3 * cidx], pImg[3 * cidx + 1], pImg[3 * cidx + 2]);
                double disp1 = input(x,y);
                for (auto dx = -1 * R; dx <= R; ++dx) {
                    for (auto dy = -1 * R; dy <= R; ++dy) {
                        int curx = x + dx;
                        int cury = y + dy;
                        const int kid = (dy + R) * size + dx + R;
                        if (curx < 0 || cury < 0 || curx >= width - 1 || cury >= height - 1)
                            continue;
                        double disp2 = input(curx,cury);
                        if(abs(disp1 - disp2) > max_disp_diff)
                            continue;
                        int idx = cury * width + curx;
                        Vector3d p2(pImg[3 * idx], pImg[3 * idx + 1], pImg[3 * idx + 2]);
                        double wcolor = (p2 - pc).squaredNorm() / (sigmar * sigmar);
                        double wdis = (dx * dx + dy * dy) / (sigmas * sigmas);
                        double w = std::exp(-1 * (wcolor + wdis)) * wunary[idx];
                        m += w;
                        acc += input(curx, cury) * w;
                    }
                }
                if(m != 0)
                    output(x, y) = acc / m;
                else
                    output(x,y) = input(x,y);
            }
        }
    }

}
