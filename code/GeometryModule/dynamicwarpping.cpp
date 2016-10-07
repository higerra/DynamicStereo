//
// Created by yanhang on 4/10/16.
//

#include "stereomodel.h"
#include "dynamic_utility.h"
#include "dynamicwarpping.h"
#include "../base/file_io.h"
#include "../base/depth.h"

#include "../base/utility.h"
#include "../base/thread_guard.h"

using namespace std;
using namespace Eigen;
using namespace cv;

namespace dynamic_stereo {
    DynamicWarpping::DynamicWarpping(const FileIO &file_io_, const int anchor_, const int tWindow_, const int nLabel_,
                                     const double wSmooth) :
            file_io(file_io_), anchor(anchor_), nLabel(nLabel_), tWindow(tWindow_){
        if (anchor - tWindow_ / 2 < 0) {
            offset = 0;
            CHECK_LT(offset + tWindow_, file_io.getTotalNum());
        } else if (anchor + tWindow_ / 2 >= file_io.getTotalNum()) {
            offset = file_io.getTotalNum() - 1 - tWindow_;
            CHECK_GE(offset, 0);
        } else
            offset = anchor - tWindow_ / 2;
        char buffer[1024] = {};
        //reading images. Set (0,0,0) to (1,1,1);
        //initialize reconstruction
        sfmModel.reset(new SfMModel());
        CHECK_NOTNULL(sfmModel.get())->init(file_io.getReconstruction());

        sprintf(buffer, "%s/midres/depth%05d.depth", file_io.getDirectory().c_str(), anchor);
        refDepth.reset(new Depth());
        CHECK_NOTNULL(refDepth.get())->readDepthFromFile(string(buffer));
        CHECK(!refDepth->getRawData().empty());

        Mat refImage = imread(file_io.getImage(anchor));
        CHECK(refImage.data);
        downsample = (double)refImage.cols / refDepth->getWidth();

        if(wSmooth > 0){
            refDepth->fillholeAndSmooth(wSmooth);
//            Mat depthTex;
//            cv::resize(refImage, depthTex, cv::Size(refDepth->getWidth(), refDepth->getHeight()));
//            sprintf(buffer, "%s/temp/smoothedDepth%05d.ply", file_io.getDirectory().c_str(), anchor);
//            utility::saveDepthAsPly(string(buffer), *(refDepth.get()), depthTex, sfmModel->getCamera(anchor), 2);
        }

        initZBuffer();
    }

    void DynamicWarpping::updateZBuffer(const Depth* depth, Depth* zb, const theia::Camera &cam1,
                                        const theia::Camera &cam2) const {
        CHECK_NOTNULL(zb);
        CHECK_NOTNULL(depth);
        const int w = depth->getWidth();
        const int h = depth->getHeight();
        CHECK_EQ(depth->getWidth(), zb->getWidth());
        CHECK_EQ(depth->getHeight(), zb->getHeight());
        for (auto y = 0; y < h; ++y) {
            for (auto x = 0; x < w; ++x) {
                Vector2d refPt((double)x, (double)y);
                Vector3d spt = cam1.PixelToUnitDepthRay(refPt * downsample) * depth->getDepthAt(refPt) +
                               cam1.GetPosition();
                Vector2d imgpt;
                double curd = cam2.ProjectPoint(spt.homogeneous(), &imgpt);
                imgpt /= (double) downsample;
                int intx = round(imgpt[0]+0.5);
                int inty = round(imgpt[1]+0.5);
                if (curd > 0 && imgpt[0] >= 0 && imgpt[1] >= 0 && imgpt[0] < w-1 && imgpt[1] < h-1) {
                    if ((*zb)(intx, inty) < 0 || (zb->getDepthAt(imgpt) >= 0 && curd <= zb->getDepthAt(imgpt))) {
                        zb->setDepthAt(imgpt, curd);
                        //zb(intx, inty) = curd;
                    }
                }
            }
        }
    }

    void DynamicWarpping::initZBuffer() {
        CHECK(!refDepth->getRawData().empty());
        //for each frame, find nearest depth
        char buffer[1024] = {};
        zBuffers.resize((size_t)tWindow);
        for(auto v=0; v<zBuffers.size(); ++v)
            zBuffers[v].reset(new Depth());

        min_depths.resize(zBuffers.size());
        max_depths.resize(zBuffers.size());

        const int width = refDepth->getWidth();
        const int height = refDepth->getHeight();

        auto createZBuffer = [&](int tid, int nThread){
            for(auto i=tid; i<zBuffers.size(); i = i+nThread){
                zBuffers[i]->initialize(width, height, -1);
                updateZBuffer(refDepth.get(), zBuffers[i].get(), sfmModel->getCamera(anchor), sfmModel->getCamera(i+offset));
                utility::computeMinMaxDepth(*(sfmModel.get()), i+offset, min_depths[i], max_depths[i]);
            }
        };

        const int num_thread = 6;
        vector<thread_guard> threads(num_thread);
        for(auto tid=0; tid<num_thread; ++tid){
            std::thread t(createZBuffer, tid, num_thread);
            threads[tid].bind(t);
        }
        for(auto& t: threads)
            t.join();

//	    for(auto i=0; i<zBuffers.size(); ++i){
//		    sprintf(buffer, "%s/temp/zBufferb%05d_%05d.ply", file_io.getDirectory().c_str(), anchor, i+offset);
//		    Mat tex;
//		    cv::resize(images[i],tex,cv::Size(zBuffers[i].getWidth(), zBuffers[i].getHeight()));
//		    printf("Saving point cloud %d\n", i+offset);
//		    utility::saveDepthAsPly(string(buffer), zBuffers[i], tex, sfmModel.getCamera(i+offset), downsample);
//	    }
    }

    void DynamicWarpping::warpToAnchor(const vector<Mat>& images,
                                       const std::vector<std::vector<Eigen::Vector2i> >& segmentsDisplay,
                                       const std::vector<std::vector<Eigen::Vector2i> >& segmentsFlashy,
                                       std::vector<cv::Mat> &output,
                                       const int kFrame) const {
        CHECK_EQ(images.size(), tWindow);
        output.resize((size_t)images.size());
        char buffer[1024] = {};

        const int width = images[0].cols;
        const int height = images[0].rows;

        const theia::Camera &cam1 = sfmModel->getCamera(anchor);

        double dispMargin = 5;
        const double epsilon = 1e-5;
        const double min_visibleRatio = 0.2;

        const int tx = -1;
        const int ty = -1;

        for(auto i=0; i<output.size(); ++i){
            output[i] = images[anchor-offset].clone();
        }

        const Vector3d occlToken(0,0,0);
        const Vector3d outToken(-1,-1,-1);

        auto projectPoint = [&](const Vector3d& spt, const int v){
            //return (-1,-1,-1) if out of bound, (0,0,0) if occluded
            const theia::Camera& cam2 = sfmModel->getCamera(v+offset);
            Vector2d imgpt;
            double d2 = cam2.ProjectPoint(spt.homogeneous(), &imgpt);
            if(d2 < 0)
                return outToken;
            Vector2d dimgpt = imgpt / downsample;
            if(dimgpt[0] < 0 || dimgpt[1] < 0 || dimgpt[0] >= zBuffers[v]->getWidth()-1 || dimgpt[1] >= zBuffers[v]->getHeight()-1)
                return outToken;
            double zdepth = zBuffers[v]->getDepthAt(dimgpt);
            double curdisp = depthToDisp(d2, min_depths[v], max_depths[v]);
            double zdisp = depthToDisp(zdepth, min_depths[v], max_depths[v]);
            if(zdisp - curdisp >= dispMargin)
                return occlToken;
            return interpolation_util::bilinear<uchar,3>(images[v].data, width, height, imgpt);
        };

        const double borderMargin = 0;
        auto threadFuncDisplay = [&](const int tid, const int num_thread){
            for(auto sid=tid; sid<segmentsDisplay.size(); sid+=num_thread){
                const vector<Vector2i>& curSeg = segmentsDisplay[sid];
                const int invalidMargin = (int)curSeg.size() / 50;
                //end frame and start frame
                int startFrame = 0, endFrame = (int)images.size() - 1;
                vector<vector<Vector3d> > segColors(curSeg.size());
                for(auto& segc:segColors)
                    segc.resize(images.size(), Vector3d(0,0,0));
                vector<int> invalidCount(images.size(), 0);
                for(auto i=0; i<curSeg.size(); ++i){
                    Vector2d dsegpt = curSeg[i].cast<double>() / downsample;
                    if(dsegpt[0] < -borderMargin || dsegpt[1] < -borderMargin ||
                       dsegpt[0] >= refDepth->getWidth()-1+borderMargin || dsegpt[1] >= refDepth->getHeight()-1+borderMargin)
                        continue;
                    Vector3d spt = cam1.GetPosition() + refDepth->getDepthAt(dsegpt) * cam1.PixelToUnitDepthRay(curSeg[i].cast<double>());
                    for(auto fid=anchor-offset-1; fid >= 0; --fid){
                        segColors[i][fid] = projectPoint(spt, fid);
                        if((segColors[i][fid]-outToken).norm() < epsilon){
                            invalidCount[fid]++;
                        }
                    }
                    for(auto fid=anchor-offset; fid < images.size(); ++fid){
                        segColors[i][fid] = projectPoint(spt, fid);
                        if((segColors[i][fid]-outToken).norm() < epsilon){
                            invalidCount[fid]++;
                        }
                    }
                }
                for(auto fid=anchor-offset-1; fid>=0; --fid){
                    if(invalidCount[fid] > invalidMargin) {
                        startFrame = std::max(startFrame, fid);
                        break;
                    }
                }
                for(auto fid=anchor-offset; fid < images.size(); ++fid){
                    if(invalidCount[fid] > invalidMargin) {
                        endFrame = std::min(endFrame, fid);
                        break;
                    }
                }
                printf("seg %d, start:%d, end:%d\n", sid, startFrame, endFrame);

                if(endFrame - startFrame < min_visibleRatio * images.size())
                    continue;
                for(auto i=0; i<curSeg.size(); ++i) {
                    for (auto v = 0; v < kFrame; ++v) {
                        if(v == anchor-offset)
                            continue;
                        int id = v % (endFrame-startFrame+1);
                        Vec3b pix;
                        if(segColors[i][id][0] < 0)
                            pix = Vec3b(0,0,0);
                        else
                            pix = Vec3b(segColors[i][id][0], segColors[i][id][1], segColors[i][id][2]);
                        output[v].at<Vec3b>((int)curSeg[i][1], (int)curSeg[i][0]) = pix;
                    }
                }
            }
        };

        const int num_thread = 6;
        vector<thread_guard> threads_display((size_t)num_thread);
        for(auto i=0; i<num_thread; ++i){
            std::thread t(threadFuncDisplay, i, num_thread);
            threads_display[i].bind(t);
        }
        for(auto& t: threads_display)
            t.join();
    }

    void DynamicWarpping::preWarping(std::vector<cv::Mat> &warped,
                                     const bool fullSize,
                                     std::vector<cv::Mat>* visMaps) const {

        vector<Mat> dimages((size_t)tWindow);
        const int nLevel = fullSize ? 1 : (int)std::log2(downsample) + 1;
        for(auto i=0; i<dimages.size(); ++i){
            vector<Mat> pyramid((size_t)nLevel);
            pyramid[0] = imread(file_io.getImage(offset+i));
            for(auto k=1; k<nLevel; ++k)
                pyrDown(pyramid[k-1], pyramid[k]);
            dimages[i] = pyramid.back().clone();
            for(auto y=0; y<dimages[i].rows; ++y){
                for(auto x=0; x<dimages[i].cols; ++x){
                    if(dimages[i].at<Vec3b>(y,x) == Vec3b(0,0,0))
                        dimages[i].at<Vec3b>(y,x) = Vec3b(1,1,1);
                }
            }
        }

        const int dw = dimages[0].cols;
        const int dh = dimages[0].rows;

        warped.resize(dimages.size());

        if(visMaps != nullptr){
            visMaps->resize(dimages.size());
            for(auto& m: *visMaps){
                m.create(dh, dw, CV_8UC1);
                m.setTo(cv::Scalar(Visibility::VISIBLE));
            }
        }

        const theia::Camera& cam1 = sfmModel->getCamera(anchor);
        const int disparity_margin = 2;

#pragma omp parallel for
        for(int i=0; i<dimages.size(); ++i) {
            const theia::Camera &cam2 = sfmModel->getCamera(i + offset);
            //warped[i] = dimages[anchor-offset].clone();
            if (i == anchor - offset) {
                warped[i] = dimages[anchor - offset].clone();
                continue;
            }
            warped[i] = Mat(dh, dw, CV_8UC3, Scalar::all(0));
            for (int y = 0; y < dh; ++y) {
                for (int x = 0; x < dw; ++x) {
                    Vector2d depthPt, imgpt;
                    if(fullSize) {
                        depthPt = Vector2d(x, y) / downsample;
                        imgpt = Vector2d(x,y);
                    }
                    else {
                        imgpt = Vector2d(x,y) * downsample;
                        depthPt = Vector2d(x, y);
                    }
                    depthPt[0] = std::min(depthPt[0], (double)refDepth->getWidth() - 1.0);
                    depthPt[1] = std::min(depthPt[1], (double)refDepth->getHeight() - 1.0);

                    Vector3d ray = cam1.PixelToUnitDepthRay(imgpt);
                    Vector3d spt = cam1.GetPosition() + ray * refDepth->getDepthAt(depthPt);

                    double curd = cam2.ProjectPoint(spt.homogeneous(), &imgpt);
                    if(fullSize)
                        depthPt = imgpt / downsample;
                    else{
                        imgpt = imgpt / downsample;
                        depthPt = imgpt;
                    }
                    if (depthPt[0] >= 0 && depthPt[1] >= 0 && depthPt[0] < zBuffers[i]->getWidth() - 1&&
                        depthPt[1]< zBuffers[i]->getHeight() - 1) {
                        double zDepth = zBuffers[i]->getDepthAt(depthPt);
                        if (zDepth > 0 && curd > 0) {
                            double curdisp = depthToDisp(curd, min_depths[i], max_depths[i]);
                            double zdisp = depthToDisp(zDepth, min_depths[i], max_depths[i]);
                            if (zdisp - curdisp >= disparity_margin) {
                                warped[i].at<Vec3b>(y, x) = Vec3b(0, 0, 0);
                                if(visMaps != nullptr)
                                    (*visMaps)[i].at<uchar>(y,x) = (uchar)Visibility::OCCLUDED;
                            } else {
                                Vector3d pix2 = interpolation_util::bilinear<uchar, 3>(dimages[i].data, dw, dh, imgpt);
                                warped[i].at<Vec3b>(y, x) = Vec3b((uchar) pix2[0], (uchar) pix2[1], (uchar) pix2[2]);
                            }
                        } else {
                            warped[i].at<Vec3b>(y, x) = Vec3b(0, 0, 0);
                            if(visMaps != nullptr)
                                (*visMaps)[i].at<uchar>(y,x) = (uchar)Visibility::OUTSIDE;
                        }
                    }else{
                        if(visMaps != nullptr)
                            (*visMaps)[i].at<uchar>(y,x) = (uchar)Visibility::OUTSIDE;
                    }
                }
            }
        }
    }
}//namespace dynamic_stereo