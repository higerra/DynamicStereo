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
    DynamicWarpping::DynamicWarpping(const FileIO &file_io_, const int anchor_, const int tWindow_, const int nLabel_) :
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
                    if ((*zb)(intx, inty) < 0 || zb->getDepthAt(imgpt) >= 0 && curd <= zb->getDepthAt(imgpt)) {
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
        printf("Computing zBuffer...\n");
        zBuffers.resize((size_t)tWindow);
        for(auto v=0; v<zBuffers.size(); ++v)
            zBuffers[v].reset(new Depth());

	    min_depths.resize(zBuffers.size());
	    max_depths.resize(zBuffers.size());
	    cout << endl;

        const int width = refDepth->getWidth();
        const int height = refDepth->getHeight();

	    auto createZBuffer = [&](int tid, int nThread){
		    for(auto i=tid; i<zBuffers.size(); i = i+nThread){
			    zBuffers[i]->initialize(width, height, -1);
//            if (i + offset <= depthind[0]) {
//                updateZBuffer(depths[0], zBuffers[i], sfmModel.getCamera(depthind[0]), sfmModel.getCamera(i + offset));
//                printf("Update zBuffer %d with %d\n", i+offset, depthind[0]);
//            }
//            else if (i + offset >= depthind.back()) {
//                updateZBuffer(depths.back(), zBuffers[i], sfmModel.getCamera(depthind.back()),
//                              sfmModel.getCamera(i + offset));
//                printf("Update zBuffer %d with %d\n", i, depthind.front());
//            } else {
//	            for (auto j = 1; j < depthind.size(); ++j) {
//		            if (i + offset >= depthind[j - 1] && i + offset < depthind[j]) {
//			            updateZBuffer(depths[j - 1], zBuffers[i], sfmModel.getCamera(depthind[j - 1]),
//			                          sfmModel.getCamera(i + offset));
//			            updateZBuffer(depths[j], zBuffers[i], sfmModel.getCamera(depthind[j]),
//                                  sfmModel.getCamera(i + offset));
//			            printf("Update zBuffer %d with %d and %d\n", i + offset, depthind[j - 1], depthind[j]);
//		            }
//	            }
//            }

                updateZBuffer(refDepth.get(), zBuffers[i].get(), sfmModel->getCamera(anchor), sfmModel->getCamera(i+offset));

//	        sprintf(buffer, "%s/temp/zBuffer%05d.ply", file_io.getDirectory().c_str(), i+offset);
//	        Mat dimg;
//	        cv::resize(images[i], dimg, cv::Size(zBuffers[i].getWidth(), zBuffers[i].getHeight()));
//	        utility::saveDepthAsPly(string(buffer), zBuffers[i], dimg, sfmModel.getCamera(i+offset), downsample);
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
                                       const std::vector<std::vector<Eigen::Vector2d> >& segmentsDisplay,
                                       const std::vector<std::vector<Eigen::Vector2d> >& segmentsFlashy,
                                       std::vector<cv::Mat> &output,
                                       const int kFrame) const {
        printf("Warpping...\n");
        CHECK_EQ(images.size(), tWindow);
        output.resize((size_t)images.size());
        char buffer[1024] = {};

        const int width = images[0].cols;
        const int height = images[0].rows;

        const theia::Camera &cam1 = sfmModel->getCamera(anchor);

	    double dispMargin = 10;
        const double epsilon = 1e-5;
        const double min_visibleRatio = 0.4;

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


        auto threadFuncDisplay = [&](const int tid, const int num_thread){
            for(auto sid=tid; sid<segmentsDisplay.size(); sid+=num_thread){
                printf("Segment %d(%d) on thread %d\n", sid, (int)segmentsDisplay.size(), tid);
                const vector<Vector2d>& curSeg = segmentsDisplay[sid];
                //end frame and start frame
                int startFrame = 0, endFrame = (int)images.size() - 1;
                vector<vector<Vector3d> > segColors(curSeg.size());
                for(auto& segc:segColors)
                    segc.resize(images.size(), Vector3d(0,0,0));
                for(auto i=0; i<curSeg.size(); ++i){
                    Vector2d dsegpt = curSeg[i] / downsample;
                    if(dsegpt[0] < 0 || dsegpt[1] < 0 || dsegpt[0] >= refDepth->getWidth()-1 || dsegpt[1] >= refDepth->getHeight()-1)
                        continue;
                    Vector3d spt = cam1.GetPosition() + refDepth->getDepthAt(dsegpt) * cam1.PixelToUnitDepthRay(curSeg[i]);
                    for(auto fid=anchor-offset-1; fid >= 0; --fid){
                        segColors[i][fid] = projectPoint(spt, fid);
                        if((segColors[i][fid]-outToken).norm() < epsilon){
                            startFrame = std::max(startFrame, fid);
                            break;
                        }
                    }
                    for(auto fid=anchor-offset; fid < images.size(); ++fid){
                        segColors[i][fid] = projectPoint(spt, fid);
                        if((segColors[i][fid]-outToken).norm() < epsilon){
                            endFrame = std::min(endFrame, fid);
                            break;
                        }
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
//        auto threadFuncFlashy = [&](const int tid, const int num_thread){
//
//        };

    }

    void DynamicWarpping::preWarping(const cv::Mat &mask, std::vector<cv::Mat> &warped) const {

        vector<Mat> dimages((size_t)tWindow);
        const int nLevel = (int)std::log2(downsample) + 1;
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
        CHECK_EQ(refDepth->getWidth(), dw);
        CHECK_EQ(refDepth->getHeight(), dh);

        Mat maskd;
        cv::resize(mask, maskd, cv::Size(dw, dh), 0, 0, INTER_NEAREST);

        warped.resize(dimages.size());

        const theia::Camera& cam1 = sfmModel->getCamera(anchor);
        const int disparity_margin = 10;

#pragma omp parallel for
        for(int i=0; i<dimages.size(); ++i) {
            cout << i+offset << ' ' << flush;
            const theia::Camera& cam2 = sfmModel->getCamera(i+offset);
            warped[i] = dimages[anchor-offset].clone();
            if(i == anchor - offset){
                continue;
            }
            for (int y = 0; y < dh; ++y) {
                for (int x = 0; x < dw; ++x) {
                    if (maskd.at<uchar>(y, x) < 200) {
                        warped[i].at<Vec3b>(y,x) = dimages[anchor-offset].at<Vec3b>(y,x);
                        continue;
                    }
                    Vector3d ray = cam1.PixelToUnitDepthRay(Vector2d(x*downsample, y*downsample));
                    Vector3d spt = cam1.GetPosition() + ray * refDepth->operator()(x,y);
                    Vector2d imgpt;
                    double curd = cam2.ProjectPoint(spt.homogeneous(), &imgpt);
                    imgpt = imgpt / downsample;
                    if(imgpt[0]>=0 && imgpt[1] >= 0 && imgpt[0] < dw-1 && imgpt[1] < dh-1){
                        double zDepth = zBuffers[i]->getDepthAt(imgpt);
                        if(zDepth > 0 && curd > 0){
                            double curdisp = depthToDisp(curd, min_depths[i], max_depths[i]);
                            double zdisp = depthToDisp(zDepth, min_depths[i], max_depths[i]);
                            if(zdisp - curdisp >= disparity_margin) {
                                warped[i].at<Vec3b>(y,x) = Vec3b(0,0,0);
                            }else{
                                Vector3d pix2 = interpolation_util::bilinear<uchar,3>(dimages[i].data, dw, dh, imgpt);
                                warped[i].at<Vec3b>(y,x) = Vec3b((uchar)pix2[0], (uchar)pix2[1], (uchar)pix2[2]);
                            }
                        }else {
                            warped[i].at<Vec3b>(y,x) = Vec3b(0,0,0);
                        }
                    }
                }
            }
        }
        cout << "done" << endl;
    }
}//namespace dynamic_stereo