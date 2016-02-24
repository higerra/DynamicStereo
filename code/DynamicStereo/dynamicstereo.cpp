//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"

using namespace std;
using namespace cv;
using namespace Eigen;
namespace dynamic_stereo{
    DynamicStereo::DynamicStereo(const dynamic_stereo::FileIO &file_io_, const int anchor_,
                                 const int tWindow_, const int downsample_):
            file_io(file_io_), anchor(anchor_), tWindow(tWindow_), downsample(downsample_), depthResolution(128), pR(5),
            min_disp(-1), max_disp(-1){

        offset = anchor >= tWindow / 2 ? anchor - tWindow / 2 : 0;
        CHECK_GT(file_io.getTotalNum(), offset + tWindow);
        CHECK(theia::ReadReconstruction(file_io.getReconstruction(), &reconstruction)) << "Can not open reconstruction file";
        CHECK_EQ(reconstruction.NumViews(), file_io.getTotalNum());
        CHECK(downsample == 1 || downsample == 2 || downsample == 4 || downsample == 8) << "Invalid downsample ratio!";
        images.resize((size_t)tWindow);
        cout << "Reading..." << endl;


        for(auto i=0; i<tWindow; ++i){
            Mat tempMat = imread(file_io.getImage(i + offset));
            cv::Size dsize(tempMat.cols / downsample, tempMat.rows / downsample);
            pyrDown(tempMat, images[i], dsize);
        }
        CHECK_GT(images.size(), 2) << "Too few images";
        width = images.front().cols;
        height = images.front().rows;

        refDepth.initialize(width, height,0.0);
    }

    void DynamicStereo::verifyEpipolarGeometry(const int id1, const int id2,
                                               const Eigen::Vector2d& pt,
                                               cv::Mat &imgL, cv::Mat &imgR) {
        CHECK_GE(id1 - offset, 0);
        CHECK_GE(id2 - offset, 0);
        CHECK_LT(id1 - offset, images.size());
        CHECK_LT(id2 - offset, images.size());
        CHECK_GE(pt[0], 0);
        CHECK_GE(pt[1], 0);
        CHECK_LT(pt[0], (double)width);
        CHECK_LT(pt[1], (double)height);

        theia::Camera cam1 = reconstruction.View(id1 - offset)->Camera();
        theia::Camera cam2 = reconstruction.View(id2 - offset)->Camera();

        Vector3d ray1 = cam1.PixelToUnitDepthRay(pt);
        ray1.normalize();

        imgL = images[id1-offset].clone();
        imgR = images[id2-offset].clone();

        cv::circle(imgL, cv::Point(pt[0], pt[1]), 3, cv::Scalar(0,0,255), 2);

        for(double i=0; i<10000; i+=0.1){
            Vector3d curpt = cam1.GetPosition() + ray1 * i;
            Vector4d curpt_homo(curpt[0], curpt[1], curpt[2], 1.0);
            Vector2d imgpt;
            double depth = cam2.ProjectPoint(curpt_homo, &imgpt);
            //printf("curpt:(%.2f,%.2f,%.2f), Depth:%.3f, pt:(%.2f,%.2f)\n", curpt[0], curpt[1], curpt[2], depth, imgpt[0], imgpt[1]);
            cv::circle(imgR, cv::Point(imgpt[0], imgpt[1]), 3, cv::Scalar(0,0,255));
        }
    }



    void DynamicStereo::runStereo() {

    }
}