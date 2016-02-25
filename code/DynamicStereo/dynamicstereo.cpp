//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"

using namespace std;
using namespace cv;
using namespace Eigen;
namespace dynamic_stereo{
    DynamicStereo::DynamicStereo(const dynamic_stereo::FileIO &file_io_, const int anchor_,
                                 const int tWindow_, const int downsample_, const double weight_smooth_, const int dispResolution_):
            file_io(file_io_), anchor(anchor_), tWindow(tWindow_), downsample(downsample_), dispResolution(dispResolution_), pR(2),
            weight_smooth(weight_smooth_), MRFRatio(1000),
            min_disp(-1), max_disp(-1), dispScale(1000){

        cout << "Reading..." << endl;
        offset = anchor >= tWindow / 2 ? anchor - tWindow / 2 : 0;
        CHECK_GT(file_io.getTotalNum(), offset + tWindow);
        CHECK(theia::ReadReconstruction(file_io.getReconstruction(), &reconstruction)) << "Can not open reconstruction file";
        CHECK_EQ(reconstruction.NumViews(), file_io.getTotalNum());
        CHECK(downsample == 1 || downsample == 2 || downsample == 4 || downsample == 8) << "Invalid downsample ratio!";
        images.resize((size_t)tWindow);

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
        CHECK_LT(pt[0], (double)width * downsample);
        CHECK_LT(pt[1], (double)height * downsample);

        theia::Camera cam1 = reconstruction.View(id1)->Camera();
        theia::Camera cam2 = reconstruction.View(id2)->Camera();

        Vector3d ray1 = cam1.PixelToUnitDepthRay(pt);
        ray1.normalize();

        cv::resize(images[id1-offset], imgL, cv::Size(width*downsample, height*downsample));
        cv::resize(images[id2-offset], imgR, cv::Size(width*downsample, height*downsample));

        cv::circle(imgL, cv::Point(pt[0], pt[1]), 3, cv::Scalar(0,0,255), 3);

        for(double i=0; i<10000; i+=0.1){
            Vector3d curpt = cam1.GetPosition() + ray1 * i;
            Vector4d curpt_homo(curpt[0], curpt[1], curpt[2], 1.0);
            Vector2d imgpt;
            double depth = cam2.ProjectPoint(curpt_homo, &imgpt);
            if(depth < 0)
                continue;
            //printf("curpt:(%.2f,%.2f,%.2f), Depth:%.3f, pt:(%.2f,%.2f)\n", curpt[0], curpt[1], curpt[2], depth, imgpt[0], imgpt[1]);
            cv::Point cvpt(((int)imgpt[0]), ((int)imgpt[1]));
            cv::circle(imgR, cvpt, 2, cv::Scalar(0,0,255));
        }
    }



    void DynamicStereo::runStereo() {
        initMRF();
        std::shared_ptr<MRF> mrf = createProblem();
        mrf->clearAnswer();
        for (auto i = 0; i < width * height; ++i)
            mrf->setLabel(i, 0);
        double initData = (double) mrf->dataEnergy() / MRFRatio;
        double initSmooth = (double) mrf->smoothnessEnergy() / MRFRatio;
        float t;
        printf("Solving...");
        mrf->optimize(10, t);

        double finalData = (double) mrf->dataEnergy() / MRFRatio;
        double finalSmooth = (double) mrf->smoothnessEnergy() / MRFRatio;
        printf("Done.\n Init energy:(%.3f,%.3f,%.3f), final energy: (%.3f,%.3f,%.3f), time usage: %.2f\n", initData,
               initSmooth, initData + initSmooth,
               finalData, finalSmooth, finalData + finalSmooth, t);

        //assign depth to depthmap
        const double epsilon = 0.00001;
        for(auto x=0; x<width; ++x) {
            for (auto y = 0; y < height; ++y) {
                double l = (double) mrf->getLabel(y * width + x);
                double disp = min_disp + l * (max_disp - min_disp) / (double) dispResolution;
                if(disp < epsilon)
                    refDepth.setDepthAtInt(x, y, -1);
                else
                    refDepth.setDepthAtInt(x, y, 1.0 / disp);
            }
        }

        refDepth.updateStatics();
        double max_depth = refDepth.getMaxDepth();
	double min_depth = refDepth.getMinDepth();
        CHECK_GT(max_depth, 0);

	Depth d2;
	d2.initialize(width, height, 0.0);
	for(auto x=0; x<width; ++x){
	    for(auto y=0; y<height; ++y){
		double curd = refDepth.getDepthAtInt(x,y);
		d2.setDepthAtInt(x,y,(curd-min_depth) / (max_depth-min_depth) * 255.0);
	    }
	}
        char buffer[1024] = {};
        sprintf(buffer, "%s/temp/depth%05d.jpg", file_io.getDirectory().c_str(), anchor);
        d2.saveImage(buffer);
    }
}
