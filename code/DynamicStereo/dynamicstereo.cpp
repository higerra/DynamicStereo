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
            file_io(file_io_), anchor(anchor_), tWindow(tWindow_), downsample(downsample_), dispResolution(dispResolution_), pR(3),
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
        computeMinMaxDepth();
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

        Vector3d ray1 = cam1.PixelToUnitDepthRay(pt*downsample);
        ray1.normalize();

        imgL = images[id1-offset].clone();
        imgR = images[id2-offset].clone();

        cv::circle(imgL, cv::Point(pt[0], pt[1]), 3, cv::Scalar(0,0,255), 3);

        const double min_depth = 1.0 / max_disp;
        const double max_depth = 1.0 / min_disp;
        printf("min depth:%.3f, max depth:%.3f\n", min_depth, max_depth);

        for(double i=min_depth; i<max_depth; i+=0.1){
            Vector3d curpt = cam1.GetPosition() + ray1 * i;
            Vector4d curpt_homo(curpt[0], curpt[1], curpt[2], 1.0);
            Vector2d imgpt;
            double depth = cam2.ProjectPoint(curpt_homo, &imgpt);
            if(depth < 0)
                continue;
            imgpt = imgpt / (double)downsample;
            //printf("curpt:(%.2f,%.2f,%.2f), Depth:%.3f, pt:(%.2f,%.2f)\n", curpt[0], curpt[1], curpt[2], depth, imgpt[0], imgpt[1]);
            cv::Point cvpt(((int)imgpt[0]), ((int)imgpt[1]));
            cv::circle(imgR, cvpt, 2, cv::Scalar(0,0,255));
        }
    }



    void DynamicStereo::runStereo() {

        //debug for sample patch
//        vector<vector<double> > testP(2);
//        const int tf2 = 3;
//        Vector2d tloc1 = Vector2d(680,387) / downsample;
//        Vector2d tloc2 = Vector2d(100,100) / downsample;
//        MRF_util::samplePatch(images[anchor-offset], tloc1, pR, testP[0]);
//        MRF_util::samplePatch(images[tf2-offset], tloc2, pR, testP[1]);
//        double testncc = MRF_util::medianMatchingCost(testP, 0);
//        cout << "Test ncc: " << testncc << endl;

        initMRF();
        std::shared_ptr<MRF> mrf = createProblem();
        mrf->clearAnswer();

        //randomly initialize
        srand(time(NULL));
        for (auto i = 0; i < width * height; ++i) {
            mrf->setLabel(rand() % dispResolution, 0);
        }

        double initData = (double) mrf->dataEnergy() / MRFRatio;
        double initSmooth = (double) mrf->smoothnessEnergy() / MRFRatio;
        float t;
        cout << "Solving..." << endl << flush;
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
                    refDepth.setDepthAtInt(x, y, l);
            }
        }

        refDepth.updateStatics();
        double max_disp = refDepth.getMaxDepth();
        double min_disp = refDepth.getMinDepth();
//        CHECK_GT(max_depth, 0);
//
//        Depth d2;
//        d2.initialize(width, height, 0.0);
//        for(auto x=0; x<width; ++x){
//            for(auto y=0; y<height; ++y){
//                double curd = refDepth.getDepthAtInt(x,y);
//                d2.setDepthAtInt(x,y,(curd-min_depth) / (max_depth-min_depth) * 255.0);
//            }
//        }
        char buffer[1024] = {};
        sprintf(buffer, "%s/temp/depth%05d_resolution%d.jpg", file_io.getDirectory().c_str(), anchor, dispResolution);
        refDepth.saveImage(buffer);
    }

	void DynamicStereo::warpToAnchor() const{
		cout << "Warpping..." << endl;
		vector<Mat> fullimages(images.size());
		for(auto i=0; i<fullimages.size(); ++i)
			fullimages[i] = imread(file_io.getImage(i+offset));
		vector<Mat> warpped(fullimages.size());

		const int w = fullimages[0].cols;
		const int h = fullimages[0].rows;

		const theia::Camera cam1 = reconstruction.View(anchor)->Camera();

		for(auto i=0; i<fullimages.size(); ++i){
			cout << i+offset << ' ' << flush;
			warpped[i] = fullimages[anchor-offset].clone();
			if(i == anchor-offset)
				continue;
			const theia::Camera cam2 = reconstruction.View(i+offset)->Camera();
			for(auto y=downsample; y<h-downsample; ++y){
				for(auto x=downsample; x<w-downsample; ++x){
					Vector3d ray = cam1.PixelToUnitDepthRay(Vector2d(x,y));
					ray.normalize();
					double depth = refDepth.getDepthAt(Vector2d(x/downsample, y/downsample));
					Vector3d spt = cam1.GetPosition() + ray * depth;
					Vector4d spt_homo(spt[0], spt[1], spt[2], 1.0);
					Vector2d imgpt;
					cam2.ProjectPoint(spt_homo, &imgpt);
					if(imgpt[0] >= 1 && imgpt[1] >= 1 && imgpt[0] < w -1 && imgpt[1] < h - 1){
						Vector3d pix2 = interpolation_util::bilinear<uchar, 3>(fullimages[i].data, w, h, imgpt);
						warpped[i].at<Vec3b>(y,x) = Vec3b(pix2[0], pix2[1], pix2[2]);
					}
				}
			}
		}

		cout << endl << "Saving..." << endl;
		char buffer[1024] = {};
		for(auto i=0; i<warpped.size(); ++i){
			sprintf(buffer, "%s/temp/warpped_b%05d_f%05d.jpg", file_io.getDirectory().c_str(),  anchor, i+offset);
			imwrite(buffer, warpped[i]);
		}
	}
}
