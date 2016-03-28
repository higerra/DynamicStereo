//
// Created by yanhang on 3/24/16.
//

#include <stlplus3/file_system.hpp>
#include "lineSeg.h"

using namespace std;
using namespace cv;
using namespace Eigen;
namespace dynamic_stereo{

    int Line::sampleNum = 100;

    void Line::extractHist(const cv::Mat &img) {

    }


    LineSeg::LineSeg(const FileIO &file_io_, const int anchor_, const int tWindow):file_io(file_io_), anchor(anchor_) {
        CHECK(theia::ReadReconstruction(file_io.getReconstruction(), &reconstruction)) << "Run SfM first";

        offset = anchor >= tWindow / 2 ? anchor - tWindow / 2 : 0;
        CHECK_LT(file_io.getTotalNum(), offset + tWindow);

        images.resize((size_t)tWindow);
        for(auto i=0; i<tWindow; ++i)
            images[i] = imread(file_io.getImage(i+offset));
        CHECK(!images.empty());
        width = images[0].cols;
        height = images[0].rows;


        const vector<theia::ViewId>& vids = reconstruction.ViewIds();
        orderedId.resize(vids.size());
        for(auto i=0; i<vids.size(); ++i) {
            const theia::View* v = reconstruction.View(vids[i]);
            std::string nstr = v->Name().substr(5,5);
            int idx = atoi(nstr.c_str());
            orderedId[i] = IdPair(idx, vids[i]);
        }
        std::sort(orderedId.begin(), orderedId.end(),
                  [](const std::pair<int, theia::ViewId>& v1, const std::pair<int, theia::ViewId>& v2){return v1.first < v2.first;});
    }

    void LineSeg::undistort(const cv::Mat& input, cv::Mat &output, const theia::Camera& cam) const {
        Matrix3d K;
        cam.GetCalibrationMatrix(&K);
        double d1 = cam.RadialDistortion1();
        double d2 = cam.RadialDistortion2();
        output = Mat(input.size(), CV_8UC3);
        for(auto y=0; y<input.rows; ++y){
            for(auto x=0; x<input.cols; ++x){
                Vector3d imgptH(x,y,1.0);
                Vector3d camptH = K.inverse() * imgptH;
                Vector2d campt(camptH[0]/camptH[2], camptH[1]/camptH[2]);
                double disx, disy;
                theia::RadialDistortPoint<double>(campt[0], campt[1], d1,d2,&disx,&disy);
                imgptH = K * Vector3d(disx,disy,1.0);
                Vector2d imgpt(imgptH[0]/imgptH[2], imgptH[1]/imgptH[2]);
                if(imgpt[0] > 0 && imgpt[1] > 0 && imgpt[0] < output.cols -1 && imgpt[1] < output.rows - 1) {
                    Vector3d pix = interpolation_util::bilinear<uchar, 3>(input.data, input.cols,
                                                                          input.rows, imgpt);
                    output.at<Vec3b>(y,x) = Vec3b((uchar)pix[0], (uchar)pix[1], (uchar)pix[2]);
                }
            }
        }
    }

    void LineSeg::runLSD() {
        char buffer[1024] = {};
        const double min_length = 30;

        for(auto i=0; i<file_io.getTotalNum(); ++i){
//            printf("Undistoring frame %d\n", i);
//            Mat undis;
//            const theia::Camera& cam = reconstruction.View(orderedId[i].second)->Camera();
//            undistort(images[i], undis, cam);
//            sprintf(buffer, "%s/temp/undistored%05d.jpg", file_io.getDirectory().c_str(), i);
//            imwrite(buffer, undis);
            Mat img = imread(file_io.getImage(i));
            printf("Detecting line on frame %d\n", i);
            Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
            Mat grayimg;
            cvtColor(img, grayimg, CV_RGB2GRAY);
            vector<Vec4f> templines;
            ls->detect(grayimg, templines);
            Mat outImg = img.clone();
            for(auto l: templines){
                Vector2d startpt(l[0], l[1]);
                Vector2d endpt(l[2], l[3]);
                double length = (endpt - startpt).norm();
                if(length < min_length)
                    continue;
                cv::line(outImg, cv::Point(startpt[0], startpt[1]), cv::Point(endpt[0], endpt[1]), Scalar(0,0,255), 2);
            }
            sprintf(buffer, "%s/line", file_io.getDirectory().c_str());
            if(!stlplus::folder_exists(string(buffer)))
                stlplus::folder_create(string(buffer));
            sprintf(buffer, "%s/line/line%05d.jpg", file_io.getDirectory().c_str(), i);
            imwrite(buffer, outImg);
        }
    }
}//namespace dynamic_stereo