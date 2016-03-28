//
// Created by yanhang on 3/28/16.
//

#include "gridwarpping.h"
using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo {
    GridWarpping::GridWarpping(const FileIO &file_io_, const std::vector<cv::Mat> &images_,
                               const StereoModel<EnergyType> &model_,
                               const theia::Reconstruction &reconstruction_, const int downsample_, const int offset_,
                               const int gw, const int gh) :
            file_io(file_io_), images(images_), model(model_), reconstruction(reconstruction_),
            offset(offset_), downsample(downsample_), gridW(gw), gridH(gh){
        CHECK(!images.empty());
        width = images[0].cols;
        height = images[0].rows;
        CHECK_EQ(width, model.width * downsample);
        CHECK_EQ(height, model.height * downsample);
        blockW = (double) width / gridW;
        blockH = (double) height / gridH;
        gridLoc.resize((size_t) (gridW + 1) * (gridH + 1));
        for (auto x = 0; x <= gridW; ++x) {
            for (auto y = 0; y <= gridH; ++y) {
                gridLoc[y * (gridW + 1) + x] = Eigen::Vector2d(blockW * x, blockH * y);
                if(x == gridW)
                    gridLoc[y * (gridW + 1) + x][0] -= 1.1;
                if(y == gridH)
                    gridLoc[y * (gridW + 1) + x][1] -= 1.1;
                printf("%.2f,%.2f\n", gridLoc[y * (gridW + 1) + x][0], gridLoc[y * (gridW + 1) + x][1]);
            }
        }
    }

    void GridWarpping::getGridIndAndWeight(const Eigen::Vector2d &pt, Eigen::Vector4i &ind,
                                           Eigen::Vector4d &w) const {
        CHECK_LT(pt[0], width - 1);
        CHECK_LT(pt[1], height - 1);
        int x = (int)floor(pt[0] / blockW);
        int y = (int)floor(pt[1] / blockH);

        //////////////
        // 1--2
        // |  |
        // 4--3
        /////////////
        ind = Vector4i(y*(gridW+1)+x, y*(gridW+1)+x+1, (y+1)*(gridW+1)+x+1, (y+1)*(gridW+1)+x);
        for(auto i=0; i<4; ++i){
            CHECK_LT(ind[i], gridLoc.size());
            w[i] = (pt - gridLoc[ind[i]]).norm();
        }
    }
} //namespace dynamic_stereo