//
// Created by yanhang on 3/3/16.
//

#include "optimization.h"
#include "external/MRF2.2/mrf.h"
#include "external/MRF2.2/GCoptimization.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{

    FirstOrderOptimize::FirstOrderOptimize(const FileIO& file_io_, const int kFrames_,const cv::Mat& image_, const std::vector<EnergyType> &MRF_data_,
                                           const float MRFRatio_, const int nLabel_, const EnergyType &weight_smooth_):
            StereoOptimization(file_io_, kFrames_, image_, MRF_data_, MRFRatio_, nLabel_), weight_smooth(weight_smooth_){
        assignSmoothWeight();
    }

    void FirstOrderOptimize::optimize(Depth &result, const int max_iter) const {
        EnergyFunction *energy_function = new EnergyFunction(new DataCost(const_cast<EnergyType *>(MRF_data.data())),
                                                             new SmoothnessCost(1, 4, weight_smooth,
                                                                                const_cast<EnergyType *>(hCue.data()),
                                                                                const_cast<EnergyType *>(vCue.data())));
        shared_ptr<MRF> mrf(new Expansion(width, height, nLabel, energy_function));
        mrf->initialize();

        //randomly initialize
        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(0, nLabel - 1);
        for (auto i = 0; i < width * height; ++i)
            mrf->setLabel(i, distribution(generator));

        float initDataEnergy = (float) mrf->dataEnergy() / MRFRatio;
        float initSmoothEnergy = (float) mrf->smoothnessEnergy() / MRFRatio;
        float t;
        mrf->optimize(max_iter, t);
        float finalDataEnergy = (float) mrf->dataEnergy() / MRFRatio;
        float finalSmoothEnergy = (float) mrf->smoothnessEnergy() / MRFRatio;

        printf("Graph cut finished.\nInitial energy: (%.3f, %.3f, %.3f)\nFinal energy: (%.3f,%.3f,%.3f)\nTime usage: %.2fs\n",
               initDataEnergy, initSmoothEnergy, initDataEnergy + initSmoothEnergy,
               finalDataEnergy, finalSmoothEnergy, finalDataEnergy + finalSmoothEnergy, t);

        result.initialize(width, height, -1);
        for(auto i=0; i<width * height; ++i)
            result.setDepthAtInd(i, mrf->getLabel(i));
    }

    double FirstOrderOptimize::evaluateEnergy(const Depth& disp) const {
        return 0.0;
    }

    void FirstOrderOptimize::assignSmoothWeight() {
        const double t = 0.3;
        hCue.resize(width * height, 0);
        vCue.resize(width * height, 0);
        double ratio = 441.0;
        const Mat &img = image;
        for (auto y = 0; y < height; ++y) {
            for (auto x = 0; x < width; ++x) {
                Vec3b pix1 = img.at<Vec3b>(y, x);
                //pixel value range from 0 to 1, not 255!
                Vector3d dpix1 = Vector3d(pix1[0], pix1[1], pix1[2]) / 255.0;
                if (y < height - 1) {
                    Vec3b pix2 = img.at<Vec3b>(y + 1, x);
                    Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]) / 255.0;
                    double diff = (dpix1 - dpix2).norm();
                    if (diff > t)
                        vCue[y * width + x] = 0;
                    else
                        vCue[y * width + x] = (EnergyType) ((diff - t) * (diff - t) * ratio);
                }
                if (x < width - 1) {
                    Vec3b pix2 = img.at<Vec3b>(y, x + 1);
                    Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]) / 255.0;
                    double diff = (dpix1 - dpix2).norm();
                    if (diff > t)
                        hCue[y * width + x] = 0;
                    else
                        hCue[y * width + x] = (EnergyType) ((diff - t) * (diff - t) * ratio);
                }
            }
        }
    }
}//namespace dynamic_stereo