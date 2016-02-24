//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
    namespace MRF_util{
        void samplePatch(const cv::Mat& img, const Vector2d& loc, const int pR, std::vector<double>& pix) {
            const int w = img.cols;
            const int h = img.rows;
            pix.reserve((size_t) 3 * (2 * pR + 1) * (2 * pR + 1));
            int index = 0;
            for (int dx = -1 * pR; dx <= pR; ++dx) {
                for (int dy = -1 * pR; dy <= pR; ++dy, ++index) {
                    Vector2d curloc(loc[0] + dx, loc[1] + dy);
                    if (curloc[0] < 0 || curloc[1] < 0 || curloc[0] >= img.cols - 1 || curloc[1] >= img.rows - 1){
                        pix[index * 3] = -1;
                        pix[index * 3 + 1] = -1;
                        pix[index * 3 + 2] = -1;
                    }
                    Vector3d pv = interpolation_util::bilinear<uchar, 3>(img.data, w, h, curloc);
                    pix[index * 3] = pv[0];
                    pix[index * 3 + 1] = pv[1];
                    pix[index * 3 + 2] = pv[2];
                }
            }
        }
    }//namespace MRF_util

    void DynamicStereo::computeMinMaxDepth() {
        const theia::View* anchorView = reconstruction.View(anchor - offset);
        const theia::Camera cam = anchorView->Camera();
        vector<theia::TrackId> trackIds = anchorView->TrackIds();
        vector<double> depths;
        for(const auto tid: trackIds){
            const theia::Track* t = reconstruction.Track(tid);
            Vector4d spacePt = t->Point();
            Vector2d imgpt;
            double curdepth = cam.ProjectPoint(spacePt, &imgpt);
            if(curdepth > 0)
                depths.push_back(curdepth);
        }

        //ignore furthest 1% and nearest 1% points
        const double lowRatio = 0.01;
        const double highRatio = 0.09;
        const size_t lowKth = (size_t)lowRatio * depths.size();
        const size_t highKth = (size_t)highRatio * depths.size();
        nth_element(depths.begin(), depths.begin() + lowKth, depths.end());
        nth_element(depths.begin(), depths.begin() + highKth, depths.end());

        //min_disp should be correspond to high depth
        min_disp = 1.0 / depths[highKth];
        max_disp = 1.0 / depths[lowKth];
    }

    void DynamicStereo::initMRF() {
        MRF_data.resize((size_t)width * height * depthResolution);
        MRF_smooth.resize((size_t)depthResolution * depthResolution);
        CHECK(!MRF_data.empty() && !MRF_smooth.empty()) << "Can not allocate memory for MRF";

        //init Potts model
        for(auto i=0; i<depthResolution; ++i){
            for(auto j=0; j<depthResolution; ++j){
                if(i == j)
                    MRF_smooth[i*depthResolution+j] = (MRF::CostVal)0 ;
                else
                    MRF_smooth[i*depthResolution+j] = (MRF::CostVal)1;
            }
        }
    }

    void DynamicStereo::assignDataTerm() {
        CHECK_GT(min_disp, 0);
        CHECK_GT(max_disp, 0);
        for(int y=0; y<height; ++y){
            for(int x=0; x<width; ++x){
                for(int d=0; d<depthResolution; ++d){
                    double disp = min_disp + d * (max_disp - min_disp) / (double)depthResolution;
                }
            }
        }
    }

    std::shared_ptr<MRF> DynamicStereo::createProblem() {

    }
}//namespace dynamic_stereo