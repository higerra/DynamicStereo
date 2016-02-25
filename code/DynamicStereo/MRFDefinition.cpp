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
                    if (curloc[0] < 0 || curloc[1] < 0 || curloc[0] >= w - 1 || curloc[1] >= h - 1){
                        pix[index * 3] = -1;
                        pix[index * 3 + 1] = -1;
                        pix[index * 3 + 2] = -1;
                        continue;
                    }
                    Vector3d pv = interpolation_util::bilinear<uchar, 3>(img.data, w, h, curloc);
                    pix[index * 3] = pv[0];
                    pix[index * 3 + 1] = pv[1];
                    pix[index * 3 + 2] = pv[2];
                }
            }
        }

        double medianMatchingCost(const vector<vector<double> >& patches, const int refId){
            CHECK_GE(refId, 0);
            CHECK_LT(refId, patches.size());
            const vector<double>& pRef = patches[refId];
            vector<double> mCost;
            mCost.reserve(patches.size()- 1);
            for(auto i=0; i<patches.size(); ++i){
                if(i == refId)
                    continue;
                vector<double> p1, p2;
                for(auto j=0; j<pRef.size(); ++j){
                    if(pRef[j] >= 0 && patches[i][j] >= 0){
                        p1.push_back(pRef[j]);
                        p2.push_back(patches[i][j]);
                    }
                }
                if(p1.empty() || p2.empty())
                    continue;
                mCost.push_back(math_util::normalizedCrossCorrelation(p1,p2));
            }
            size_t kth = mCost.size() / 2;
            nth_element(mCost.begin(), mCost.begin() + kth, mCost.end());
            return mCost[kth];
        }

    }//namespace MRF_util

    void DynamicStereo::computeMinMaxDepth() {
        const theia::View* anchorView = reconstruction.View(anchor);
        const theia::Camera cam = anchorView->Camera();
        vector<theia::TrackId> trackIds = anchorView->TrackIds();
        vector<double> disps;
        for(const auto tid: trackIds){
            const theia::Track* t = reconstruction.Track(tid);
            Vector4d spacePt = t->Point();
            Vector2d imgpt;
            double curdepth = cam.ProjectPoint(spacePt, &imgpt);
            if(curdepth > 0)
                disps.push_back(1.0 / curdepth);
        }
        cout << "compute min max depth:" << endl;
        for(auto d: disps)
            cout << d << ' ';
        cout << endl;
        //ignore furthest 1% and nearest 1% points
        const double lowRatio = 0.01;
        const double highRatio = 0.09;
        const size_t lowKth = (size_t)(lowRatio * disps.size());
        const size_t highKth = (size_t)highRatio * disps.size();
        printf("lowKth: %d, highKth: %d\n", (int)lowKth, (int)highKth);
        //min_disp should be correspond to high depth
        nth_element(disps.begin(), disps.begin() + lowKth, disps.end());
        min_disp = disps[lowKth];
        nth_element(disps.begin(), disps.begin() + highKth, disps.end());
        max_disp = disps[highKth];
    }

    void DynamicStereo::initMRF() {
        MRF_data.resize((size_t)width * height * dispResolution);
        MRF_smooth.resize((size_t)dispResolution * dispResolution);
        CHECK(!MRF_data.empty() && !MRF_smooth.empty()) << "Can not allocate memory for MRF";

//        //init Potts model
//        for(auto i=0; i<dispResolution; ++i){
//            for(auto j=0; j<dispResolution; ++j){
//                if(i == j)
//                    MRF_smooth[i*dispResolution+j] = (MRF::CostVal)0 ;
//                else
//                    MRF_smooth[i*dispResolution+j] = (MRF::CostVal)1;
//            }
//        }
        computeMinMaxDepth();
        cout << "Assigning data term..." << endl << flush;
        assignDataTerm();
        cout << "Assigning smoothness weight..." << endl << flush;
        assignSmoothWeight();
    }

    void DynamicStereo::assignDataTerm() {
        CHECK_GT(min_disp, 0);
        CHECK_GT(max_disp, 0);
        theia::Camera cam1 = reconstruction.View(anchor)->Camera();

        int index = 0;
        int unit = width * height / 10;
        //Be careful about down sample ratio!!!!!
        for(int y=0; y<height; ++y){
            for(int x=0; x<width; ++x, ++index){
                if(index % unit == 0)
                    cout << '.' << flush;
                Vector3d ray = cam1.PixelToUnitDepthRay(Vector2d(x*downsample, y*downsample));
                ray.normalize();
//#pragma omp parallel for
                for(int d=0; d<dispResolution; ++d){
                    //compute 3D point
                    bool verbose = false;
                    cout << "-----------------------" << endl << flush;
                    printf("(%d,%d,%d)\n", x, y, d);
                    printf("min_disp: %.2f, max_disp: %.2f\n", min_disp, max_disp);
                    double disp = min_disp + d * (max_disp - min_disp) / (double)dispResolution;
                    cout << "Disparity:" << disp << endl << flush;
                    Vector3d spt = cam1.GetPosition() + ray * 1.0 / disp;
                    printf("Space point: (%.2f,%.2f,%.2f)\n", spt[0], spt[1], spt[2]);
                    Vector4d spt_homo(spt[0], spt[1], spt[2], 1.0);

                    //project onto other views and compute matching cost
                    vector<vector<double> > patches(images.size());
                    int validCount = 0;
                    for(auto v=0; v<images.size(); ++v){
                        theia::Camera cam2 = reconstruction.View(v+offset)->Camera();
                        Vector2d imgpt;
                        cam2.ProjectPoint(spt_homo, &imgpt);
                        imgpt = imgpt / (double)downsample;
                        if(imgpt[0] > 5 && imgpt[1] > 5 && imgpt[0] < width-5 && imgpt[1] < height-5)
                            validCount++;
                        //TODO: shifting window

                        MRF_util::samplePatch(images[v], imgpt, pR, patches[v]);
                    }

                    //compute the matching cost starting from each images
                    vector<double> mCostGroup(images.size()); //matching cost
                    for(auto v=0; v<patches.size(); ++v)
                        mCostGroup.push_back(MRF_util::medianMatchingCost(patches, v));
                    size_t kth = mCostGroup.size() / 2;
                    nth_element(mCostGroup.begin(), mCostGroup.begin()+kth, mCostGroup.end());
                    MRF_data[dispResolution * (y*width+x) + d] = (int)((mCostGroup[kth] - 1) * (mCostGroup[kth] - 1) * MRFRatio);
//#pragma omp critical
                    if(validCount >= images.size() / 3){
                        cout << "Patch: " << endl;
                        for(const auto& pat: patches){
                            for(auto pv: pat)
                                cout << pv << ' ';
                            cout << endl;
                        }
                        for(auto cv: mCostGroup)
                            cout << cv << ' ';
                        cout << endl;
                        printf("mCostGroup:%.3f, dataCost:%d\n",mCostGroup[kth],
                               MRF_data[dispResolution * (y * width + x) + d]);
                        getchar();
                    }
                }
            }
        }
        cout << "done" << endl;
    }

    void DynamicStereo::assignSmoothWeight() {
        //mean of the gaussian distribution of gradient for edge
        const double t = 0.3;
        hCue.resize(width * height);
        vCue.resize(width * height);
        const Mat& img = images[anchor-offset];
        for(auto y=0; y<height; ++y) {
            for (auto x = 0; x < width; ++x) {
                Vec3b pix1 = img.at<Vec3b>(y,x);
                //pixel value range from 0 to 1, not 255!
                Vector3d dpix1 = Vector3d(pix1[0], pix1[1], pix1[2]) / 255.0;
                if(y < height - 1){
                    Vec3b pix2 = img.at<Vec3b>(y+1,x);
                    Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]) / 255.0;
                    double diff = (dpix1 - dpix2).norm();
                    if(diff > t)
                        hCue[y*width+x] = 0;
                    else
                        hCue[y*width+x] = (MRF::CostVal) ((diff - t) * (diff - t) * weight_smooth * MRFRatio);
                }
                if(x < width - 1){
                    Vec3b pix2 = img.at<Vec3b>(y,x+1);
                    Vector3d dpix2(pix2[0], pix2[1], pix2[2]);
                    double diff = (dpix1 - dpix2).norm();
                    if(diff > t)
                        hCue[y*width+x] = 0;
                    else
                        hCue[y*width+x] = (MRF::CostVal) ((diff - t) * (diff - t) * weight_smooth * MRFRatio);
                }
            }
        }
    }

    std::shared_ptr<MRF> DynamicStereo::createProblem() {

        //use truncated linear cost for smoothness
        EnergyFunction *energy_function = new EnergyFunction(new DataCost(MRF_data.data()),
                                                             new SmoothnessCost(1, 2, (MRF::CostVal)(weight_smooth * MRFRatio), hCue.data(), vCue.data()));
        shared_ptr<MRF> mrf(new Expansion(width, height, dispResolution, energy_function));
        mrf->initialize();

        return mrf;
    }
}//namespace dynamic_stereo