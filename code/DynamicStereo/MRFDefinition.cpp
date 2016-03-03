//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"
#include "external/MRF2.2/GCoptimization.h"
using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo {
    namespace MRF_util {
        void samplePatch(const cv::Mat &img, const Vector2d &loc, const int pR, std::vector<double> &pix) {
            const int w = img.cols;
            const int h = img.rows;
            pix.resize((size_t) 3 * (2 * pR + 1) * (2 * pR + 1));
            int index = 0;
            for (int dx = -1 * pR; dx <= pR; ++dx) {
                for (int dy = -1 * pR; dy <= pR; ++dy, ++index) {
                    Vector2d curloc(loc[0] + dx, loc[1] + dy);
                    if (curloc[0] < 0 || curloc[1] < 0 || curloc[0] >= w - 1 || curloc[1] >= h - 1) {
                        pix[index * 3] = -1;
                        pix[index * 3 + 1] = -1;
                        pix[index * 3 + 2] = -1;
                    } else {
                        Vector3d pv = interpolation_util::bilinear<uchar, 3>(img.data, w, h, curloc);
                        pix[index * 3] = pv[0];
                        pix[index * 3 + 1] = pv[1];
                        pix[index * 3 + 2] = pv[2];
                    }
                }
            }
        }

        void getNCCArray(const vector<vector<double> >& patches, const int refId, vector<double>& mCost){
            const vector<double> &pRef = patches[refId];
            mCost.reserve(patches.size() - 1);
            for (auto i = 0; i < patches.size(); ++i) {
                if (i == refId)
                    continue;
                vector<double> p1, p2;
                for (auto j = 0; j < pRef.size(); ++j) {
                    if (pRef[j] >= 0 && patches[i][j] >= 0) {
                        p1.push_back(pRef[j]);
                        p2.push_back(patches[i][j]);
                    }
                }
                if (p1.size() < pRef.size() / 2)
                    continue;
                mCost.push_back(math_util::normalizedCrossCorrelation(p1, p2));
            }
        }

        void getSSDArray(const vector<vector<double> >& patches, const int refId, vector<double>& mCost){
            const vector<double> &pRef = patches[refId];
            mCost.reserve(patches.size() - 1);

            for (auto i = 0; i < patches.size(); ++i) {
                if (i == refId)
                    continue;
                vector<double> p1, p2;
                for (auto j = 0; j < pRef.size(); ++j) {
                    if (pRef[j] >= 0 && patches[i][j] >= 0) {
                        p1.push_back(pRef[j]);
                        p2.push_back(patches[i][j]);
                    }
                }
                if (p1.size() < pRef.size() / 2)
                    continue;
                double ssd = 0.0;
                for(auto j=0; j<p1.size(); ++j)
                    ssd += (p1[j]-p2[j]) * (p1[j]-p2[j]);
                mCost.push_back(ssd / (double)p1.size());
            }
        }

        double medianMatchingCost(const vector<vector<double> > &patches, const int refId) {
            CHECK_GE(refId, 0);
            CHECK_LT(refId, patches.size());
            vector<double> mCost;
            getSSDArray(patches, refId, mCost);
            //if the patch is not visible in >50% frames, assign large penalty.
            if (mCost.size() < patches.size() / 2)
                return -1;
            size_t kth = mCost.size() / 2;
            nth_element(mCost.begin(), mCost.begin() + kth, mCost.end());
            return mCost[kth];
        }

        double sumMatchingCostHalf(const vector<vector<double> >& patches, const int refId){
            CHECK_GE(refId, 0);
            CHECK_LT(refId, patches.size());
            const double theta = 90;
            auto phid = [theta](const double v){
                return -1 * std::log2(1 + std::exp(-1 * v / theta));
            };
            vector<double> mCost;
            getSSDArray(patches, refId, mCost);
            //if the patch is not visible in >50% frames, assign large penalty.
            if (mCost.size() < 2)
                return 1;
            if(mCost.size() == 2)
                return std::min(phid(mCost[0]), phid(mCost[1]));
            //sum of best half
            sort(mCost.begin(), mCost.end(), [](double x1, double x2){return x1 <= x2;});
            const size_t kth = mCost.size() / 2;
            double res = 0.0;

            for(auto i=0; i<kth; ++i){
                res += phid(mCost[i]);
            }
            return res / (double)kth;
        }

    }//namespace MRF_util


    void DynamicStereo::computeMinMaxDisparity(){
        if(min_disp > 0 && max_disp > 0)
            return;
        if(reconstruction.NumTracks() == 0){
            CHECK_GT(min_disp, 0) << "Please specify the minimum disparity";
            CHECK_GT(max_disp, 0) << "Please specify the minimum disparity";
            return;
        }
        const theia::View *anchorView = reconstruction.View(anchor);
        const theia::Camera cam = anchorView->Camera();
        vector<theia::TrackId> trackIds = anchorView->TrackIds();
        vector<double> disps;
        for (const auto tid: trackIds) {
            const theia::Track *t = reconstruction.Track(tid);
            Vector4d spacePt = t->Point();
            Vector2d imgpt;
            double curdepth = cam.ProjectPoint(spacePt, &imgpt);
            if (curdepth > 0)
                disps.push_back(1.0 / curdepth);
        }
        //ignore furthest 1% and nearest 1% points
        const double lowRatio = 0.01;
        const double highRatio = 0.99;
        const size_t lowKth = (size_t) (lowRatio * disps.size());
        const size_t highKth = (size_t) (highRatio * disps.size());
        //min_disp should be correspond to high depth
        nth_element(disps.begin(), disps.begin() + lowKth, disps.end());
        min_disp = disps[lowKth];
        nth_element(disps.begin(), disps.begin() + highKth, disps.end());
        max_disp = disps[highKth];
    }

    void DynamicStereo::initMRF() {
        MRF_data.resize((size_t) width * height * dispResolution);
        CHECK(!MRF_data.empty()) << "Can not allocate memory for MRF";

        cout << "Assigning data term..." << endl << flush;
        assignDataTerm();
    }

    void DynamicStereo::assignDataTerm() {
	    CHECK_GT(min_disp, 0);
	    CHECK_GT(max_disp, 0);

        const int tx = 195 / downsample;
        const int ty = 127 / downsample;

	    //read from cache
	    char buffer[1024] = {};
	    sprintf(buffer, "%s/temp/cacheMRFdata", file_io.getDirectory().c_str());
	    ifstream fin(buffer, ios::binary);

        bool recompute = true;
	    if (fin.is_open()) {
            int frame, resolution, tw, ds, type;
            double mindisp, maxdisp;
            fin.read((char *) &frame, sizeof(int));
            fin.read((char *) &resolution, sizeof(int));
            fin.read((char *) &tw, sizeof(int));
            fin.read((char *) &ds, sizeof(int));
            fin.read((char *) &type, sizeof(int));
            fin.read((char *) &mindisp, sizeof(double));
            fin.read((char *) &maxdisp, sizeof(double));
            printf("Cached data: anchor:%d, resolution:%d, twindow:%d, downsample:%d, Energytype:%d, min_disp:%.5f, max_disp:%.5f\n",
                   frame, resolution, tw, ds, type, mindisp, maxdisp);
            if (frame == anchor && resolution == dispResolution && tw == tWindow &&
                type == sizeof(EnergyType) && ds == downsample && min_disp == mindisp && max_disp == maxdisp) {
                printf("Reading unary term from cache...\n");
                fin.read((char *) MRF_data.data(), MRF_data.size() * sizeof(EnergyType));
                recompute = false;
            }
        }
        if(recompute) {
            theia::Camera cam1 = reconstruction.View(anchor)->Camera();

            int index = 0;
            int unit = width * height / 10;
            //Be careful about down sample ratio!!!!!
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x, ++index) {
                    if (index % unit == 0)
                        cout << '.' << flush;

                    Vector3d ray = cam1.PixelToUnitDepthRay(Vector2d(x * downsample, y * downsample));
                    ray.normalize();
#pragma omp parallel for
                    for (int d = 0; d < dispResolution; ++d) {
                        //compute 3D point
                        double disp = min_disp + d * (max_disp - min_disp) / (double) dispResolution;

                        //sample in 3D space
                        vector<Vector4d> sptBase;
                        for (auto dy = -1 * pR; dy <= pR; ++dy) {
                            for (auto dx = -1 * pR; dx <= pR; ++dx) {
                                Vector2d pt(x + dx, y + dy);
                                if (pt[0] < 0 || pt[1] < 0 || pt[0] > width - 1 || pt[1] > height - 1) {
                                    sptBase.push_back(Vector4d(0, 0, 0, 0));
                                    continue;
                                }
                                Vector3d ray = cam1.PixelToUnitDepthRay(pt * downsample);
                                ray.normalize();
                                Vector3d spt = cam1.GetPosition() + ray * 1.0 / disp;
                                Vector4d spt_homo(spt[0], spt[1], spt[2], 1.0);
                                sptBase.push_back(spt_homo);
                            }
                        }

                        //project onto other views and compute matching cost
                        vector<vector<double> > patches(images.size());
                        for (auto v = 0; v < images.size(); ++v) {
                            theia::Camera cam2 = reconstruction.View(v + offset)->Camera();
                            patches[v].reserve((size_t) 3 * (2 * pR + 1) * (2 * pR + 1));
                            for (auto &spt: sptBase) {
                                if (spt[3] == 0) {
                                    patches[v].push_back(-1);
                                    patches[v].push_back(-1);
                                    patches[v].push_back(-1);
                                } else {
                                    Vector2d imgpt;
                                    cam2.ProjectPoint(spt, &imgpt);
                                    imgpt = imgpt / (double) downsample;
                                    if (imgpt[0] < 0 || imgpt[1] < 0 || imgpt[0] > width - 1 || imgpt[1] > height - 1) {
                                        patches[v].push_back(-1);
                                        patches[v].push_back(-1);
                                        patches[v].push_back(-1);
                                    } else {
                                        Vector3d c = interpolation_util::bilinear<uchar, 3>(images[v].data, width,
                                                                                            height, imgpt);
                                        patches[v].push_back(c[0]);
                                        patches[v].push_back(c[1]);
                                        patches[v].push_back(c[2]);
                                    }
                                }
                            }
//                        Vector3d spt = cam1.GetPosition() + ray * 1.0 / disp;
//                        Vector4d spt_homo(spt[0], spt[1], spt[2], 1.0);
//                        //project onto other views and compute matching cost
//                        vector<vector<double> > patches(images.size());
//                        for (auto v = 0; v < images.size(); ++v) {
//                            theia::Camera cam2 = reconstruction.View(v + offset)->Camera();
//                            Vector2d imgpt;
//                            cam2.ProjectPoint(spt_homo, &imgpt);
//                            imgpt = imgpt / (double) downsample;
//                            //TODO: shifting window
//                            MRF_util::samplePatch(images[v], imgpt, pR, patches[v]);

                        }

//                        if(x == tx && y == ty){
//                            printf("----------------------\n");
//                            printf("Debug for (%d,%d,%d)\n", x, y, d);
//                            for(auto v=0; v<patches.size(); ++v){
//                                printf("view %d: ", v);
//                                for(auto p: patches[v])
//                                    printf("%.3f ", p);
//                                printf("\n");
//                            }
//                        }

                        //compute the matching cost starting from each images
//                    vector<double> mCostGroup(images.size()); //matching cost
//                    for(auto v=0; v<patches.size(); ++v)
//                        mCostGroup[v] = MRF_util::medianMatchingCost(patches, v);
//                    size_t kth = mCostGroup.size() / 2;
//                    nth_element(mCostGroup.begin(), mCostGroup.begin()+kth, mCostGroup.end());
//                    MRF_data[dispResolution * (y*width+x) + d] = (int)((mCostGroup[kth] - 1) * (mCostGroup[kth] - 1) * MRFRatio);
                        //double mCost = MRF_util::medianMatchingCost(patches, anchor - offset);
                        double mCost = MRF_util::sumMatchingCostHalf(patches, anchor - offset);
//                    MRF_data[dispResolution * (y * width + x) + d] = (EnergyType) ((mCost - 1) * (mCost - 1) *
//                                                                                   MRFRatio);
                        MRF_data[dispResolution * (y * width + x) + d] = (EnergyType) ((mCost + 1) * MRFRatio);
                    }

                }
            }
            cout << "done" << endl;
            //caching
            ofstream fout(buffer, ios::binary);
            if (!fout.is_open()) {
                printf("Can not open cache file to write: %s\n", buffer);
                return;
            }
            printf("Writing unary term to cache...\n");
            int sz = sizeof(EnergyType);
            fout.write((char *) &anchor, sizeof(int));
            fout.write((char *) &dispResolution, sizeof(int));
            fout.write((char *) &tWindow, sizeof(int));
            fout.write((char *) &downsample, sizeof(int));
            fout.write((char *) &sz, sizeof(int));
            fout.write((char *) &min_disp, sizeof(double));
            fout.write((char *) &max_disp, sizeof(double));
            fout.write((char *) MRF_data.data(), MRF_data.size() * sizeof(EnergyType));

            fout.close();
        }

	    for(auto y=0; y<height; ++y){
		    for(auto x=0; x<width; ++x){
			    EnergyType min_energy = numeric_limits<EnergyType>::max();
			    for (int d = 0; d < dispResolution; ++d) {
				    const EnergyType curEnergy = MRF_data[dispResolution * (y * width + x) + d];
				    if ((double) curEnergy < min_energy) {
					    dispUnary.setDepthAtInt(x, y, (double) d);
					    min_energy = curEnergy;
				    }
			    }
		    }
	    }

    }

}//namespace dynamic_stereo
