//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"
#include "external/MRF2.2/GCoptimization.h"
#include "local_matcher.h"
using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo {
    void DynamicStereo::computeMinMaxDisparity(){
        if(min_disp > 0 && max_disp > 0)
            return;
        if(reconstruction.NumTracks() == 0){
            CHECK_GT(min_disp, 0) << "Please specify the minimum disparity";
            CHECK_GT(max_disp, 0) << "Please specify the minimum disparity";
            return;
        }
        const theia::View *anchorView = reconstruction.View(orderedId[anchor].second);
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
        min_disp = disps[lowKth] * 0.6;
        nth_element(disps.begin(), disps.begin() + highKth, disps.end());
        max_disp = disps[highKth] * 1.5;

	    model->min_disp = min_disp;
	    model->max_disp = max_disp;
    }

    void DynamicStereo::initMRF() {
	    CHECK(model.get());
	    model->allocate();
        cout << "Assigning data term..." << endl << flush;
        assignDataTerm();
        assignSmoothWeight();
    }

    void DynamicStereo::assignDataTerm() {
	    CHECK_GT(min_disp, 0);
	    CHECK_GT(max_disp, 0);
	    //read from cache
	    char buffer[1024] = {};
	    sprintf(buffer, "%s/temp/cacheMRFdata%d", file_io.getDirectory().c_str(), dispResolution);
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
                fin.read((char *) model->unary.data(), model->unary.size() * sizeof(EnergyType));
                recompute = false;
            }
        }
        if(recompute) {
            const theia::Camera& cam1 = reconstruction.View(orderedId[anchor].second)->Camera();

            int index = 0;
            int unit = width * height / 10;
            //Be careful about down sample ratio!!!!!
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x, ++index) {
                    if (index % unit == 0)
                        cout << '.' << flush;
#pragma omp parallel for
                    for (int d = 0; d < dispResolution; ++d) {
                        //compute 3D point
                        double depth = model->dispToDepth(d);

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
                                //ray.normalize();
                                Vector3d spt = cam1.GetPosition() + ray * depth;
                                Vector4d spt_homo(spt[0], spt[1], spt[2], 1.0);
                                sptBase.push_back(spt_homo);
                            }
                        }

                        //project onto other views and compute matching cost
                        vector<vector<double> > patches(images.size());
                        for (auto v = 0; v < images.size(); ++v) {
                            const theia::Camera& cam2 = reconstruction.View(orderedId[v + offset].second)->Camera();
                            for (const auto &spt: sptBase) {
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
                        }
                        double mCost = local_matcher::sumMatchingCost(patches, anchor - offset);
                        //double mCost = local_matcher::medianMatchingCost(patches, anchor-offset);
	                    model->operator()(y*width+x, d) = (EnergyType) ((1 + mCost) * model->MRFRatio);
                        //MRF_data[dispResolution * (y * width + x) + d] = (EnergyType) ((1 - mCost) * MRFRatio);
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
            fout.write((char *) model->unary.data(), model->unary.size() * sizeof(EnergyType));

            fout.close();
        }

	    for(auto y=0; y<height; ++y){
		    for(auto x=0; x<width; ++x){
			    EnergyType min_energy = numeric_limits<EnergyType>::max();
			    for (int d = 0; d < dispResolution; ++d) {
				    const EnergyType curEnergy = model->operator()(y*width+x, d);
				    if ((double) curEnergy < min_energy) {
					    dispUnary.setDepthAtInt(x, y, (double) d);
					    min_energy = curEnergy;
				    }
			    }
		    }
	    }

    }

	void DynamicStereo::assignSmoothWeight() {
		const double t = 0.3;
		double ratio = 441.0;
		vector<EnergyType> &vCue = model->vCue;
		vector<EnergyType> &hCue = model->hCue;
		const Mat &img = model->image;
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
