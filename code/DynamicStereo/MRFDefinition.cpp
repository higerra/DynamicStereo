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

    void DynamicStereo::initMRF() {
	    CHECK(model.get());
	    model->allocate();
		double min_depth, max_depth;
		utility::computeMinMaxDepth(sfmModel, anchor, min_depth, max_depth);
		CHECK_GT(min_depth, 0.0);
		CHECK_GT(max_depth, 0.0);
		model->min_disp = 1.0 / max_depth;
		model->max_disp = 1.0 / min_depth;
        cout << "Assigning data term..." << endl << flush;
        assignDataTerm();
        assignSmoothWeight();
    }

	void DynamicStereo::getPatchArray(const double x, const double y, const int d, const int r, const theia::Camera& refCam, const int startid, const int endid, vector<vector<double> >& patches) const {
		double depth = model->dispToDepth(d);
		//sample in 3D space
		vector<Vector4d> sptBase;
		const double scale = 2.0;
		const double dr = (double)r / (double)downsample * scale;
		for (double dy = -1 * dr; dy <= dr; dy += scale  / (double)downsample) {
			for (double dx = -1 * dr; dx <= dr; dx += scale / (double)downsample) {
				Vector2d pt(x + dx, y + dy);
				if (pt[0] < 0 || pt[1] < 0 || pt[0] > width - 1 || pt[1] > height - 1) {
					sptBase.push_back(Vector4d(0, 0, 0, 0));
					continue;
				}
				Vector3d ray = refCam.PixelToUnitDepthRay(pt * (double)downsample);
				Vector3d spt = refCam.GetPosition() + ray * depth;
				Vector4d spt_homo(spt[0], spt[1], spt[2], 1.0);
				sptBase.push_back(spt_homo);
			}
		}

		//project onto other views and compute matching cost
		patches.resize((size_t)(endid - startid + 1));
		for (auto v = startid; v <= endid; ++v) {
			CHECK_GE(v,0);
			CHECK_LT(v, images.size());
			//const theia::Camera& cam2 = reconstruction.View(orderedId[v + offset].second)->Camera();
			const theia::Camera& cam2 = sfmModel.getCamera(v+offset);

			//printf("--------------Sample from frame %d\n", v+offset);
			for (const auto &spt: sptBase) {
				if (spt[3] == 0) {
					patches[v-startid].push_back(-1);
					patches[v-startid].push_back(-1);
					patches[v-startid].push_back(-1);
				} else {
					Vector2d imgpt;
					cam2.ProjectPoint(spt, &imgpt);
					//printf("image pt: (%.2f,%.2f)\t", imgpt[0], imgpt[1]);
					imgpt = imgpt / (double) downsample;

					if (imgpt[0] < 0 || imgpt[1] < 0 || imgpt[0] > width - 1 || imgpt[1] > height - 1) {
						patches[v-startid].push_back(-1);
						patches[v-startid].push_back(-1);
						patches[v-startid].push_back(-1);
					} else {
						Vector3d c = interpolation_util::bilinear<uchar, 3>(images[v].data, width,
						                                                    height, imgpt);
						patches[v-startid].push_back(c[0]);
						patches[v-startid].push_back(c[1]);
						patches[v-startid].push_back(c[2]);
					//	printf("Color: (%.2f,%.2f,%.2f)\n", c[0], c[1], c[2]);
					}
				}
			}
		}
	}

    void DynamicStereo::assignDataTerm() {
	    CHECK_GT(model->min_disp, 0);
	    CHECK_GT(model->max_disp, 0);
	    //read from cache
	    char buffer[1024] = {};
	    sprintf(buffer, "%s/temp/cacheMRFdata%05dR%dD%d", file_io.getDirectory().c_str(), anchor, dispResolution, downsample);
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
            printf("Cached data: anchor:%d, resolution:%d, twindow:%d, downsample:%d, Energytype:%d, min_disp:%.15f, max_disp:%.15f\n",
                   frame, resolution, tw, ds, type, mindisp, maxdisp);
		    printf("Current config:: anchor:%d, resolution:%d, twindow:%d, downsample:%d, Energytype:%d, min_disp:%.15f, max_disp:%.15f\n",
		           anchor, dispResolution, tWindowStereo, downsample, (int)sizeof(EnergyType), model->min_disp, model->max_disp);
            if (frame == anchor && resolution == dispResolution && tw == tWindowStereo &&
                type == sizeof(EnergyType) && ds == downsample) {
                printf("Reading unary term from cache...\n");
                fin.read((char *) model->unary.data(), model->unary.size() * sizeof(EnergyType));
                recompute = false;
            }
        }
        if(recompute) {
            //const theia::Camera& cam1 = reconstruction.View(orderedId[anchor].second)->Camera();

			const theia::Camera& cam1 = sfmModel.getCamera(anchor);
            int index = 0;
            int unit = width * height / 10;
            const int stereoOffset = anchor - tWindowStereo / 2;
            printf("Stereo Offset: %d, stereo frame range: %d->%d\n", stereoOffset, anchor-stereoOffset, anchor-stereoOffset+tWindowStereo);
            //Be careful about down sample ratio!!!!!
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x, ++index) {
                    if (index % unit == 0)
                        cout << '.' << flush;
#pragma omp parallel for
                    for (int d = 0; d < dispResolution; ++d) {
                        //compute 3D point
	                    vector<vector<double> > patches;
	                    getPatchArray((double)x,(double)y,d, pR, cam1, anchor-stereoOffset, anchor-stereoOffset+tWindowStereo-1, patches);
                        double mCost = local_matcher::sumMatchingCost(patches, anchor - stereoOffset);
                        //double mCost = local_matcher::medianMatchingCost(patches, (int)patches.size() / 2);
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
            fout.write((char *) &tWindowStereo, sizeof(int));
            fout.write((char *) &downsample, sizeof(int));
            fout.write((char *) &sz, sizeof(int));
            fout.write((char *) &model->min_disp, sizeof(double));
            fout.write((char *) &model->max_disp, sizeof(double));
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
		vector<EnergyType> &vCue = model->vCue;
		vector<EnergyType> &hCue = model->hCue;
		const double t = 40;
		const Mat &img = model->image;
		for (auto y = 0; y < height; ++y) {
			for (auto x = 0; x < width; ++x) {
				Vec3b pix1 = img.at<Vec3b>(y, x);
				//pixel value range from 0 to 1, not 255!
				Vector3d dpix1 = Vector3d(pix1[0], pix1[1], pix1[2]);
				if (y < height - 1) {
					Vec3b pix2 = img.at<Vec3b>(y + 1, x);
					Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]);
					double diff = (dpix1 - dpix2).squaredNorm();
					vCue[y*width+x] = (EnergyType) std::log(1+std::exp(-1*diff/t));
				}
				if (x < width - 1) {
					Vec3b pix2 = img.at<Vec3b>(y, x + 1);
					Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]);
					double diff = (dpix1 - dpix2).squaredNorm();
					hCue[y*width+x] = (EnergyType) std::log(1+std::exp(-1*diff/t));
				}
			}
		}
	}
}//namespace dynamic_stereo
