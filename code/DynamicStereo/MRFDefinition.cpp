//
// Created by yanhang on 2/24/16.
//

#include "dynamicstereo.h"
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>


using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
    namespace MRF_util{
        void samplePatch(const cv::Mat& img, const Vector2d& loc, const int pR, std::vector<double>& pix) {
            const int w = img.cols;
            const int h = img.rows;
            pix.resize((size_t) 3 * (2 * pR + 1) * (2 * pR + 1));
            int index = 0;
            for (int dx = -1 * pR; dx <= pR; ++dx) {
                for (int dy = -1 * pR; dy <= pR; ++dy, ++index) {
                    Vector2d curloc(loc[0] + dx, loc[1] + dy);
                    if (curloc[0] < 0 || curloc[1] < 0 || curloc[0] >= w - 1 || curloc[1] >= h - 1){
                        pix[index * 3] = -1;
                        pix[index * 3 + 1] = -1;
                        pix[index * 3 + 2] = -1;
                    }else{
                        Vector3d pv = interpolation_util::bilinear<uchar, 3>(img.data, w, h, curloc);
                        pix[index * 3] = pv[0];
                        pix[index * 3 + 1] = pv[1];
                        pix[index * 3 + 2] = pv[2];
                    }
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
                if(p1.size() < pRef.size() / 2)
                    continue;
                mCost.push_back(math_util::normalizedCrossCorrelation(p1,p2));
            }
            //if the patch is not visible in >50% frames, assign large penalty.
            if(mCost.size() < patches.size() / 2)
                return -1;
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
        //ignore furthest 1% and nearest 1% points
        const double lowRatio = 0.01;
        const double highRatio = 0.99;
        const size_t lowKth = (size_t)(lowRatio * disps.size());
        const size_t highKth = (size_t)(highRatio * disps.size());
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
#pragma omp parallel for
                for(int d=0; d<dispResolution; ++d){
                    //compute 3D point
                    double disp = min_disp + d * (max_disp - min_disp) / (double)dispResolution;
                    Vector3d spt = cam1.GetPosition() + ray * 1.0 / disp;
                    Vector4d spt_homo(spt[0], spt[1], spt[2], 1.0);

                    //project onto other views and compute matching cost
                    vector<vector<double> > patches(images.size());
                    for(auto v=0; v<images.size(); ++v){
                        theia::Camera cam2 = reconstruction.View(v+offset)->Camera();
                        Vector2d imgpt;
                        cam2.ProjectPoint(spt_homo, &imgpt);
                        imgpt = imgpt / (double)downsample;
                        //TODO: shifting window
                        MRF_util::samplePatch(images[v], imgpt, pR, patches[v]);
                    }

                    //compute the matching cost starting from each images
//                    vector<double> mCostGroup(images.size()); //matching cost
//                    for(auto v=0; v<patches.size(); ++v)
//                        mCostGroup[v] = MRF_util::medianMatchingCost(patches, v);
//                    size_t kth = mCostGroup.size() / 2;
//                    nth_element(mCostGroup.begin(), mCostGroup.begin()+kth, mCostGroup.end());
//                    MRF_data[dispResolution * (y*width+x) + d] = (int)((mCostGroup[kth] - 1) * (mCostGroup[kth] - 1) * MRFRatio);
                    double mCost = MRF_util::medianMatchingCost(patches, anchor-offset);
                    MRF_data[dispResolution * (y*width+x) + d] = (EnergyType)((mCost-1)*(mCost-1)*MRFRatio);
                }
            }
        }
        cout << "done" << endl;
    }

    void DynamicStereo::assignSmoothWeight() {
        //mean of the gaussian distribution of gradient for edge
        const double t = 0.3;
        hCue.resize(width * height, 0);
        vCue.resize(width * height, 0);
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
                        vCue[y*width+x] = 0;
                    else
                        vCue[y*width+x] = (EnergyType) ((diff - t) * (diff - t)  * MRFRatio);
                }
                if(x < width - 1){
                    Vec3b pix2 = img.at<Vec3b>(y,x+1);
                    Vector3d dpix2 = Vector3d(pix2[0], pix2[1], pix2[2]) / 255.0;
                    double diff = (dpix1 - dpix2).norm();
                    if(diff > t)
                        hCue[y*width+x] = 0;
                    else
                        hCue[y*width+x] = (EnergyType) ((diff - t) * (diff - t) * MRFRatio);
                }
            }
        }
    }

//    std::shared_ptr<MRF> DynamicStereo::createProblem() {
//
//        //use truncated linear cost for smoothness
//        EnergyFunction *energy_function = new EnergyFunction(new DataCost(MRF_data.data()),
//                                                             new SmoothnessCost(1, 4, (MRF::CostVal)(weight_smooth * MRFRatio), hCue.data(), vCue.data()));
//        shared_ptr<MRF> mrf(new Expansion(width, height, dispResolution, energy_function));
//        mrf->initialize();
//
//        return mrf;
//    }

    std::shared_ptr<DynamicStereo::GraphicalModel> DynamicStereo::createGraphcialModel() {
        opengm::SimpleDiscreteSpace<> space((size_t)width * height, (size_t) dispResolution);
        shared_ptr<GraphicalModel> model(new GraphicalModel(space));

        size_t shape[] = {size_t(dispResolution), size_t(dispResolution)};
        //unary term
        for(auto vid=0; vid<width * height; ++vid){
            opengm::ExplicitFunction<EnergyType> f(shape, shape+1);
            for(auto l=0; l<dispResolution; ++l)
                f(l) = MRF_data[vid*dispResolution+l];
            GraphicalModel::FunctionIdentifier fid = model->addFunction(f);
            size_t variableIndices[] = {(size_t)vid};
            model->addFactor(fid, variableIndices, variableIndices+1);
        }

        //pairwise term: truncated absolute difference
        vector<double> pairTable((size_t)(dispResolution * dispResolution));
        const int max_diff = 4;
        for(auto l1=0; l1<dispResolution; ++l1){
            for(auto l2=0; l2<dispResolution; ++l2){
                pairTable[l1*dispResolution + l2] = std::max(std::abs(l1-l2), max_diff);
            }
        }

        for(auto y=0; y<height-1; ++y){
            for(auto x=0; x<width-1; ++x) {
                opengm::ExplicitFunction<EnergyType> fx(shape, shape + 2), fy(shape, shape+2);
                size_t vIdxV[] = {(size_t)y * width + x, (size_t)(y + 1) * width + x};
                size_t vIdxH[] = {(size_t)y * width + x, (size_t)y * width + x + 1};
                for(size_t l1=0; l1 < dispResolution; ++l1){
                    for(size_t l2=0; l2<dispResolution; ++l2) {
                        fy(l1, l2) = (EnergyType) (weight_smooth * pairTable[l1 * dispResolution + l2] * vCue[vIdxV[0]] * MRFRatio);
                        fx(l1, l2) = (EnergyType) (weight_smooth * pairTable[l1 * dispResolution + l2] * hCue[vIdxH[0]] * MRFRatio);
                    }
                }
                GraphicalModel::FunctionIdentifier fidx = model->addFunction(fx);
                GraphicalModel::FunctionIdentifier fidy = model->addFunction(fy);
                model->addFactor(fidx, vIdxH, vIdxH+2);
                model->addFactor(fidy, vIdxV, vIdxV+2);
            }
        }
        return model;
    }

    void DynamicStereo::optimize(std::shared_ptr<GraphicalModel> model) {
        //solve with alpha-expansion
        typedef opengm::external::MinSTCutKolmogorov<size_t, EnergyType> MinStCutType;
        typedef opengm::GraphCut<GraphicalModel, opengm::Minimizer, MinStCutType> MinGraphCut;
        typedef opengm::AlphaExpansion<GraphicalModel, MinGraphCut> MinAlphaExpansion;

        float t = (float)getTickCount();

        MinAlphaExpansion solver(*model.get());
        cout << "Solving.." << endl;
        solver.infer();
        t = ((float)getTickCount() - t) / (float)getTickFrequency();
        printf("Done. Final energy:%.2f, time usage:%.2fs\n", (double)solver.value() / MRFRatio, t);

        //copy result into depth map
        vector<GraphicalModel::LabelType> x;
        solver.arg(x);
        CHECK_EQ(x.size(), width * height);
        for(auto i=0; i<x.size(); ++i)
            refDepth.setDepthAtInd(i, (double)x[i]);
        refDepth.updateStatics();
    }

//    void DynamicStereo::optimize(std::shared_ptr<MRF> model) {
//        model->clearAnswer();
//        //randomly initialize
//        srand(time(NULL));
//        for (auto i = 0; i < width * height; ++i) {
//            model->setLabel(rand() % dispResolution, 0);
//        }
//
//        double initData = (double) model->dataEnergy() / MRFRatio;
//        double initSmooth = (double) model->smoothnessEnergy() / MRFRatio;
//        float t;
//        cout << "Solving..." << endl << flush;
//        model->optimize(10, t);
//
//        double finalData = (double) model->dataEnergy() / MRFRatio;
//        double finalSmooth = (double) model->smoothnessEnergy() / MRFRatio;
//        printf("Done.\n Init energy:(%.3f,%.3f,%.3f), final energy: (%.3f,%.3f,%.3f), time usage: %.2f\n", initData,
//               initSmooth, initData + initSmooth,
//               finalData, finalSmooth, finalData + finalSmooth, t);
//
//        //assign depth to depthmap
//        const double epsilon = 0.00001;
//        for(auto x=0; x<width; ++x) {
//            for (auto y = 0; y < height; ++y) {
//                double l = (double) model->getLabel(y * width + x);
//                double disp = min_disp + l * (max_disp - min_disp) / (double) dispResolution;
//                if(disp < epsilon)
//                    refDepth.setDepthAtInt(x, y, -1);
//                else
//                    refDepth.setDepthAtInt(x, y, l);
//            }
//        }
//        refDepth.updateStatics();
//    }
}//namespace dynamic_stereo
