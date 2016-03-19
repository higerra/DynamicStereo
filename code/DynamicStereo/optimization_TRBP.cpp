//
// Created by yanhang on 3/3/16.
//
#include <map>
#include "optimization.h"
#include "external/segment_ms/msImageProcessor.h"
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/graphcut.hxx>
//#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>

#include <opengm/functions/sparsemarray.hxx>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    SecondOrderOptimizeTRBP::SecondOrderOptimizeTRBP(const FileIO& file_io_, const int kFrames_,const cv::Mat& image_, const std::vector<EnergyType> &MRF_data_,
                                                     const float MRFRatio_, const int nLabel_):
            StereoOptimization(file_io_, kFrames_, image_, MRF_data_, MRFRatio_, nLabel_){
        segment_ms::msImageProcessor ms_segmentator;
        ms_segmentator.DefineBgImage(image.data, segment_ms::COLOR, image.rows, image.cols);
        const int hs = 4;
        const float hr = 5.0f;
        const int min_a = 40;
        ms_segmentator.Segment(hs, hr, min_a, meanshift::SpeedUpLevel::MED_SPEEDUP);
        refSeg.resize((size_t)image.cols * image.rows);
        const int * labels = ms_segmentator.GetLabels();
        for(auto i=0; i<image.cols * image.rows; ++i)
            refSeg[i] = labels[i];

        laml = 9 * kFrames;
        lamh = 108 * kFrames;
    }

    void SecondOrderOptimizeTRBP::optimize(Depth &result, const int max_iter) const {
        char buffer[1024] = {};
        //formulate problem with OpenGM
        typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
        typedef opengm::SparseFunction<EnergyType, size_t, size_t> SparseFunction;
        typedef opengm::GraphicalModel<EnergyType, opengm::Adder,
                opengm::ExplicitFunction<EnergyType>, Space> GraphicalModel;
        typedef opengm::TrbpUpdateRules<GraphicalModel, opengm::Minimizer> UpdateRules;
        typedef opengm::MessagePassing<GraphicalModel, opengm::Minimizer, UpdateRules, opengm::MaxDistance> TRBP;
        //typedef opengm::GraphCut<GraphicalModel, opengm::Minimizer

        Space space((size_t)(width * height), (size_t)nLabel);
        GraphicalModel gm(space);
        //add unary terms
        size_t shape[] = {(size_t) nLabel, (size_t) nLabel, (size_t)nLabel};
        for (auto i = 0; i < width * height; ++i) {
            opengm::ExplicitFunction<EnergyType> f(shape, shape + 1);
            for (auto l = 0; l < nLabel; ++l)
                f(l) = (EnergyType)MRF_data[nLabel * i + l];
            GraphicalModel::FunctionIdentifier fid = gm.addFunction(f);
            size_t vid[] = {(size_t) i};
            gm.addFactor(fid, vid, vid + 1);
        }

        //add triple terms
        cout << "Adding triple terms" << endl << flush;

        const int trun = 4;
        //SparseFunction fv(shape, shape+3, (EnergyType)(MRFRatio * trun));// fh(shape, shape+3, (EnergyType)(MRFRatio * trun));
        opengm::ExplicitFunction<EnergyType> fv(shape, shape+3);
        int count = 0;
        for (int l0 = 0; l0 < nLabel; ++l0) {
            for (int l1 = 0; l1 < nLabel; ++l1) {
                for (int l2 = 0; l2 < nLabel; ++l2) {
                    int labelDiff = l0 + l2 - 2 * l1;
                    size_t coord[] = {(size_t)l0, (size_t)l1, (size_t)l2};
                    if (abs(labelDiff) <= trun){
                        //fv.insert(coord, (EnergyType)(MRFRatio * labelDiff));
                        fv((size_t)l0,(size_t)l1,(size_t)l2) = (EnergyType)(labelDiff * MRFRatio);
                        count++;
                    } else
                        fv((size_t)l0,(size_t)l1,(size_t)l2) = 0;
                }
            }
        }
        GraphicalModel::FunctionIdentifier fidtriple = gm.addFunction(fv);
        cout << "Non zero count: " << count << endl << flush;

        for (size_t y = 1; y < 10; ++y) {
            for (size_t x = 1; x < 10; ++x) {
//                int lamH, lamV;
//                if(refSeg[y*width+x] == refSeg[y*width+x+1] && refSeg[y*width+x] == refSeg[y*width+x-1])
//                    lamH = lamh;
//                else
//                    lamH = laml;
//                if(refSeg[y*width+x] == refSeg[(y+1)*width+x] && refSeg[y*width+x] == refSeg[(y-1)*width+x])
//                    lamV = lamh;
//                else
//                    lamV = laml;

                size_t vIndxV[] = {(size_t) (y - 1) * width + x, (size_t) y * width + x, (size_t) (y + 1) * width + x};
                size_t vIndxH[] = {(size_t) y * width + x - 1, (size_t) y * width + x, (size_t) y * width + x + 1};

//                SparseFunction fv(shape, shape+3, (EnergyType)(lamV * MRFRatio * trun)), fh(shape, shape+3, (EnergyType)(lamH * MRFRatio * trun));
//
//                for (int l0 = 0; l0 < nLabel; ++l0) {
//                    for (int l1 = 0; l1 < nLabel; ++l1) {
//                        for (int l2 = 0; l2 < nLabel; ++l2) {
//                            int labelDiff = l0 + l2 - 2 * l1;
//                            size_t coord[] = {(size_t)l0, (size_t)l1, (size_t)l2};
//                            if (abs(labelDiff) <= trun){
//                                fv.insert(coord, (EnergyType)(lamV * MRFRatio * labelDiff));
//                                fh.insert(coord, (EnergyType)(lamH * MRFRatio * labelDiff));
//                            }
//                        }
//                    }
//                }

//                GraphicalModel::FunctionIdentifier fidv = gm.addFunction(fv);
//                GraphicalModel::FunctionIdentifier fidh = gm.addFunction(fh);

                gm.addFactor(fidtriple, vIndxV, vIndxV + 3);
                gm.addFactor(fidtriple, vIndxH, vIndxH + 3);
            }
        }

        //solve
        const double converge_bound = 1e-7;
        const double damping = 0.0;
        TRBP::Parameter parameter(max_iter);
        TRBP trbp(gm, parameter);
        cout << "Solving with TRBP..." << endl << flush;
        float t = (float)getTickCount();
        trbp.infer();
        t = ((float)getTickCount() - t) / (float)getTickFrequency();
        EnergyType finalEnergy = trbp.value();
        printf("Done. Final energy: %.3f, Time usage: %.2fs\n", (float)finalEnergy / MRFRatio, t);

        vector<size_t> labels;
        trbp.arg(labels);
        CHECK_EQ(labels.size(), width * height);
        result.initialize(width, height, -1);
        for(auto i=0; i<width * height; ++i)
            result.setDepthAtInd(i, labels[i]);
    }

    double SecondOrderOptimizeTRBP::evaluateEnergy(const Depth& disp) const {
        return 0.0;
    }

}//namespace dynamic_stereo
