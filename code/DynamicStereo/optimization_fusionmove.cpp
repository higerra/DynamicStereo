//
// Created by yanhang on 3/3/16.
//

#include "optimization.h"
#include "external/QPBO1.4/ELC.h"
#include "external/QPBO1.4/QPBO.h"

#include "proposal.h"
#include "external/segment_ms/msImageProcessor.h"

using namespace std;
using namespace cv;
namespace dynamic_stereo{

    SecondOrderOptimizeFusionMove::SecondOrderOptimizeFusionMove(const FileIO& file_io_, const int kFrames_,
                                                                 const cv::Mat &image_,
                                                                 const std::vector<EnergyType> &MRF_data_,
                                                                 const float MRFRatio_,
                                                                 const int nLabel_, const Depth &noisyDisp_,
                                                                 const double min_disp_, const double max_disp_):
    StereoOptimization(file_io_, kFrames_, image_, MRF_data_, MRFRatio_, nLabel_), noisyDisp(noisyDisp_), min_disp(min_disp_), max_disp(max_disp_){
        //segment ref image to get CRF weight
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


    void SecondOrderOptimizeFusionMove::optimize(Depth &result, const int max_iter) const{

    }

    double SecondOrderOptimizeFusionMove::evaluateEnergy(const Depth& disp) const {
        return 0.0;
    }

    void SecondOrderOptimizeFusionMove::genProposal(std::vector<Depth> &proposals) {

    }

    void SecondOrderOptimizeFusionMove::fusionMove(Depth &p1, const Depth &p2) {
        //create problem
        ELCReduce::PBF<EnergyType> pbf;
        const int& dispResolution = nLabel;
        const int width = image.cols;
        const int height = image.rows;
        //formulate
        //unary term
        for(auto i=0; i<width*height; ++i){
            int disp1 = (int)p1.getDepthAtInd(i);
            int disp2 = (int)p2.getDepthAtInd(i);
            CHECK_GE(disp1, 0);
            CHECK_LT(disp1, dispResolution);
            CHECK_GE(disp2, 0);
            CHECK_LT(disp2, dispResolution);
            EnergyType ue1 = MRF_data[dispResolution * i + disp1];
            EnergyType ue2 = MRF_data[dispResolution * i + disp2];
            pbf.AddUnaryTerm(i, ue1, ue2);
        }

        vector<ELCReduce::VID> indices(3);
        vector<EnergyType> SE(8);
        for(auto x=1; x<width-1; ++x) {
            for (auto y = 1; y < height - 1; ++y) {
                //horizontal
                int p = y * width + x - 1;
                int q = y * width + x;
                int r = y * width + x + 1;
                EnergyType lam;
                if(refSeg[p] == refSeg[q] && refSeg[p] == refSeg[r])
                    lam = lamh;
                else
                    lam = laml;

                SE[0] = (EnergyType)(p1.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
                SE[1] = (EnergyType)(p1.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
                SE[2] = (EnergyType)(p1.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
                SE[3] = (EnergyType)(p1.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
                SE[4] = (EnergyType)(p2.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
                SE[5] = (EnergyType)(p2.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
                SE[6] = (EnergyType)(p2.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
                SE[7] = (EnergyType)(p2.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
                for(auto &S: SE)
                    S = S * lam;

                indices[0] = p; indices[1] = q; indices[2] = r;
                pbf.AddHigherTerm(3, indices.data(), SE.data());

                //vertical
                p = (y - 1) * width + x;
                r = (y + 1) * width + x;
                SE[0] = (EnergyType)(p1.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
                SE[1] = (EnergyType)(p1.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
                SE[2] = (EnergyType)(p1.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
                SE[3] = (EnergyType)(p1.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
                SE[4] = (EnergyType)(p2.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
                SE[5] = (EnergyType)(p2.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
                SE[6] = (EnergyType)(p2.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q));
                SE[7] = (EnergyType)(p2.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q));
                for(auto &S: SE)
                    S = S * lam;

                indices[0] = p; indices[1] = q; indices[2] = r;
                pbf.AddHigherTerm(3, indices.data(), SE.data());
            }
        }

        //reduce
        cout << "Reducing with ELC..." << endl;
        ELCReduce::PBF<EnergyType> qpbf;
        printf("Number of variables: %d\n", pbf.maxID());
        pbf.reduceHigher();
        pbf.toQuadratic(qpbf, width * height);
        int numVar = qpbf.maxID();
        printf("Done. number of variables:%d (ori %d)\n", numVar, width * height);
        printf("Convering to QPBO object...\n");

        kolmogorov::qpbo::QPBO<EnergyType> qpbo(numVar, numVar * 4);
        qpbf.convert(qpbo, numVar);

        printf("Number of nodes in qpbo: %d\n", qpbo.GetNodeNum());

        printf("Done\n");
        //solve
        cout << "Solving..." << endl << flush;
        float t = (float)getTickCount();
        qpbo.MergeParallelEdges();
        qpbo.Solve();
        qpbo.ComputeWeakPersistencies();
        t = ((float)getTickCount() - t) / (float)getTickFrequency();
        printf("Done. Time usage:%.3f\n", t);


        //fusion
        float unlabeled = 0.0;
        for(auto i=0; i<width * height; ++i){
            int l = qpbo.GetLabel(i);
            int disp1 = (int)p1.getDepthAtInd(i);
            int disp2 = (int)p2.getDepthAtInd(i);
            if(l == 0)
                p1.setDepthAtInd(i, disp1);
            else if(l < 0) {
                p1.setDepthAtInd(i, disp1);
                unlabeled += 1.0;
            }
            else
                p1.setDepthAtInd(i, disp2);
        }

        printf("Unlabeled pixels: %.2f, Ratio: %.2f\n", unlabeled, unlabeled / (float)(width * height));
    }

}//namespace dynamic_stereo

