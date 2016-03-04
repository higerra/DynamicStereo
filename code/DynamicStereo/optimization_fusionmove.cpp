//
// Created by yanhang on 3/3/16.
//

#include <list>
#include <random>
#include "optimization.h"
#include "external/QPBO1.4/ELC.h"
#include "external/QPBO1.4/QPBO.h"

#include "proposal.h"
#include "external/segment_ms/msImageProcessor.h"

using namespace std;
using namespace cv;
namespace dynamic_stereo {

    SecondOrderOptimizeFusionMove::SecondOrderOptimizeFusionMove(const FileIO &file_io_, const int kFrames_,
                                                                 const cv::Mat &image_,
                                                                 const std::vector<EnergyType> &MRF_data_,
                                                                 const float MRFRatio_,
                                                                 const int nLabel_, const Depth &noisyDisp_,
                                                                 const double min_disp_, const double max_disp_) :
            StereoOptimization(file_io_, kFrames_, image_, MRF_data_, MRFRatio_, nLabel_), noisyDisp(noisyDisp_),
            min_disp(min_disp_), max_disp(max_disp_), trun(4.0), average_over(20) {
        //segment ref image to get CRF weight
        segment_ms::msImageProcessor ms_segmentator;
        ms_segmentator.DefineBgImage(image.data, segment_ms::COLOR, image.rows, image.cols);
        const int hs = 4;
        const float hr = 5.0f;
        const int min_a = 40;
        ms_segmentator.Segment(hs, hr, min_a, meanshift::SpeedUpLevel::MED_SPEEDUP);
        refSeg.resize((size_t) image.cols * image.rows);
        const int *labels = ms_segmentator.GetLabels();
        for (auto i = 0; i < image.cols * image.rows; ++i)
            refSeg[i] = labels[i];

        laml = 9.0 * (double)kFrames / 255.0;
        lamh = 108.0 * (double)kFrames / 255.5;
    }


    void SecondOrderOptimizeFusionMove::optimize(Depth &result, const int max_iter) const {
        vector<Depth> proposals;
        genProposal(proposals);
        proposals.push_back(noisyDisp);

        char buffer[1024] = {};

        //initialize by random
        result.initialize(width, height, -1);

        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(0, nLabel-1);
        for(auto i=0; i<width*height; ++i)
            result.setDepthAtInd(i, (double)distribution(generator));

        list<double> diffE;
        double lastEnergy = evaluateEnergy(result);
        double initialEnergy = lastEnergy;
        int iter = 0;

        const double termination = 10;
        float timming = (float)getTickCount();
        while (true) {
            if(iter == max_iter)
                break;
            cout << "Iteration " << iter << " using proposal " << iter % (proposals.size()) << endl;
            const Depth& proposal = proposals[iter % (proposals.size())];
            cout << "Fusing..." << endl;
            fusionMove(result, proposal);
            double e = evaluateEnergy(result);
            cout << "Done, energy: " << e << endl;
            double energyDiff = lastEnergy - e;

            if(diffE.size() >= average_over)
                diffE.pop_front();
            diffE.push_back(energyDiff);
            double average_diffe = std::accumulate(diffE.begin(), diffE.end(), 0.0) / (double)diffE.size();
            lastEnergy = e;

//            sprintf(buffer, "%s/temp/fusionmove_iter%05d.jpg", file_io.getDirectory().c_str(), iter);
//            result.saveImage(buffer, 255.0 / (double)nLabel);

            if(average_diffe < termination) {
                cout << "Converge!" << endl;
                break;
            }

            iter++;
        }
        timming = ((float)getTickCount() - timming) / (float)getTickFrequency();
        printf("All done. Initial energy: %.3f, final energy: %.3f, time usage: %.2fs\n", initialEnergy, lastEnergy, timming);


    }

    double SecondOrderOptimizeFusionMove::evaluateEnergy(const Depth &disp) const {
        CHECK_EQ(disp.getWidth(), width);
        CHECK_EQ(disp.getHeight(), height);
        double e = 0.0;
        for (auto i = 0; i < disp.getWidth() * disp.getHeight(); ++i) {
            int l = (int) disp[i];
            e += (double)(MRF_data[nLabel * i + l]) / (double)(MRFRatio);
        }

        for (auto x = 1; x < width - 1; ++x) {
            for (auto y = 1; y < height - 1; ++y) {
                int id1, id2, id3;
                double d1, d2, d3;
                double lam;
                id1 = y * width + x - 1;
                id2 = y * width + x;
                id3 = y * width + x + 1;
                if (refSeg[id1] == refSeg[id2] && refSeg[id1] == refSeg[id3])
                    lam = lamh;
                else
                    lam = laml;
                d1 = disp(x - 1, y);
                d2 = disp(x, y);
                d3 = disp(x + 1, y);
                e += std::min(std::abs(d1 + d3 - 2 * d2), trun) * lam;

                id1 = (y - 1) * width + x;
                id2 = y * width + x;
                id3 = (y - 1) * width + x;
                if (refSeg[id1] == refSeg[id2] && refSeg[id1] == refSeg[id3])
                    lam = lamh;
                else
                    lam = laml;
                d1 = disp(x, y - 1);
                d2 = disp(x, y);
                d3 = disp(x, y + 1);
                e += std::min(std::abs(d1 + d3 - 2 * d2), trun) * lam;
            }
        }
        return e;
    }

    void SecondOrderOptimizeFusionMove::genProposal(std::vector<Depth> &proposals) const {
        char buffer[1024] = {};
        cout << "Generating plane proposal" << endl;
        ProposalSegPlnMeanshift proposalFactoryMeanshift(file_io, image, noisyDisp, nLabel, min_disp, max_disp);
        proposalFactoryMeanshift.genProposal(proposals);
        vector<Depth> proposalsGb;
        ProposalSegPlnGbSegment proposalFactoryGbSegment(file_io, image, noisyDisp, nLabel, min_disp, max_disp);
        proposalFactoryGbSegment.genProposal(proposalsGb);
        proposals.insert(proposals.end(), proposalsGb.begin(), proposalsGb.end());
        for (auto i = 0; i < proposals.size(); ++i) {
            sprintf(buffer, "%s/temp/proposalPln%03d.jpg", file_io.getDirectory().c_str(), i);
            proposals[i].saveImage(buffer, 255.0 / (double) nLabel);
        }
    }

    void SecondOrderOptimizeFusionMove::fusionMove(Depth &p1, const Depth &p2) const {
        //create problem
        ELCReduce::PBF<EnergyType> pbf;
        const int &dispResolution = nLabel;
        const int width = image.cols;
        const int height = image.rows;
        //formulate
        //unary term
        for (auto i = 0; i < width * height; ++i) {
            int disp1 = (int) p1.getDepthAtInd(i);
            int disp2 = (int) p2.getDepthAtInd(i);
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
        for (auto x = 1; x < width - 1; ++x) {
            for (auto y = 1; y < height - 1; ++y) {
                //horizontal
                int p = y * width + x - 1;
                int q = y * width + x;
                int r = y * width + x + 1;
                EnergyType lam;
                if (refSeg[p] == refSeg[q] && refSeg[p] == refSeg[r])
                    lam = lamh;
                else
                    lam = laml;

                SE[0] = (EnergyType)(
                        std::min(std::abs(p1.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q)), trun) *
                        MRFRatio);
                SE[1] = (EnergyType)(
                        std::min(std::abs(p1.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q)), trun) *
                        MRFRatio);
                SE[2] = (EnergyType)(
                        std::min(std::abs(p1.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q)), trun) *
                        MRFRatio);
                SE[3] = (EnergyType)(
                        std::min(std::abs(p1.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q)), trun) *
                        MRFRatio);
                SE[4] = (EnergyType)(
                        std::min(std::abs(p2.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q)), trun) *
                        MRFRatio);
                SE[5] = (EnergyType)(
                        std::min(std::abs(p2.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q)), trun) *
                        MRFRatio);
                SE[6] = (EnergyType)(
                        std::min(std::abs(p2.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q)), trun) *
                        MRFRatio);
                SE[7] = (EnergyType)(
                        std::min(std::abs(p2.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q)), trun) *
                        MRFRatio);
                for (auto &S: SE)
                    S = (EnergyType)((double)S * lam);

                indices[0] = p;
                indices[1] = q;
                indices[2] = r;
                pbf.AddHigherTerm(3, indices.data(), SE.data());

                //vertical
                p = (y - 1) * width + x;
                r = (y + 1) * width + x;
                if (refSeg[p] == refSeg[q] && refSeg[p] == refSeg[r])
                    lam = lamh;
                else
                    lam = laml;
                SE[0] = (EnergyType)(
                        std::min(std::abs(p1.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q)), trun) *
                        MRFRatio);
                SE[1] = (EnergyType)(
                        std::min(std::abs(p1.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q)), trun) *
                        MRFRatio);
                SE[2] = (EnergyType)(
                        std::min(std::abs(p1.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q)), trun) *
                        MRFRatio);
                SE[3] = (EnergyType)(
                        std::min(std::abs(p1.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q)), trun) *
                        MRFRatio);
                SE[4] = (EnergyType)(
                        std::min(std::abs(p2.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q)), trun) *
                        MRFRatio);
                SE[5] = (EnergyType)(
                        std::min(std::abs(p2.getDepthAtInd(p) + p1.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q)), trun) *
                        MRFRatio);
                SE[6] = (EnergyType)(
                        std::min(std::abs(p2.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p1.getDepthAtInd(q)), trun) *
                        MRFRatio);
                SE[7] = (EnergyType)(
                        std::min(std::abs(p2.getDepthAtInd(p) + p2.getDepthAtInd(r) - 2 * p2.getDepthAtInd(q)), trun) *
                        MRFRatio);
                for (auto &S: SE)
                    S = (EnergyType)((double)S * lam);

                indices[0] = p;
                indices[1] = q;
                indices[2] = r;
                pbf.AddHigherTerm(3, indices.data(), SE.data());
            }
        }

        //reduce
        cout << "Reducing with ELC..." << endl;
        ELCReduce::PBF<EnergyType> qpbf;
        pbf.reduceHigher();
        pbf.toQuadratic(qpbf, width * height);
        int numVar = qpbf.maxID();

        kolmogorov::qpbo::QPBO<EnergyType> qpbo(numVar, numVar * 4);
        qpbf.convert(qpbo, numVar);
        //solve
        cout << "Solving..." << endl << flush;
        float t = (float) getTickCount();
        qpbo.MergeParallelEdges();
        qpbo.Solve();
        qpbo.ComputeWeakPersistencies();
        t = ((float) getTickCount() - t) / (float) getTickFrequency();
        printf("Done. Time usage:%.3f\n", t);

        //fusion
        float unlabeled = 0.0;
        for (auto i = 0; i < width * height; ++i) {
            int l = qpbo.GetLabel(i);
            int disp1 = (int) p1.getDepthAtInd(i);
            int disp2 = (int) p2.getDepthAtInd(i);
            if (l == 0)
                p1.setDepthAtInd(i, disp1);
            else if (l < 0) {
                p1.setDepthAtInd(i, disp1);
                unlabeled += 1.0;
            }
            else
                p1.setDepthAtInd(i, disp2);
        }

        printf("Unlabeled pixels: %.2f, Ratio: %.2f\n", unlabeled, unlabeled / (float) (width * height));
    }

}//namespace dynamic_stereo

