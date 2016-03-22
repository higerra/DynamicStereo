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
    typedef double EnergyTypeT;

    SecondOrderOptimizeFusionMove::SecondOrderOptimizeFusionMove(const FileIO &file_io_, const int kFrames_, const cv::Mat &image_,
                                                                 const std::vector<int> &MRF_data_,
                                                                 const float MRFRatio_,
                                                                 const int nLabel_,
                                                                 const Depth &noisyDisp_,
                                                                 const double min_disp_, const double max_disp_):
            StereoOptimization(file_io_, kFrames_, image_, MRF_data_, MRFRatio_, nLabel_), noisyDisp(noisyDisp_), average_over(20), trun(4), min_disp(min_disp_), max_disp(max_disp_) {
        //segment ref image to get CRF weight
        segment_ms::msImageProcessor ms_segmentator;
        ms_segmentator.DefineBgImage(image.data, segment_ms::COLOR, image.rows, image.cols);
        const int hs = 7;
        const float hr = 10.5f;
        const int min_a = 70;
        ms_segmentator.Segment(hs, hr, min_a, meanshift::SpeedUpLevel::MED_SPEEDUP);
        refSeg.resize((size_t) image.cols * image.rows);
        const int *labels = ms_segmentator.GetLabels();
        for (auto i = 0; i < image.cols * image.rows; ++i)
            refSeg[i] = labels[i];

        laml = 0.02;
        lamh = 0.2;

    }


    void SecondOrderOptimizeFusionMove::optimize(Depth &result, const int max_iter) const {
        vector<Depth> proposals;
        genProposal(proposals);
        //proposals.push_back(noisyDisp);

        char buffer[1024] = {};

        //initialize by random
        result.initialize(width, height, -1);

        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(0, nLabel - 1);
        for (auto i = 0; i < width * height; ++i) {
            //result.setDepthAtInd(i, (double) distribution(generator));
            //result.setDepthAtInd(i, noisyDisp[i]);
            result[i] = 0;
        }
        sprintf(buffer, "%s/temp/init_result.jpg", file_io.getDirectory().c_str());
        result.saveImage(buffer, 256.0 / (double)nLabel);

        list<double> diffE;
        double lastEnergy = evaluateEnergy(result);
        double initialEnergy = lastEnergy;
        int iter = 0;

        const double termination = 0.1;
        float timming = (float) getTickCount();
        const int smoothInterval = 5;
        while (true) {
            if (iter == max_iter)
                break;
            cout << "======================" << endl;

            Depth newProposal;

//            if (iter > 0 && iter % smoothInterval == 0) {
//                Depth orip1;
//                newProposal.initialize(width, height, -1);
//                orip1.initialize(width, height, -1);
//                for (auto i = 0; i < width * height; ++i) {
//                    orip1.setDepthAtInd(i, result[i]);
//                    newProposal.setDepthAtInd(i, result[i]);
//                }
//                int direction = iter / smoothInterval;
//                if (direction % 2 == 0) {
//                    //horizontally
//                    for (auto y = 0; y < height; ++y) {
//                        for (auto x = 1; x < width - 1; ++x)
//                            newProposal.setDepthAtInt(x, y, (orip1(x + 1, y) + orip1(x - 1, y)) / 2);
//                        newProposal.setDepthAtInt(width - 1, y, orip1(width - 1, y));
//                    }
//                } else {
//                    //vertically
//                    for (auto x = 0; x < width; ++x) {
//                        for (auto y = 1; y < height - 1; ++y)
//                            newProposal.setDepthAtInt(x, y, (orip1(x, y + 1) + orip1(x, y - 1)) / 2);
//                        newProposal.setDepthAtInt(x, height - 1, orip1(x, height - 1));
//                    }
//                }
//                cout << "Iteration " << iter << " using smoothing proposal " << endl;
//            } else {
//          }
            newProposal = proposals[iter % (proposals.size())];
            printf("Fusing with proposal %d\n", (int)(iter % proposals.size()));
            //after several iteration, smooth the dispartiy
            fusionMove(result, newProposal);
            double e = evaluateEnergy(result);

            double energyDiff = lastEnergy - e;

            if (diffE.size() >= average_over)
                diffE.pop_front();
            diffE.push_back(energyDiff);
            double average_diffe = std::accumulate(diffE.begin(), diffE.end(), 0.0) / (double) diffE.size();

            printf("Done. Final energy: %.5f, energy decrease: %.5f average decrease: %.5f\n", e, energyDiff,
                   average_diffe);
            lastEnergy = e;

            sprintf(buffer, "%s/temp/fusionmove_iter%05d.jpg", file_io.getDirectory().c_str(), iter);
            result.saveImage(buffer, 256.0 / (double) nLabel);

            if (iter > proposals.size() * 2 && average_diffe < termination) {
                cout << "Converge!" << endl;
                break;
            }

            iter++;
        }
        timming = ((float) getTickCount() - timming) / (float) getTickFrequency();
        printf("All done. Initial energy: %.5f, final energy: %.5f, time usage: %.2fs\n", initialEnergy, lastEnergy,
               timming);
    }

    double SecondOrderOptimizeFusionMove::evaluateEnergy(const Depth &disp) const {
        CHECK_EQ(disp.getWidth(), width);
        CHECK_EQ(disp.getHeight(), height);
        double e = 0.0;
        for (auto i = 0; i < width * height; ++i) {
            int l = (int) disp[i];
            e += (double)(MRF_data[nLabel * i + l]) / (double)(MRFRatio);
        }
        auto tripleE = [&](int id1, int id2, int id3){
            double lam;
            if (refSeg[id1] == refSeg[id2] && refSeg[id1] == refSeg[id3])
                lam = lamh;
            else
                lam = laml;
            return lapE(disp[id1], disp[id2], disp[id3]) * lam;
        };

        for (auto x = 1; x < width - 1; ++x) {
            for (auto y = 1; y < height - 1; ++y) {
                e += tripleE(y * width + x - 1, y * width + x, y * width + x + 1);
                e += tripleE((y - 1) * width + x, y * width + x, (y + 1) * width + x);
            }
        }
        return e;
    }

    void SecondOrderOptimizeFusionMove::genProposal(std::vector<Depth> &proposals) const {
        char buffer[1024] = {};
//        cout << "Generating plane proposal" << endl;
//        ProposalSegPlnMeanshift proposalFactoryMeanshift(file_io, image, noisyDisp, nLabel, min_disp, max_disp);
//        proposalFactoryMeanshift.genProposal(proposals);
//        vector<Depth> proposalsGb;
//        ProposalSegPlnGbSegment proposalFactoryGbSegment(file_io, image, noisyDisp, nLabel, min_disp, max_disp);
//        proposalFactoryGbSegment.genProposal(proposalsGb);
//        proposals.insert(proposals.end(), proposalsGb.begin(), proposalsGb.end());

        //Add fronto parallel plane
        const int num = nLabel;
        for(auto i=0; i<num; ++i){
            double disp = (double)nLabel / (double)num * i;
            Depth p;
            p.initialize(width, height, disp);
            proposals.push_back(p);
        }
    }

    void SecondOrderOptimizeFusionMove::fusionMove(Depth &p1, const Depth &p2) const {
        //create problem
        int nPix = width * height;
        kolmogorov::qpbo::QPBO<EnergyTypeT> qpbo(nPix*10, nPix*20);
        //construct graph
        auto addTripleToGraph = [&](int p, int q, int r) {
            double vp1 = p1[p], vp2 = p2[p], vq1 = p1[q], vq2 = p2[q], vr1 = p1[r], vr2 = p2[r];
            double lam;
            if (refSeg[p] == refSeg[q] && refSeg[p] == refSeg[r])
                lam = lamh;
            else
                lam = laml;
            EnergyTypeT A = (EnergyTypeT)(lapE(vp1, vq1, vr1) * lam * MRFRatio);
            EnergyTypeT B = (EnergyTypeT)(lapE(vp1, vq1, vr2) * lam * MRFRatio);
            EnergyTypeT C = (EnergyTypeT)(lapE(vp1, vq2, vr1) * lam * MRFRatio);
            EnergyTypeT D = (EnergyTypeT)(lapE(vp1, vq2, vr2) * lam * MRFRatio);
            EnergyTypeT E = (EnergyTypeT)(lapE(vp2, vq1, vr1) * lam * MRFRatio);
            EnergyTypeT F = (EnergyTypeT)(lapE(vp2, vq1, vr2) * lam * MRFRatio);
            EnergyTypeT G = (EnergyTypeT)(lapE(vp2, vq2, vr1) * lam * MRFRatio);
            EnergyTypeT H = (EnergyTypeT)(lapE(vp2, vq2, vr2) * lam * MRFRatio);
//            printf("=========================================================\n");
//            printf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", vp1,vq1,vr1,lam,lapE(vp1,vq1,vr1),A);
//            printf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", vp1,vq1,vr2,lam,lapE(vp1,vq1,vr2),B);
//            printf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", vp1,vq2,vr1,lam,lapE(vp1,vq2,vr1),C);
//            printf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", vp1,vq2,vr2,lam,lapE(vp1,vq2,vr2),D);
//            printf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", vp2,vq1,vr1,lam,lapE(vp2,vq1,vr1),E);
//            printf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", vp2,vq1,vr2,lam,lapE(vp2,vq1,vr2),F);
//            printf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", vp2,vq2,vr1,lam,lapE(vp2,vq2,vr1),G);
//            printf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", vp2,vq2,vr2,lam,lapE(vp2,vq2,vr2),H);


            double pi = (A + D + F + G) - (B + C + E + H);
            if(pi >= 0){
                qpbo.AddPairwiseTerm(p,q,0,C-A,0,G-E);
                qpbo.AddPairwiseTerm(p,r,0,0,E-A,F-B);
                qpbo.AddPairwiseTerm(q,r,0,B-A,0,D-C);
                if(pi > 0) {
                    int w = qpbo.AddNode();
                    qpbo.AddUnaryTerm(w, A, A - (EnergyTypeT)pi);
                    qpbo.AddPairwiseTerm(p, w, 0, (EnergyTypeT)pi, 0, 0);
                    qpbo.AddPairwiseTerm(q, w, 0, (EnergyTypeT)pi, 0, 0);
                    qpbo.AddPairwiseTerm(r, w, 0, (EnergyTypeT)pi, 0, 0);
                }
            }else{
                qpbo.AddPairwiseTerm(p,q,B-D,0,F-H,0);
                qpbo.AddPairwiseTerm(p,r,C-G,D-H,0,0);
                qpbo.AddPairwiseTerm(q,r,E-F,0,G-H,0);
                int w = qpbo.AddNode();
                qpbo.AddUnaryTerm(w,H+(EnergyTypeT)pi,H);
                qpbo.AddPairwiseTerm(p, w, 0, 0, -1 * (EnergyTypeT)pi, 0);
                qpbo.AddPairwiseTerm(q, w, 0, 0, -1 * (EnergyTypeT)pi, 0);
                qpbo.AddPairwiseTerm(r, w, 0, 0, -1 * (EnergyTypeT)pi, 0);
            }
        };

        qpbo.AddNode(nPix);
        for(auto i=0; i<nPix; ++i) {
            qpbo.AddUnaryTerm(i, (EnergyTypeT)MRF_data[nLabel * i + (int) p1[i]], (EnergyTypeT)MRF_data[nLabel * i + (int) p2[i]]);
        }

        for(auto y=1; y<height-1; ++y){
            for(auto x=1; x<width-1; ++x) {
                addTripleToGraph(y * width + x - 1, y * width + x, y * width + x + 1);
                addTripleToGraph((y - 1) * width + x, y * width + x, (y + 1) * width + x);
            }
        }

        //solve
        float t = (float) getTickCount();
        qpbo.MergeParallelEdges();
        qpbo.Solve();
        qpbo.ComputeWeakPersistencies();

        //qpbo.Improve();

        //fusion
        float unlabeled = 0.0;
        float changed = 0.0;
        Depth orip1;
        orip1.initialize(width, height, -1);
        for(auto i=0; i<width * height; ++i)
            orip1.setDepthAtInd(i, p1[i]);
        for (auto i = 0; i < width * height; ++i) {
            int l = qpbo.GetLabel(i);
            double disp1 = orip1.getDepthAtInd(i);
            double disp2 = p2.getDepthAtInd(i);
            if (l == 0)
                p1.setDepthAtInd(i, disp1);
            else if (l < 0) {
                p1.setDepthAtInd(i, disp1);
                unlabeled += 1.0;
            }
            else {
                p1.setDepthAtInd(i, disp2);
                changed += 1.0;
            }
        }

        printf("Unlabeled pixels: %.2f, ratio: %.2f; label changed: %.2f, ratio: %.2f\n", unlabeled,
               unlabeled / (float)nPix,
               changed, changed / (float)nPix);
    }
}//namespace dynamic_stereo

