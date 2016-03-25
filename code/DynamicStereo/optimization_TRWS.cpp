//
// Created by yanhang on 3/3/16.
//
#include <map>
#include "optimization.h"
#include "external/segment_ms/msImageProcessor.h"
#include "external/TRW_S/MRFEnergy.h"
#include "external/TRW_S/typeTruncatedLinear.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    SecondOrderOptimizeTRWS::SecondOrderOptimizeTRWS(const FileIO& file_io_, const int kFrames_, shared_ptr<StereoModel<EnergyType> > model_):
            StereoOptimization(file_io_, kFrames_, model_), trun(4){
        segment_ms::msImageProcessor ms_segmentator;
	    const Mat& image = model->image;
        ms_segmentator.DefineBgImage(image.data, segment_ms::COLOR, image.rows, image.cols);
        const int hs = 4;
        const float hr = 5.0f;
        const int min_a = 40;
        ms_segmentator.Segment(hs, hr, min_a, meanshift::SpeedUpLevel::MED_SPEEDUP);
        refSeg.resize((size_t)image.cols * image.rows);
        const int * labels = ms_segmentator.GetLabels();
        for(auto i=0; i<image.cols * image.rows; ++i)
            refSeg[i] = labels[i];

        laml = 0.02;
        lamh = 0.2;
    }

    double SecondOrderOptimizeTRWS::evaluateEnergy(const Depth &disp) const {
        CHECK_EQ(disp.getWidth(), width);
        CHECK_EQ(disp.getHeight(), height);
        double e = 0.0;
        for (auto i = 0; i < width * height; ++i) {
            int l = (int) disp[i];
            e += ((double)model->operator()(i, l) / model->MRFRatio);
        }
        auto tripleE = [&](int id1, int id2, int id3, double w){
            double lam = w * model->weight_smooth;
//            if (refSeg[id1] == refSeg[id2] && refSeg[id1] == refSeg[id3])
//                lam = lamh;
//            else
//                lam = laml;
            return lapE(disp[id1], disp[id2], disp[id3]) * lam;
        };

        for (auto x = 1; x < width - 1; ++x) {
            for (auto y = 1; y < height - 1; ++y) {
                e += tripleE(y * width + x - 1, y * width + x, y * width + x + 1, model->hCue[y*width+x]);
                e += tripleE((y - 1) * width + x, y * width + x, (y + 1) * width + x, model->vCue[y*width+x]);
            }
        }
        return e;
    }

    void SecondOrderOptimizeTRWS::optimize(Depth &result, const int max_iter) const {
        char buffer[1024] = {};
        const int kPix = width * height;

        typedef TypeBinary SmoothT;
        typedef MRFEnergy<SmoothT> TRWS;
        typedef double EnergyTypeT;

	    const int& nLabel = model->nLabel;
	    //const double& MRFRatio = model->MRFRatio;

        vector<double> labels((size_t) nLabel);
        for (auto i = 0; i < nLabel; ++i)
            labels[i] = i;
//        random_shuffle(labels.begin(), labels.end());

        result.initialize(width, height, labels[0]);

        double initE = evaluateEnergy(result);
        float start_time = (float) getTickCount();
        //lambda function to add triple term
        auto addTripleToGraph = [&](int p, int q, int r, double l, double w, shared_ptr<TRWS> mrf, shared_ptr<TRWS::NodeId> nodes) {
            double vp1 = result[p], vq1 = result[q], vr1 = result[r];
            double lam = w * model->weight_smooth;
//            if (refSeg[p] == refSeg[q] && refSeg[p] == refSeg[r])
//                lam = lamh;
//            else
//                lam = laml;
            EnergyTypeT A = (EnergyTypeT)(lapE(vp1, vq1, vr1) * lam);
            EnergyTypeT B = (EnergyTypeT)(lapE(vp1, vq1, l) * lam);
            EnergyTypeT C = (EnergyTypeT)(lapE(vp1, l, vr1) * lam);
            EnergyTypeT D = (EnergyTypeT)(lapE(vp1, l, l) * lam);
            EnergyTypeT E = (EnergyTypeT)(lapE(l, vq1, vr1) * lam);
            EnergyTypeT F = (EnergyTypeT)(lapE(l, vq1, l) * lam);
            EnergyTypeT G = (EnergyTypeT)(lapE(l, l, vr1) * lam);
            EnergyTypeT H = (EnergyTypeT)(lapE(l, l, l) * lam);

            EnergyTypeT pi = (A + D + F + G) - (B + C + E + H);

            if (pi >= 0) {
                mrf->AddEdge(nodes.get()[p], nodes.get()[q], SmoothT::EdgeData(0, C - A, 0, G - E));
                mrf->AddEdge(nodes.get()[p], nodes.get()[r], SmoothT::EdgeData(0, 0, E - A, F - B));
                mrf->AddEdge(nodes.get()[q], nodes.get()[r], SmoothT::EdgeData(0, B - A, 0, D - C));
                if (pi > 0) {
                    TRWS::NodeId nid = mrf->AddNode(SmoothT::LocalSize(), SmoothT::NodeData(A, A - pi));
                    mrf->AddEdge(nodes.get()[p], nid, SmoothT::EdgeData(0, pi, 0, 0));
                    mrf->AddEdge(nodes.get()[q], nid, SmoothT::EdgeData(0, pi, 0, 0));
                    mrf->AddEdge(nodes.get()[r], nid, SmoothT::EdgeData(0, pi, 0, 0));
                }
            } else {
                mrf->AddEdge(nodes.get()[p], nodes.get()[q], SmoothT::EdgeData(B - D, 0, F - H, 0));
                mrf->AddEdge(nodes.get()[p], nodes.get()[r], SmoothT::EdgeData(C - G, D - H, 0, 0));
                mrf->AddEdge(nodes.get()[q], nodes.get()[r], SmoothT::EdgeData(E - F, 0, G - H, 0));
                TRWS::NodeId nid = mrf->AddNode(SmoothT::LocalSize(), SmoothT::NodeData(H + pi, H));
                mrf->AddEdge(nodes.get()[p], nid, SmoothT::EdgeData(0, 0, -1 * pi, 0));
                mrf->AddEdge(nodes.get()[q], nid, SmoothT::EdgeData(0, 0, -1 * pi, 0));
                mrf->AddEdge(nodes.get()[r], nid, SmoothT::EdgeData(0, 0, -1 * pi, 0));
            }
        };

        for (auto lid = 1; lid < labels.size(); ++lid) {
            //solve nonsubmodular expansion with TRWS
            const double l = labels[lid];
            printf("================================\n");
            printf("Expanding label %d (%d/%d) with TRWS\n", (int)l, lid, nLabel);
            float start_t = (float) getTickCount();
            double inite = evaluateEnergy(result);
            std::shared_ptr<TRWS> mrf(new TRWS(SmoothT::GlobalSize()));
            std::shared_ptr<TRWS::NodeId> nodes(new TRWS::NodeId[kPix]);
            //unary term
            int nodeIdx = 0;
            for (auto i = 0; i < kPix; ++i, ++nodeIdx) {
                EnergyTypeT e1 = (EnergyTypeT) model->operator()(i, (int) result[i]) / model->MRFRatio;
	            EnergyTypeT e2 = (EnergyTypeT) model->operator()(i, (int)l) / model->MRFRatio;
                nodes.get()[nodeIdx] = mrf->AddNode(SmoothT::LocalSize(), SmoothT::NodeData(e1, e2));
            }

            //add triple term
            for (auto x = 1; x < width - 1; ++x) {
                for (auto y = 1; y < height - 1; ++y) {
                    addTripleToGraph(y * width + x - 1, y * width + x, y * width + x + 1, l, model->hCue[y*width+x], mrf, nodes);
                    addTripleToGraph((y - 1) * width + x, y * width + x, (y + 1) * width + x, l, model->vCue[y*width+x], mrf, nodes);
                }
            }

            TypeBinary::REAL energy, lowerBound;
            TRWS::Options option;
            option.m_iterMax = 500;
            mrf->Minimize_TRW_S(option, lowerBound, energy);

            for (auto i = 0; i < kPix; ++i) {
                if (mrf->GetSolution(nodes.get()[i]) > 0)
                    result[i] = l;
            }
            cout << endl;

            double e = evaluateEnergy(result);
            printf("Done, initial energy: %.3f, final energy: %.3f(%.3f by TRWS), lower bound: %.3f, time usage:%.2fs\n", inite, e,
                   energy, lowerBound,
                   ((float) getTickCount() - start_t) / (float) getTickFrequency());

            sprintf(buffer, "%s/temp/TRWS_label%05d.jpg", file_io.getDirectory().c_str(), (int)l);
            result.saveImage(buffer, 256.0 / (double) nLabel);

            mrf.reset();
            nodes.reset();
        }

        printf("All done. Initial energy: %.3f, final energy: %.3f, time usage: %.2fs\n", initE, evaluateEnergy(result),
               ((float) getTickCount() - start_time) / (float) getTickFrequency());

    }

}//namespace dynamic_stereo

