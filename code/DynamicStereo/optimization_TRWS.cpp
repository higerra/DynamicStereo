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

    SecondOrderOptimizeTRWS::SecondOrderOptimizeTRWS(const FileIO& file_io_, const int kFrames_,const cv::Mat& image_, const std::vector<EnergyType> &MRF_data_,
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

    void SecondOrderOptimizeTRWS::optimize(Depth &result, const int max_iter) const {
//        char buffer[1024] = {};
//        const int kPix = width * height;
//
//        typedef TypeTruncatedLinear SmoothT;
//        typedef MRFEnergy<SmoothT> MRF;
//
//        std::shared_ptr<MRF> mrf(new MRF(SmoothT::GlobalSize(nLabel)));
//        vector<MRF::NodeId> nodes(4 * (size_t)kPix);
//        vector<SmoothT::REAL> D((size_t)nLabel);
//
//        //unary term
//        int nodeIdx = 0;
//        for(auto i=0; i<kPix; ++i, ++nodeIdx){
//            for(auto d=0; d<nLabel; ++d)
//                D[d] = MRF_data[i * nLabel + d] / MRFRatio;
//            nodes[nodeIdx] = mrf->AddNode(SmoothT::LocalSize(), SmoothT::NodeData(D.data()));
//        }
//
//        //smoothness term
//
//
//
//        result.initialize(width, height, -1);
//        for(auto i=0; i<width * height; ++i)
//            result.setDepthAtInd(i, 0.0);
    }

    double SecondOrderOptimizeTRWS::evaluateEnergy(const Depth& disp) const {
        return 0.0;
    }

}//namespace dynamic_stereo

