//
// Created by yanhang on 3/3/16.
//

#include "optimization.h"
#include "external/segment_ms/msImageProcessor.h"
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/operations/adder.hxx>

#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>

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

    void SecondOrderOptimizeTRBP::optimize(Depth &result, const int max_iter) const{
        char buffer[1024] = {};
        //formulate problem with OpenGM
        typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
        typedef opengm::GraphicalModel<EnergyType, opengm::Adder, opengm::ExplicitFunction<EnergyType>, Space> Model;
        typedef opengm::TrbpUpdateRules<Model, opengm::Maximizer> UpdateRules;
        typedef opengm::MessagePassing<Model, opengm::Maximizer, UpdateRules, opengm::MaxDistance> TRBP;

        Space space(width * height, nLabel);
        Model gm(space);
	    //add unaryterms
	    
    }

    double SecondOrderOptimizeTRBP::evaluateEnergy(const Depth& disp) const {
        return 0.0;
    }

}//namespace dynamic_stereo
