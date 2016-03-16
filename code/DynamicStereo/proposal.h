//
// Created by yanhang on 2/28/16.
//

#ifndef DYNAMICSTEREO_PROPOSAL_H
#define DYNAMICSTEREO_PROPOSAL_H

#include <vector>
#include <iostream>
#include <string>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "base/depth.h"
#include "base/file_io.h"
namespace dynamic_stereo {

    //interface for proposal creator
    class Proposal {
    public:
        virtual void genProposal(std::vector<Depth>& proposals) = 0;
    };

    class ProposalSegPln: public Proposal{
    public:
        //constructor input:
        //  images_: reference image
        //  noisyDisp_: disparity map from only unary term
        //  num_proposal: number of proposal to generate. NOTE: currently fixed to 7
        ProposalSegPln(const FileIO& file_io_, const cv::Mat& image_, const Depth& noisyDisp_, const int dispResolution_,
                       const double min_disp_, const double max_disp_, const std::string& method_, const int num_proposal_ = 7);
        virtual void genProposal(std::vector<Depth>& proposals);
    protected:
        void fitDisparityToPlane(const std::vector<std::vector<int> >& seg, Depth& planarDisp, int id);

        //input:
        //  pid: id of parameter setting
        //  seg: stores the segmentation result. seg[i] stores pixel indices of region i
        virtual void segment(const int pid, std::vector<std::vector<int> >& seg)  = 0;

        const FileIO& file_io;
        const Depth& noisyDisp;
        const cv::Mat& image;
        const int num_proposal;
        std::vector<double> params;
        std::vector<double> mults;

        const int w;
        const int h;

        const int dispResolution;
	    const double min_disp;
	    const double max_disp;

	    const std::string method;
    };

    class ProposalSegPlnMeanshift: public ProposalSegPln{
    public:
	    ProposalSegPlnMeanshift(const FileIO& file_io_, const cv::Mat& image_, const Depth& noisyDisp_, const int dispResolution_,
	                   const double min_disp_, const double max_disp_, const int num_proposal_ = 8);
    protected:
        virtual void segment(const int pid, std::vector<std::vector<int> >& seg);
    };

	class ProposalSegPlnGbSegment: public ProposalSegPln{
	public:
		ProposalSegPlnGbSegment(const FileIO& file_io_, const cv::Mat& image_, const Depth& noisyDisp_, const int dispResolution_,
		                        const double min_disp_, const double max_disp_, const int num_proposal_ = 8);
	protected:
		virtual void segment(const int pid, std::vector<std::vector<int> >& seg);
	};



}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_PROPOSAL_H
