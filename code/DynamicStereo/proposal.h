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
#include <theia/theia.h>
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
		virtual void segment(const int pid, std::vector<std::vector<int> >& seg)  = 0;
    protected:
        void fitDisparityToPlane(const std::vector<std::vector<int> >& seg, Depth& planarDisp, int id);
        //input:
        //  pid: id of parameter setting
        //  seg: stores the segmentation result. seg[i] stores pixel indices of region i

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
		virtual void segment(const int pid, std::vector<std::vector<int> >& seg);
    protected:
    };

	class ProposalSegPlnGbSegment: public ProposalSegPln{
	public:
		ProposalSegPlnGbSegment(const FileIO& file_io_, const cv::Mat& image_, const Depth& noisyDisp_, const int dispResolution_,
		                        const double min_disp_, const double max_disp_, const int num_proposal_ = 8);
		virtual void segment(const int pid, std::vector<std::vector<int> >& seg);
	};

	class ProposalSfM: public Proposal{
	public:
		ProposalSfM(const FileIO& file_io_, const cv::Mat& img_, const theia::Reconstruction& r_, const int anchor_,
					const int nLabel_, const double min_disp_, const double max_disp_, const double downsample_, const int segNum_ = 7);
		virtual void genProposal(std::vector<Depth>& proposals);
	protected:
		inline double depthToDisp(double depth){
			return (1.0 / depth * (double)nLabel - min_disp)/ (max_disp - min_disp);
		}
		inline double dispToDepth(double disp){
			return 1.0/(min_disp + disp * (max_disp - min_disp) / (double) nLabel);
		}

		void segment(std::vector<std::vector<std::vector<int> > >& segs);

		const FileIO& file_io;
		const std::string method;
		std::vector<double> params;
		std::vector<double> mults;
		const cv::Mat& image;
		const int anchor;
		const int w;
		const int h;
		const int nLabel;
		const double min_disp;
		const double max_disp;
		const double downsample;
		const int segNum;
		const theia::Reconstruction& reconstruction;
	};

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_PROPOSAL_H
