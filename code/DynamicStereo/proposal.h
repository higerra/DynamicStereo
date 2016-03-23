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
#include <memory>
#include "base/depth.h"
#include "base/file_io.h"
#include "model.h"
namespace dynamic_stereo {

    //interface for proposal creator
    class Proposal {
    public:
	    typedef int EnergyType;
	    Proposal(const FileIO& file_io_, std::shared_ptr<StereoModel<EnergyType> > model_):
			    file_io(file_io_), model(model_){
		    CHECK(model.get());
		    w = model->width;
		    h = model->height;
	    }
        virtual void genProposal(std::vector<Depth>& proposals) = 0;
    protected:
	    const FileIO& file_io;
	    std::shared_ptr<StereoModel<EnergyType> > model;
	    int w;
	    int h;
    };

    class ProposalSegPln: public Proposal{
    public:
        //constructor input:
        //  images_: reference image
        //  noisyDisp_: disparity map from only unary term
        //  num_proposal: number of proposal to generate. NOTE: currently fixed to 7
        ProposalSegPln(const FileIO& file_io_, std::shared_ptr<StereoModel<EnergyType> > model_, const Depth& noisyDisp_, const std::string& method_, const int num_proposal_ = 7);
        virtual void genProposal(std::vector<Depth>& proposals);
		virtual void segment(const int pid, std::vector<std::vector<int> >& seg)  = 0;
    protected:
        void fitDisparityToPlane(const std::vector<std::vector<int> >& seg, Depth& planarDisp, int id);
        //input:
        //  pid: id of parameter setting
        //  seg: stores the segmentation result. seg[i] stores pixel indices of region i

        const Depth& noisyDisp;
        const int num_proposal;
        std::vector<double> params;
        std::vector<double> mults;
	    const std::string method;
    };

    class ProposalSegPlnMeanshift: public ProposalSegPln{
    public:
	    ProposalSegPlnMeanshift(const FileIO& file_io_, std::shared_ptr<StereoModel<EnergyType> > model_, const Depth& noisyDisp_, const int num_proposal_ = 8);
		virtual void segment(const int pid, std::vector<std::vector<int> >& seg);
    protected:
    };

	class ProposalSegPlnGbSegment: public ProposalSegPln{
	public:
		ProposalSegPlnGbSegment(const FileIO& file_io_, std::shared_ptr<StereoModel<EnergyType> > model_, const Depth& noisyDisp_, const int num_proposal_ = 8);
		virtual void segment(const int pid, std::vector<std::vector<int> >& seg);
	};

	class ProposalSfM: public Proposal{
	public:
		ProposalSfM(const FileIO& file_io_, std::shared_ptr<StereoModel<EnergyType> > model_, const theia::Reconstruction& r_, const int anchor_, const double downsample_, const int segNum_ = 7);
		virtual void genProposal(std::vector<Depth>& proposals);
	protected:
		void segment(std::vector<std::vector<std::vector<int> > >& segs);
		const std::string method;
		std::vector<double> params;
		std::vector<double> mults;
		const int anchor;
		const double downsample;
		const int segNum;
		const theia::Reconstruction& reconstruction;
	};

}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_PROPOSAL_H
