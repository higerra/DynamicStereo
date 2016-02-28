//
// Created by yanhang on 2/28/16.
//

#include "proposal.h"
#include "external/segment_ms/msImageProcessor.h"

namespace dynamic_stereo{
    ProposalSegPln::ProposalSegPln(const cv::Mat &image_, const Depth &noisyDisp_,
                                   const int num_proposal_):noisyDisp(noisyDisp_), image(image_), num_proposal(num_proposal_),
                                                            w(image.cols), h(image.rows){
        CHECK_EQ(num_proposal, 7) << "num_proposal should be 7";
        params.resize(4);
        params[0] = 1; params[1] = 1.5; params[2] = 10; params[3] = 100;
        //current fix num_proposal to 7

    }

    void ProposalSegPln::fitDisparityToPlane(const std::vector<std::vector<int> >& seg, Depth& planarDisp) {

    }

    void ProposalSegPln::genProposal(std::vector<Depth> &proposals) {
        proposals.resize((size_t)num_proposal);
        for(auto i=0; i<num_proposal; ++i){
            std::vector<std::vector<int> > seg;
            segment(i, seg);
            fitDisparityToPlane(seg, proposals[i]);
        }
    }

    ProposalSegPlnMeanshift::ProposalSegPlnMeanshift(const cv::Mat &image_, const Depth& noisyDisp_, const int num_proposal_):
            ProposalSegPln(image_, noisyDisp_, num_proposal_){
        mults.resize((size_t)num_proposal);
        for(auto i=0; i<mults.size(); ++i)
            mults[i] = (double)i;
    }

    void ProposalSegPlnMeanshift::segment(const int pid, std::vector<std::vector<int> > &seg) {
        CHECK_LT(pid, mults.size());
        CHECK_GE(pid, 0);
        segment_ms::msImageProcessor segmentor;
        segmentor.DefineImage(image.data, segment_ms::COLOR, image.rows, image.cols);
        segmentor.Segment((int)(params[0]*mults[pid]), (float)(params[1]*mults[pid]), (int)(params[2]*mults[pid]), meanshift::MED_SPEEDUP);

        int* labels = NULL;
        labels = segmentor.GetLabels();
        CHECK(labels != NULL);
        seg.resize((size_t) segmentor.GetRegionCount());
        for(auto i=0; i<w*h; ++i){
            int l = labels[i];
            CHECK_LT(l, seg.size());
            seg[l].push_back(i);
        }
    }
}
