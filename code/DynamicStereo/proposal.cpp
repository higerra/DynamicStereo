//
// Created by yanhang on 2/28/16.
//

#include "proposal.h"
#include "base/plane3D.h"
#include "external/segment_ms/msImageProcessor.h"

namespace dynamic_stereo{
    ProposalSegPln::ProposalSegPln(const FileIO& file_io_, const cv::Mat &image_, const Depth &noisyDisp_, const int dispResolution_,
                                   const int num_proposal_):file_io(file_io_), noisyDisp(noisyDisp_), image(image_),
                                                            dispResolution(dispResolution_), num_proposal(num_proposal_),
                                                            w(image.cols), h(image.rows){
        CHECK_EQ(num_proposal, 7) << "num_proposal should be 7";
        params.resize(4);
        params[0] = 1; params[1] = 1.5; params[2] = 10; params[3] = 100;
        //current fix num_proposal to 7

    }

    void ProposalSegPln::fitDisparityToPlane(const std::vector<std::vector<int> >& seg, Depth& planarDisp) {
        const int w = noisyDisp.getWidth();
        const int h = noisyDisp.getHeight();
        const int nPixels = w * h;
        const double min_disp = (double)0.0;
        const double dis_thres = 0.01;
        planarDisp = noisyDisp;
        for(const auto& idxs: seg){
            std::vector<Eigen::Vector3d> pts;
            for(const auto idx: idxs){
                CHECK_LT(idx, nPixels);
                double curdisp = noisyDisp.getDepthAtInd(idx);
                if(curdisp < min_disp)
                    continue;
                pts.push_back(Eigen::Vector3d(idx/w, idx%w, 1.0/curdisp));
            }
            if(pts.size() < 3)
                continue;

            //solve for plane
            Plane3D segPln;
            plane_util::planeFromPointsRANSAC(pts, segPln, dis_thres);
            double offset = segPln.getOffset();
            Eigen::Vector3d n = segPln.getNormal();
            //modify disparity value according to plane
            for(const auto idx: idxs){
                int x = idx % w;
                int y = idx / w;
                double d = (-1 * offset - n[0] * x - n[1] * y) / n[2];
                planarDisp.setDepthAtInd(idx, std::min(1.0/d, (double)dispResolution));
            }
        }
    }

    void ProposalSegPln::genProposal(std::vector<Depth> &proposals) {
        proposals.resize((size_t)num_proposal);
        for(auto i=0; i<num_proposal; ++i){
            printf("Proposal %d\n", i);
            std::vector<std::vector<int> > seg;
            printf("Segmenting...\n");
            segment(i, seg);
            printf("Fitting disparity to plane...\n");
            fitDisparityToPlane(seg, proposals[i]);
        }
    }

    ProposalSegPlnMeanshift::ProposalSegPlnMeanshift(const FileIO& file_io_, const cv::Mat &image_, const Depth& noisyDisp_, const int dispResolution_, const int num_proposal_):
            ProposalSegPln(file_io_, image_, noisyDisp_, dispResolution, num_proposal_){
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
