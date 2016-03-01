//
// Created by yanhang on 2/28/16.
//

#include "proposal.h"
#include "base/plane3D.h"
#include "external/segment_ms/msImageProcessor.h"

namespace dynamic_stereo{
    ProposalSegPln::ProposalSegPln(const FileIO& file_io_, const cv::Mat &image_, const Depth &noisyDisp_, const int dispResolution_,
                                   const double min_disp_, const double max_disp_, const int num_proposal_):file_io(file_io_), noisyDisp(noisyDisp_), image(image_),
                                                            dispResolution(dispResolution_), min_disp(min_disp_), max_disp(max_disp_), num_proposal(num_proposal_),
                                                            w(image.cols), h(image.rows){
        CHECK_EQ(num_proposal, 7) << "num_proposal should be 7";
        params.resize(4);
        params[0] = 1; params[1] = 1.5; params[2] = 10; params[3] = 100;
        //current fix num_proposal to 7

    }

    void ProposalSegPln::fitDisparityToPlane(const std::vector<std::vector<int> >& seg, Depth& planarDisp, int id) {
        const int w = noisyDisp.getWidth();
        const int h = noisyDisp.getHeight();
        const int nPixels = w * h;
        const double epsilon = (double)1e-05;
        const double dis_thres = 2;

        const double max_dim = 255.0;
        const double min_depth = 1.0 / max_disp;
        const double max_depth = 1.0 / min_disp;

        //lambda functions to map between disparity and depth. Depth are rescaled to 0~max_dim for numerical stability
        auto dispToDepth = [=](const double dispv){
            double d2 = (dispv * (max_disp - min_disp) / dispResolution + min_disp);
            CHECK_NE(d2, 0);
            double d = 1.0 / d2;
            return (d - min_depth)  / (max_depth - min_depth) * (double)max_dim;
        };

        auto depthToDisp = [=](const double depthv){
            double d = depthv / (double)max_dim * (max_depth - min_depth) + min_depth;
            CHECK_NE(d, 0);
            return ((1.0 / d) - min_disp) * dispResolution / (max_disp - min_disp);
        };


        planarDisp = noisyDisp;
        for(auto i=0; i<planarDisp.getRawData().size(); ++i)
            planarDisp.getRawData()[i] = dispToDepth(noisyDisp.getRawData()[i]);

	    //unit testing for lambda function
	    for(double testdisp = 0.0; testdisp<dispResolution; testdisp+=1.0) {
		    double testdisp2 = depthToDisp(dispToDepth(testdisp));
		    CHECK_LT(std::abs(testdisp2 - testdisp), epsilon);
	    }

	    int tx = -1, ty = -1;

        for(const auto& idxs: seg){
	        bool verbose = false;
            std::vector<Eigen::Vector3d> pts;
            for(const auto idx: idxs){
                CHECK_LT(idx, nPixels);
                double curdisp = noisyDisp.getDepthAtInd(idx);
                pts.push_back(Eigen::Vector3d(idx%w, idx/w, dispToDepth(curdisp)));
	            if(idx == ty*w + tx)
		            verbose = true;
            }
            if(pts.size() < 3)
                continue;
	        //check if all pixels are located colinear
	        Eigen::MatrixXd A(pts.size(), 3);
	        for(auto i=0; i<pts.size(); ++i){
		        A(i,0) = pts[i][0];
		        A(i,1) = pts[i][1];
		        A(i,2) = 1.0;
	        }
	        Eigen::Matrix3d A2 = A.transpose() * A;
	        double A2det = A2.determinant();
	        if(A2det < epsilon) {
		        if(verbose){
			        std::cout << "Points colinear" << std::endl;
			        for(int i=0; i<pts.size(); ++i)
				        printf("(%.2f,%.2f,%.2f), depth: %.2f)\n", pts[i][0], pts[i][1], pts[i][2], planarDisp.getDepthAtInt((int)pts[i][0], (int)pts[i][1]));
			        std::cout << "Det: " << A2det << std::endl;
		        }
		        continue;
	        }

	        //solve for plane
            Plane3D segPln;
            if(!plane_util::planeFromPointsRANSAC(pts, segPln, dis_thres, 1000, verbose)) {
	            if(verbose)
		            std::cout << "plane from points returns false" << std::endl;
	            continue;
            }

            double offset = segPln.getOffset();
            Eigen::Vector3d n = segPln.getNormal();
//	        if(verbose) {
//		        for(auto i=0; i<pts.size(); ++i)
//			        std::cout << pts[i][0] << ' ' << pts[i][1] << ' ' << pts[i][2] << std::endl;
//		        printf("Optimal plane: (%.5f,%.5f,%.5f), %.5f\n", n[0], n[1], n[2], offset);
//	        }

	        if(std::abs(n[2]) < epsilon){
//		        for(int i=0; i<pts.size(); ++i)
//			        std::cout << pts[i][0] << ' ' << pts[i][1] << ' ' << pts[i][2] << std::endl;
//		        std::cout << A2 << std::endl;
//		        std::cout << "Det: " << A2det << std::endl;
//		        printf("Optimal plane: (%.5f,%.5f,%.5f), %.5f\n", n[0], n[1], n[2], offset);
		        //CHECK_GE(std::abs(n[2]), epsilon);
		        continue;
	        }

            //modify disparity value according to plane
            for(const auto idx: idxs){
                int x = idx % w;
                int y = idx / w;
                double newdepth = (-1 * offset - n[0]*x - n[1] * y) / n[2];
                double d = std::max(std::min(newdepth, (double)max_dim), 0.0);
	            if(verbose) {
		            double oridepth = dispToDepth(noisyDisp.getDepthAtInd(idx));
		            printf("inter: (%d,%d,%.5f), ori disp: %.5f, new disp: %.5f\n", x, y, oridepth, oridepth, newdepth);
	            }
                //planarDisp.setDepthAtInd(idx, std::max(std::min(depthToDisp(d), (double)dispResolution), 0.0));
	            planarDisp.setDepthAtInd(idx, d);
            }
        }

	    Depth tempd;
	    tempd.initialize(w, h, 0.0);
	    for(auto i=0; i<nPixels; ++i)
		    tempd.getRawData()[i] = dispToDepth(noisyDisp.getRawData()[i]);
	    char buffer[1024] = {};
	    sprintf(buffer, "%s/temp/tdepth%03d_1.jpg", file_io.getDirectory().c_str(), id);
	    tempd.saveImage(std::string(buffer));
	    sprintf(buffer, "%s/temp/tdepth%03d_2.jpg", file_io.getDirectory().c_str(), id);
	    planarDisp.saveImage(std::string(buffer));
    }

    void ProposalSegPln::genProposal(std::vector<Depth> &proposals) {
        proposals.resize((size_t)num_proposal);
        for(auto i=0; i<proposals.size(); ++i){
            printf("Proposal %d\n", i);
            std::vector<std::vector<int> > seg;
            printf("Segmenting...\n");
            segment(i, seg);
            printf("Fitting disparity to plane...\n");
            fitDisparityToPlane(seg, proposals[i], i);
        }
    }

    ProposalSegPlnMeanshift::ProposalSegPlnMeanshift(const FileIO& file_io_, const cv::Mat &image_, const Depth& noisyDisp_,
                                                     const int dispResolution_, const double min_disp_, const double max_disp_,const int num_proposal_):
            ProposalSegPln(file_io_, image_, noisyDisp_, dispResolution_, min_disp_, max_disp_, num_proposal_){
        mults.resize((size_t)num_proposal);
        for(auto i=0; i<mults.size(); ++i)
            mults[i] = (double)i+1;
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

        cv::Vec3b colorTable[] = {cv::Vec3b(255, 0, 0), cv::Vec3b(0, 255, 0), cv::Vec3b(0, 0, 255), cv::Vec3b(255, 255, 0),
                                  cv::Vec3b(255, 0, 255), cv::Vec3b(0, 255, 255), cv::Vec3b(128,128,0), cv::Vec3b(128,0,128), cv::Vec3b(0,128,128)};
        cv::Mat segTestRes(h, w, image.type());
        for (auto y = 0; y < h; ++y) {
            for (auto x = 0; x < w; ++x) {
                int l = labels[y * w + x];
                segTestRes.at<cv::Vec3b>(y,x) = colorTable[l%9];
            }
        }
        char buffer[1024] = {};
        sprintf(buffer, "%s/temp/meanshift%03d.jpg", file_io.getDirectory().c_str(), pid);
        imwrite(buffer, segTestRes);

        for(auto i=0; i<w*h; ++i){
            int l = labels[i];
            CHECK_LT(l, seg.size());
            seg[l].push_back(i);
        }
    }
}
