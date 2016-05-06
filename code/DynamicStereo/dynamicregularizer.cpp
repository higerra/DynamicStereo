//
// Created by yanhang on 4/29/16.
//

#include "dynamicregularizer.h"
#include "../base/utility.h"
#include "../base/thread_guard.h"
#include <Eigen/SparseCore>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{

    void dynamicRegularization(const std::vector<cv::Mat>& input,
                               const std::vector<std::vector<Eigen::Vector2d> >& segments,
                               std::vector<cv::Mat>& output, const double weight_smooth){
	    CHECK(!input.empty());
	    const int width = input[0].cols;
	    const int height = input[0].rows;
	    const int channel = input[0].channels();
	    const int N = (int)input.size();

	    output.resize(input.size());
	    for(auto i=0; i<input.size(); ++i)
		    output[i] = input[i].clone();

        vector<uchar *> inputPtr(input.size(), NULL);
	    vector<uchar *> outputPtr(output.size(), NULL);
	    for(auto i=0; i<input.size(); ++i)
		    inputPtr[i] = input[i].data;
	    for(auto i=0; i<output.size(); ++i)
		    outputPtr[i] = output[i].data;

	    //prepare output

	    const double huber_theta = 10;
        auto threadFun = [&](const int tid, const int num_thread) {
	        vector<vector<double> > DP(N);
	        for (auto &d: DP)
		        d.resize(256, 0.0);
	        vector<vector<int> > backTrack(N);
	        for (auto &b: backTrack)
		        b.resize(256, 0);
	        for (auto sid = tid; sid < segments.size(); sid += num_thread) {
		        printf("Smoothing segment %d on thread %d, kPix:%d\n", sid, tid, (int)segments[sid].size());
		        for (auto i = 0; i < segments[sid].size(); ++i) {
			        const int x = (int)segments[sid][i][0];
			        const int y = (int)segments[sid][i][1];
			        for (auto c = 0; c < channel; ++c) {
				        const int pixId = channel * (y * width + x) + c;
				        //reinitialize tables
				        for (auto &d: DP) {
					        for (auto &dd: d)
						        dd = 0.0;
				        }
				        for (auto &b: backTrack) {
					        for (auto &bb: b)
						        bb = 0;
				        }
				        //start DP
				        for (auto p = 0; p < 256; ++p)
					        DP[0][p] = math_util::huberNorm((double) inputPtr[0][pixId] - (double) p,
					                                        huber_theta);
				        for (auto v = 1; v < input.size(); ++v) {
					        for (auto p = 0; p < 256; ++p) {
						        DP[v][p] = numeric_limits<double>::max();
						        double mdata;
						        if(input[v].at<Vec3b>(y,x) != Vec3b(0,0,0)) {
							        mdata = math_util::huberNorm((double) inputPtr[v][pixId] - (double) p,
							                                     huber_theta);
						        }
						        else
							        mdata = 0;
						        for (auto pf = 0; pf < 256; ++pf) {
							        double curv = DP[v - 1][pf] + mdata + weight_smooth * math_util::huberNorm(
									        (double) pf - (double) p, huber_theta);
							        if (curv < DP[v][p]) {
								        DP[v][p] = curv;
								        backTrack[v][p] = pf;
							        }
						        }
					        }
				        }
				        //back track
				        //last frame
				        double minE = std::numeric_limits<double>::max();
				        for (auto p = 0; p < 256; ++p) {
					        if (DP[N - 1][p] < minE) {
						        minE = DP[N - 1][p];
						        outputPtr[N - 1][pixId] = (uchar) p;
					        }
				        }
				        for (auto v = N - 2; v >= 0; --v) {
					        outputPtr[v][pixId] =
							        (uchar) backTrack[v + 1][outputPtr[v + 1][pixId]];
				        }
			        }
		        }
	        }
        };


	    const int num_thread = 6;
	    vector<thread_guard> threads((size_t)num_thread);
	    for(auto tid=0; tid<num_thread; ++tid){
		    std::thread t(threadFun, tid, num_thread);
		    threads[tid].bind(t);
	    }
	    for(auto &t: threads)
		    t.join();
    }


	void regularizationPoisson(const vector<Mat>& input,
							   const vector<vector<Eigen::Vector2d> >& segments, const cv::Mat& inputMask,
							   vector<Mat>& output, const double weight_smooth){
		CHECK(!input.empty());
		CHECK_EQ(inputMask.cols, input[0].cols);
		CHECK_EQ(inputMask.rows, input[0].rows);

		output.resize(input.size());
		for(auto i=0; i<output.size(); ++i)
			output[i] = input[i].clone();
		Vec3b invalidToken(0,0,0);

		for(auto sid=0; sid<segments.size(); ++sid){
			const vector<Vector2d>& curSeg = segments[sid];
			const int kVar = (int)(curSeg.size() * input.size());
			printf("Smoothing segment %d, kVar:%d\n", sid, kVar);
			SparseMatrix<double> A(kVar, kVar);
			VectorXd b(kVar);
			vector<Triplet<double> > triplets;
			triplets.reserve((size_t)kVar * 4);
			for(auto i=0; i<curSeg.size(); ++i){
				double leftv = 0, rightv = 0;

			}
		}

	}
}//namespace dynamic_stereo