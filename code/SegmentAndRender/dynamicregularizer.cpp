//
// Created by yanhang on 4/29/16.
//

#include "dynamicregularizer.h"
#include "../base/utility.h"
#include "../base/thread_guard.h"
#include "../base/depth.h"
#include <Eigen/Sparse>
#include <Eigen/SPQRSupport>

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
							   const vector<vector<Eigen::Vector2d> >& segments,
							   vector<Mat>& output, const double ws, const double wt){
		CHECK(!input.empty());

		using Triplet = Eigen::Triplet<double>;

		const int width = input[0].cols;
		const int height = input[0].rows;
		const int kPix = width * height;
		const int chn = input[0].channels();

		output.resize(input.size());
		for(auto i=0; i<output.size(); ++i)
			output[i] = input[i].clone();
		Vec3b invalidToken(0,0,0);

		auto threadFunc = [&](const int tid, const int num_thread){
			for(auto i=tid; i<1; i+=num_thread){
				printf("Optimizing segment %d(%d) on thread %d\n", tid, (int)segments.size(), tid);
				cout << "init..." << endl << flush;
				const vector<Vector2d>& segment = segments[i];
				const int segSize = (int)segment.size();

				Mat idMap(height, width, CV_32SC1, Scalar::all(-1));
				for(auto pid=0; pid<segment.size(); ++pid)
					idMap.at<int>((int)segment[pid][1], (int)segment[pid][0]) = pid;

				const int kVar = segSize * (int)input.size();
				int kConstraint = 0;
				vector<vector<Triplet> > triplets((size_t)chn);
				vector<vector<double> > rhs((size_t)chn);
				for(auto c=0; c<chn; ++c) {
					triplets[c].reserve((size_t) (kVar + (kVar * 3) * 3));
					rhs[c].reserve((size_t) (kVar * 4));
				}

				cout << "data constraint..." << endl << flush;
				//data constraint
				for(auto pid=0; pid<segment.size(); ++pid){
					const int x = (int)segment[pid][0];
					const int y = (int)segment[pid][1];
					for(auto v=0; v<input.size(); ++v){
						Vec3b curpix = input[v].at<Vec3b>(y,x);
						if(curpix == invalidToken)
							continue;
						for(auto c=0; c<chn; ++c){
							triplets[c].push_back(Triplet((double)kConstraint, (double)(segSize * v + pid), 1.0));
							rhs[c].push_back((double)curpix[c]);
						}
						kConstraint++;
					}
				}

				cout << "adding smoothness constraint" << endl << flush;
				//smoothness constraint
				for(auto pid=0; pid<segment.size(); ++pid){
					const int x = (int)segment[pid][0];
					const int y = (int)segment[pid][1];

					for(auto v=0; v<input.size(); ++v){
						Vector3d rightV(0,0,0);
						double leftV = 0.0;
						//x direction
						if(x > 0){
							int neiId = idMap.at<int>(y,x-1);
							if(neiId >= 0){
								for(auto c=0; c<chn; ++c)
									triplets[c].push_back(Triplet((double)kConstraint, (double)(segSize*v+neiId), -1 * ws));
							}else{
								Vec3b curPix = input[v].at<Vec3b>(y,x-1);
								for(auto c=0; c<chn; ++c)
									rightV[c] += (double)curPix[c] * ws;
							}
							leftV += ws;
						}
						if(x < width - 1){
							int neiId = idMap.at<int>(y,x+1);
							if(neiId >= 0){
								for(auto c=0; c<chn; ++c)
									triplets[c].push_back(Triplet((double)kConstraint, (double)(segSize*v+neiId), -1 * ws));
							}else{
								Vec3b curPix = input[v].at<Vec3b>(y,x+1);
								for(auto c=0; c<chn; ++c)
									rightV[c] += (double)curPix[c] * ws;
							}
							leftV += ws;
						}
						if(leftV > 0){
							for(auto c=0; c<chn; ++c){
								triplets[c].push_back(Triplet((double)kConstraint, (double)(segSize*v+pid), leftV));
								rhs[c].push_back(rightV[c]);
							}
							kConstraint++;
						}

						//y direction
						rightV = Vector3d(0,0,0);
						leftV = 0.0;
						if(y > 0){
							int neiId = idMap.at<int>(y-1,x);
							if(neiId >= 0){
								for(auto c=0; c<chn; ++c)
									triplets[c].push_back(Triplet((double)kConstraint, (double)(segSize*v+neiId), -1 * ws));
							}else{
								Vec3b curPix = input[v].at<Vec3b>(y-1,x);
								for(auto c=0; c<chn; ++c)
									rightV[c] += (double)curPix[c] * ws;
							}
							leftV += ws;
						}
						if(y < height - 1){
							int neiId = idMap.at<int>(y+1,x);
							if(neiId >= 0){
								for(auto c=0; c<chn; ++c)
									triplets[c].push_back(Triplet((double)kConstraint, (double)(segSize*v+neiId), -1 * ws));
							}else{
								Vec3b curPix = input[v].at<Vec3b>(y+1,x);
								for(auto c=0; c<chn; ++c)
									rightV[c] += (double)curPix[c] * ws;
							}
							leftV += ws;
						}
						if(leftV > 0){
							for(auto c=0; c<chn; ++c){
								triplets[c].push_back(Triplet((double)kConstraint, (double)(segSize*v+pid), leftV));
								rhs[c].push_back(rightV[c]);
							}
							kConstraint++;
						}
					}

					//t direction
					for(auto v=1; v<input.size()-1; ++v){
						for(auto c=0; c<chn; ++c){
							triplets[c].push_back(Triplet((double)kConstraint, (double)(segSize*v+pid), 2.0 * wt));
							triplets[c].push_back(Triplet((double)kConstraint, (double)(segSize*(v-1)+pid), -1.0 * wt));
							triplets[c].push_back(Triplet((double)kConstraint, (double)(segSize*(v+1)+pid), -1.0 * wt));
							rhs[c].push_back(0.0);
						}
						kConstraint++;
					}
				}

				printf("Solving...\n");
				printf("kConstraint:%d, kVar: %d\n", kConstraint, kVar);
				//solve
				for(auto c=0; c<chn; ++c){
					printf("Channel %d\n", c);
					Eigen::Map<VectorXd> b(rhs[c].data(), kConstraint);
					printf("c1\n");
					Eigen::SparseMatrix<double> A(kConstraint, kVar);
					printf("c2\n");
					A.setFromTriplets(triplets[c].begin(), triplets[c].end());
					printf("A.size():%d,%d\n", A.rows(), A.cols());
					//Eigen::SimplicialCholesky<SparseMatrix<double> > solver(A);
					//Eigen::SparseQR<SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver(A);
					Eigen::SPQR<SparseMatrix<double> > solver(A);

					printf("c4\n");
					VectorXd solution = solver.solve(b);
					printf("c5\n");
					//write back pixels
					printf("Write result\n");
					for(auto pid=0; pid<segment.size(); ++pid){
						const int x = (int)segment[pid][0];
						const int y = (int)segment[pid][1];
						for(auto v=0; v<output.size(); ++v){
							output[v].at<Vec3b>(y,x)[c] = solution[(int)segment.size() * v + pid];
						}
					}
				}
				cout << "done" << endl << flush;
			}
		};

		const int num_thread = 1;
		vector<thread_guard> threads(num_thread);
		for(auto tid=0; tid<num_thread; ++tid){
			std::thread t(threadFunc, tid, num_thread);
			threads[tid].bind(t);
		}
		for(auto& t: threads)
			t.join();
	}

}//namespace dynamic_stereo