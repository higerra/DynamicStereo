//
// Created by yanhang on 4/29/16.
//

#include "dynamicregularizer.h"
#include "../base/utility.h"
#include "../base/thread_guard.h"
#include "../base/depth.h"
#include "RPCA.h"
#include <Eigen/Sparse>
#include <Eigen/SPQRSupport>
#include "../GeometryModule/dynamicwarpping.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{

    void getSegmentRange(const vector<Mat>& visMaps,
                         const std::vector<std::vector<Eigen::Vector2i> >& segments,
                         std::vector<Eigen::Vector2i>& ranges){
        ranges.resize(segments.size(), Vector2i(0, visMaps.size() - 1));

#pragma omp parallel for
        for(auto sid=0; sid<segments.size(); ++sid){
            const vector<Vector2i>& seg = segments[sid];
            vector<int> invalidCount(visMaps.size(), 0);
            const int invalidMargin = seg.size() / 50;

            for(auto v=0; v < visMaps.size(); ++v){
                for(const auto& pix: seg){
                    if(visMaps[v].at<uchar>(pix[1], pix[0]) != Visibility::VISIBLE)
                        invalidCount[v]++;
                }
            }

            for(int v=visMaps.size() / 2-1; v>=0; --v){
                if(invalidCount[v] > invalidMargin) {
                    ranges[sid][0] = std::max(ranges[sid][0], v);
                    break;
                }
            }
            for(int v=visMaps.size() / 2; v < visMaps.size(); ++v){
                if(invalidCount[v] > invalidMargin) {
                    ranges[sid][1] = std::min(ranges[sid][1], v);
                    break;
                }
            }
        }
    }

    void filterShortSegments(std::vector<std::vector<Eigen::Vector2i> >& segments,
                             std::vector<Eigen::Vector2i>& ranges,
                             const int minFrame){
        vector<vector<Vector2i> > segmentsFiltered;
        vector<Vector2i> rangesFiltered;
        for(auto sid=0; sid < segments.size(); ++sid){
            if(ranges[sid][1] - ranges[sid][0] + 1 >= minFrame) {
                segmentsFiltered.push_back(segments[sid]);
                rangesFiltered.push_back(ranges[sid]);
            }
        }
        segments.swap(segmentsFiltered);
        ranges.swap(rangesFiltered);
    }

    void renderToMask(const std::vector<cv::Mat>& input, const std::vector<std::vector<Eigen::Vector2i> >& segments,
                      const std::vector<Eigen::Vector2i>& ranges, std::vector<cv::Mat>& output){
        CHECK(!input.empty());
        output.resize(input.size());
        for(auto v=0; v<output.size(); ++v)
            output[v] = input[input.size() / 2].clone();

        for(auto sid=0; sid < segments.size(); ++sid){
            const vector<Vector2i>& seg = segments[sid];
            for(auto v=0; v<output.size(); ++v){
                int input_fid = ranges[sid][0] + v % (ranges[sid][1] - ranges[sid][0] + 1);
                for(const auto& pix: seg) {
                    output[v].at<Vec3b>(pix[1], pix[0]) = input[input_fid].at<Vec3b>(pix[1], pix[0]);
                }
            }
        }
    }

    void regularizationAnisotropic(const std::vector<cv::Mat>& input,
                               const std::vector<std::vector<Eigen::Vector2i> >& segments,
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
                    const int x = segments[sid][i][0];
                    const int y = segments[sid][i][1];
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
                               const vector<vector<Eigen::Vector2i> >& segments,
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
            for(auto sid=tid; sid<segments.size(); sid+=num_thread) {
                printf("Optimizing segment %d(%d) on thread %d\n", sid, (int) segments.size(), tid);
                const vector<Vector2i> &segment = segments[sid];
                const int segSize = (int) segment.size();

                Mat idMap(height, width, CV_32SC1, Scalar::all(-1));
                for (auto pid = 0; pid < segment.size(); ++pid)
                    idMap.at<int>(segment[pid][1], segment[pid][0]) = pid;

                const int kVar = segSize * (int) input.size();
                vector<Triplet>triplets;
                vector<VectorXd> rhs((size_t) chn, VectorXd(kVar));
                triplets.reserve((size_t) (kVar * 8));

                for (auto c = 0; c < chn; ++c) {
                    for(auto i=0; i<kVar; ++i)
                        rhs[c][i] = 0.0;
                }

                auto addSpatialSmoothTerm = [&](const int fid, const int varId, const cv::Point &neiPt,
                                                double &leftV) {
                    int neiId = idMap.at<int>(neiPt);
                    if (neiId >= 0) {
                        neiId += segSize * fid;
                        triplets.push_back(Triplet(varId, neiId, -1.0 * ws));
                    } else {
                        Vec3b neiPix = input[fid].at<Vec3b>(neiPt);
                        for (auto c = 0; c < chn; ++c)
                            rhs[c][varId] += (double) neiPix[c] * ws;
                    }
                    leftV += ws;
                };

                for(auto pid=0; pid<segment.size(); ++pid){
                    const int x = (int)segment[pid][0];
                    const int y = (int)segment[pid][1];
                    for(auto v=0; v<input.size(); ++v){
                        Vec3b curpix = input[v].at<Vec3b>(y,x);
                        const int varId = segSize * v + pid;
                        //data constraint
                        double leftV = 0.0;
                        if (curpix != invalidToken) {
                            leftV += 1.0;
                            for (auto c = 0; c < chn; ++c) {
                                rhs[c][v * segSize + pid] += (double) curpix[c];
                            }
                        }
                        //smooth constraint
                        //spatial
                        if (x > 0) {
                            addSpatialSmoothTerm(v, varId, cv::Point(x-1,y), leftV);
                        }
                        if (x < width - 1) {
                            addSpatialSmoothTerm(v, varId, cv::Point(x+1,y), leftV);
                        }
                        if(y > 0){
                            addSpatialSmoothTerm(v, varId, cv::Point(x,y-1), leftV);
                        }
                        if(y < height - 1){
                            addSpatialSmoothTerm(v, varId, cv::Point(x,y+1), leftV);
                        }
                        //temporal
                        if(v > 0){
                            leftV += wt;
                            triplets.push_back(Triplet(varId, varId-segSize, -1*wt));
                        }
                        if(v < input.size() - 1){
                            leftV += wt;
                            triplets.push_back(Triplet(varId, varId+segSize, -1*wt));
                        }

                        if(leftV > 0)
                            triplets.push_back(Triplet(varId, varId, leftV));
                    }
                }
                //solve
                Eigen::SparseMatrix<double> A(kVar, kVar);
                printf("Constructing matrix...\n");
                A.setFromTriplets(triplets.begin(), triplets.end());
                printf("Decomposing...\n");
                //Eigen::SimplicialLDLT<SparseMatrix<double> > solver(A);
                //Eigen::SparseLU<SparseMatrix<double> > solver(A);
                //Eigen::SPQR<SparseMatrix<double> > solver(A);
                Eigen::ConjugateGradient<SparseMatrix<double> > solver(A);
                solver.setMaxIterations(20000);
                printf("Done\n");
                for(auto c=0; c<chn; ++c){
                    printf("Channel %d\n", c);
                    VectorXd solution = solver.solve(rhs[c]);
//					write back pixels
                    for(auto pid=0; pid<segment.size(); ++pid){
                        const int x = (int)segment[pid][0];
                        const int y = (int)segment[pid][1];
                        for(auto v=0; v<output.size(); ++v){
                            output[v].at<Vec3b>(y,x)[c] = solution[segSize * v + pid];
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

    void regularizationRPCA(const std::vector<cv::Mat> &input,
                            const std::vector<std::vector<Eigen::Vector2i> > &segments,
                            std::vector<cv::Mat> &output, double lambda){
        CHECK(!input.empty());
        const int width = input[0].cols;
        const int height = input[0].rows;

        output.resize(input.size());
        for(auto v=0; v<input.size(); ++v)
            output[v] = input[v].clone();
        if(lambda < 0)
            lambda = std::sqrt((double)input.size());

        const int numThread = 6;
        auto threadFunc = [&](int tid, int nt){
            for(auto sid=tid; sid < segments.size(); sid += nt){
                printf("Running RPCA for segment %d on thread %d\n", sid, tid);
                for(auto c=0; c<3; ++c){
                    MatrixXd pixelMat((int)input.size(), (int)segments[sid].size());
                    for(auto v=0; v<input.size(); ++v){
                        int index = 0;
                        for(const auto& pt: segments[sid]){
                            pixelMat(v, index) = (double)input[v].at<Vec3b>((int)pt[1], (int)pt[0])[c];
                            index++;
                        }
                    }

                    MatrixXd res, error;
                    int numIter;
                    RPCAOption option;
                    option.lambda = lambda;
                    solveRPCA(pixelMat, res, error, numIter, option);

                    for(auto v=0; v<output.size(); ++v){
                        int index = 0;
                        for(const auto& pt: segments[sid]){
                            double p = std::max(std::min(res(v, index), 255.0), 0.0);
                            output[v].at<Vec3b>(pt[1], pt[0])[c] = (uchar)p;
                            index++;
                        }
                    }
                }
            }
        };

//		for(auto tid=0; tid < segments.size(); ++tid){
//			threadFunc(tid, 1);
//		}

//		float start_t = (float)cv::getTickCount();
//		threadFunc(0,1);
//		printf("Time usage: %.2fs\n", ((float)getTickCount() - start_t) / (float)getTickFrequency());

        vector<thread_guard> threads((size_t) numThread);
        for(int tid=0; tid < threads.size(); ++tid){
            std::thread t(threadFunc, tid, numThread);
            threads[tid].bind(t);
        }

        for(auto& t: threads)
            t.join();

    }


    void temporalMedianFilter(const std::vector<cv::Mat>& input,
                              const std::vector<std::vector<Eigen::Vector2i> >& segments,
                              std::vector<cv::Mat>& output, const int r){
        CHECK(!input.empty());
        const int h = input[0].rows;
        const int w = input[0].cols;
        output.resize(input.size());
        for(auto i=0; i<input.size(); ++i)
            output[i] = input[i].clone();

        vector<Vector2i> bound(input.size());
        for(auto i=0; i<input.size(); ++i){
            if (i - r < 0) {
                bound[i][0] = 0;
                bound[i][1] = 2 * r + 1;
            }
            else if (i + r >= input.size()) {
                bound[i][0] = (int)input.size() - 2 * r - 2;
                bound[i][1] = (int)input.size() - 1;
            } else {
                bound[i][0] = i - r;
                bound[i][1] = i + r;
            }
        }

#pragma omp parallel for
        for(auto sid=0; sid < segments.size(); ++sid){
            const vector<Vector2i>& segment = segments[sid];
            vector<uchar> rc(input.size()),gc(input.size()),bc(input.size());
            for(const auto& pt: segment){
                for(auto i=0; i<input.size(); ++i) {
                    int index = 0;
                    for(auto t = bound[i][0]; t <= bound[i][1]; ++t){
                        Vec3b pix = input[t].at<Vec3b>(pt[1],pt[0]);
                        if(pix == Vec3b(0,0,0))
                            continue;
                        rc[index] = pix[0];
                        gc[index] = pix[1];
                        bc[index] = pix[2];
                        index++;
                    }
                    if(rc.size() < 3){
                        output[i].at<Vec3b>(pt[1],pt[0]) = Vec3b(0,0,0);
                    }else {
                        nth_element(rc.begin(), rc.begin() + r, rc.begin() + index);
                        nth_element(gc.begin(), gc.begin() + r, gc.begin() + index);
                        nth_element(bc.begin(), bc.begin() + r, bc.begin() + index);
                        output[i].at<Vec3b>(pt[1], pt[0]) = Vec3b(rc[r], gc[r], bc[r]);
                    }
                }
            }
        }
    }

    void regularizationFlashy(const std::vector<cv::Mat> &input,
                              const std::vector<std::vector<Eigen::Vector2d> > &segments,
                              std::vector<cv::Mat> &output){
        CHECK(!input.empty());
        CHECK_EQ(input.size(), output.size());

        const int numThreads = 6;
        auto threadFunc = [&](const int tid, const int nt){

        };

        vector<thread_guard> threads((size_t) numThreads);
        for(int tid=0; tid < numThreads; ++tid){
            std::thread t(threadFunc, tid, numThreads);
            threads[tid].bind(t);
        }

        for(auto& t: threads)
            t.join();
    }

    void dumpOutSegment(const std::vector<cv::Mat>& images,
                        const std::vector<Eigen::Vector2d>& segment,
                        const std::string& path){
        ofstream fout(path.c_str());
        CHECK(fout.is_open()) << "dumpOutSegment: can not open file to write: " << path;
//		int maxX = -1, minX = numeric_limits<int>::max(), maxY = -1, minY = numeric_limits<int>::max();
//		for(const auto& pt: segment){
//			minX = std::min(minX, (int)pt[0]);
//			maxX = std::max(maxX, (int)pt[0]);
//			minY = std::min(minY, (int)pt[1]);
//			maxY = std::max(maxX, (int)pt[1]);
//		}
//
//		fout << (int)images.size() << ' ' << (int)segment.size() << ' ' << minX << ' ' << maxX << ' ' << minY << ' ' << maxY << endl;
//		for(const auto& pt: segment)
//			fout << (int)pt[0] << ' ' << (int)pt[1] << endl;

        for (auto v = 0; v < images.size(); ++v) {
            for(const auto& pt: segment){
                Vec3b pix = images[v].at<Vec3b>((int)pt[1], (int)pt[0]);
                fout << (int)pix[0] << ' ' << (int)pix[1] << ' ' << (int)pix[2] << ' ';
            }
            fout << endl;
        }

        fout.close();
    }


}//namespace dynamic_stereo