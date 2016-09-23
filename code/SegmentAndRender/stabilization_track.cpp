//
// Created by yanhang on 9/19/16.
//

#include "stabilization.h"
#include "../SubspaceStab/tracking.h"
#include "../SubspaceStab/warping.h"
#include "../base/utility.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo {

    bool findOptimalInterval(const std::vector<cv::Point2f>& pos, Vector2i& interval,
                             const double threshold, const int minInterval = 5) {
        double score = 0.0;
        if(pos.size() < minInterval)
            return false;
        //use l2 norm of variance
        double minVar = std::numeric_limits<double>::max();
        //integral to speed up
        vector<double> sumX(pos.size()), sumY(pos.size());
        sumX[0] = pos[0].x;
        sumY[0] = pos[0].y;
        for(auto v=1; v<pos.size(); ++v){
            sumX[v] = sumX[v-1] + pos[v].x;
            sumY[v] = sumY[v-1] + pos[v].y;
        }

        vector<double> ax(pos.size()), ay(pos.size());
        interval[0] = std::numeric_limits<int>::max();
        interval[1] = -1;
        for (auto v1 = 0; v1 < pos.size() - minInterval; ++v1) {
            for (auto v2 = v1 + minInterval; v2 < pos.size(); ++v2) {
                int index = 0;
                for(auto i=v1; i<=v2; ++i, ++index){
                    ax[index] = pos[i].x;
                    ay[index] = pos[i].y;
                }
                double meanx = (sumX[v2] - sumX[v1] + ax[0]) / (double)(v2-v1+1);
                double meany = (sumY[v2] - sumY[v1] + ay[0]) / (double)(v2-v1+1);
                double vx = math_util::variance(vector<double>(ax.begin(), ax.begin()+index), meanx);
                double vy = math_util::variance(vector<double>(ay.begin(), ay.begin()+index), meany);
                double curVar = std::sqrt(vx*vx + vy*vy);
                if(curVar <= threshold){
                    interval[0] = std::min(interval[0], v1);
                    interval[1] = std::max(interval[1], v2);
                }
            }
        }
        if(interval[0] >= 0 && interval[1] < pos.size())
            return true;
        return false;
    }

    void trackStabilization(const std::vector<cv::Mat> &input, std::vector<cv::Mat> &output,
                            const double threshold, const int tWindow) {
        CHECK(!input.empty());
        const int width = input[0].cols;
        const int height = input[0].rows;

        substab::FeatureTracks trackMat;
        LOG(INFO) << "Generating track matrix";
        substab::Tracking::genTrackMatrix(input, trackMat, tWindow, -1);

        const vector<size_t>& offset = trackMat.offset;
        const int kTrack = (int)offset.size();

        vector<Vector2i> optIntervals(offset.size(), Vector2i(0,0));
        for(auto tid=0; tid<kTrack; ++tid){
            Vector2i interval(0,0);
            bool is_inliner = findOptimalInterval(trackMat.tracks[tid], interval, threshold, tWindow);
            if(is_inliner){
                interval += Vector2i(offset[tid], offset[tid]);
                optIntervals[tid] = interval;
            }
        }

        //warp frame by frame
        const int gridW = 32;
        const int gridH = 32;

        auto cvPtToVec = [](const cv::Point2f& pt){
            return Vector2d(pt.x, pt.y);
        };

        output.resize(input.size());
        output[0] = input[0].clone();

        for(auto v = 1; v < input.size(); ++v){
            LOG(INFO) << "From frame " << v << " to frame " << v-1;

            vector<Eigen::Vector2d> src, tgt;
            for(auto tid=0; tid < kTrack; ++tid){
                if(optIntervals[tid][0] <= v -1 && optIntervals[tid][1] >= v){
                    src.push_back(cvPtToVec(trackMat.tracks[tid][v-offset[tid]]));
                    tgt.push_back(cvPtToVec(trackMat.tracks[tid][v-1-offset[tid]]));
                }
            }

            //visualize track after filtering
            substab::GridWarpping gridWarping(input[v].cols, input[v].rows, gridW, gridH);

            //Since we have to update feature locations after warping, we do a forward warp here.
            //This may cause blury with gaussian splatting...
            double residual1 = 0.0;
            for(auto i=0; i<src.size(); ++i){
                residual1 += (tgt[i] - src[i]).norm();
            }
            LOG(INFO) << "Residual before optimization: " << residual1;
            gridWarping.computeWarpingField(src, tgt, 0.1);
            //gridWarping.computeWarpingField(tgt, src, 0.1);

            double residual2 = 0;
            for(auto i=0; i<src.size(); ++i){
                Vector2d newLoc = gridWarping.warpPoint(src[i]);
                residual2 += (newLoc - tgt[i]).norm();
            }
            LOG(INFO) << "Residual after optimization: " << residual2;
            LOG(INFO) << "Residual change: " << residual1 - residual2;

            gridWarping.warpImageForward(input[v], output[v], 0.0);
            //gridWarping.warpImageBackward(input[v], output[v]);
            //update the feature location in frame v
            for(auto tid=0; tid < kTrack; ++tid){
                if(offset[tid] <= v && trackMat.tracks[tid].size() + offset[tid] > v){
                    Vector2d newLoc = gridWarping.warpPoint(cvPtToVec(trackMat.tracks[tid][v-offset[tid]]));
                    //if newLoc falls outside image, the remaining part of this track is abandoned
//                    if(newLoc[0] < 0 || newLoc[1] < 0 || newLoc[0] >= width - 1 || newLoc[1] >= height - 1){
//                        optIntervals[tid][1] = -1;
//                        trackMat.offset[tid] = input.size() + 1;
//                    }else{
//                        trackMat.tracks[tid][v-offset[tid]].x = newLoc[0];
//                        trackMat.tracks[tid][v-offset[tid]].y = newLoc[1];
//                    }

                    trackMat.tracks[tid][v-offset[tid]].x = newLoc[0];
                    trackMat.tracks[tid][v-offset[tid]].y = newLoc[1];
                }
            }

            {
                //debug visualize result
                int index = 0;
                bool inputmode = true;
                bool reddot = true;
                while(true){
                    if(index == 0)
                        LOG(INFO) << v;
                    else
                        LOG(INFO) << v - 1;

                    Mat inputVis = input[v-index%2].clone();
                    Mat outputVis = output[v-index%2].clone();


                    const int scale = 3;
                    Mat largeInput, largeOutput;
                    cv::resize(inputVis, largeInput, cv::Size(width * scale, height * scale));
                    cv::resize(outputVis, largeOutput, cv::Size(width * scale, height * scale));

                    if(reddot) {
                        for (auto tid = 0; tid < src.size(); ++tid) {
                            if (index == 0)
                                cv::circle(largeInput, cv::Point2f(src[tid][0] * scale, src[tid][1] * scale), 1, cv::Scalar(0, 0, 255),
                                           2);
                            else
                                cv::circle(largeInput, cv::Point2f(tgt[tid][0] * scale, tgt[tid][1] * scale), 1, cv::Scalar(0, 0, 255),
                                           2);
                        }

                        for (auto tid = 0; tid < src.size(); ++tid) {
                            if (index == 0) {
                                Vector2d newLoc = gridWarping.warpPoint(src[tid]);
                                cv::circle(largeOutput, cv::Point2f(newLoc[0] * scale, newLoc[1] * scale), 1, Scalar(0, 0, 255), 2);
                            } else
                                cv::circle(largeOutput, cv::Point2f(tgt[tid][0] * scale, tgt[tid][1] * scale), 1, Scalar(0, 0, 255), 2);
                        }
                    }

                    Mat cont;
                    hconcat(largeInput, largeOutput, cont);
                    imshow("compare", cont);

                    index = (index + 1) % 2;
                    char key = (char)waitKey(100);
                    if(key == 'q') {
                        break;
                    }else if(key == 's'){
                        imwrite("stab_snap.png", largeInput);
                    }
                    else if(key == 'r')
                        reddot = !reddot;

                }
            }
        }
    }
}

