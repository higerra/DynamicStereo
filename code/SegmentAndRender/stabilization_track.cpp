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
        constexpr double max_motion = 2;

        if(pos.size() < minInterval)
            return false;
        vector<double> ax(pos.size()), ay(pos.size());
        interval[0] = -1;
        interval[1] = -1;
        for (auto v1 = 0; v1 < pos.size() - minInterval; ++v1) {
            for (auto v2 = v1 + minInterval; v2 < pos.size(); ++v2) {
                int index = 0;
                double meanx = 0.0, meany = 0.0;
                for(auto i=v1; i<=v2; ++i, ++index){
                    ax[index] = pos[i].x;
                    ay[index] = pos[i].y;
                    meanx += ax[index];
                    meany += ay[index];
                }
                meanx /= static_cast<double>(v2-v1+1);
                meany /= static_cast<double>(v2-v1+1);

                double vx = 0.0, vy = 0.0;
                for(auto i=0; i<index; ++i){
                    vx += (ax[i] - meanx) * (ax[i] - meanx);
                    vy += (ay[i] - meany) * (ay[i] - meany);
                }
                vx /= static_cast<double>(index - 1);
                vy /= static_cast<double>(index - 1);

                double curVar = std::sqrt(vx*vx + vy*vy);

                double cur_max_motion = 0.0;
                for(auto i=1; i<index; ++i){
                    Vector2d motion_vector = Vector2d(ax[i], ay[i]) - Vector2d(ax[i-1], ay[i-1]);
                    double motion = motion_vector.norm();
                    cur_max_motion = std::max(cur_max_motion, motion);
                }
                if(curVar <= threshold && cur_max_motion <= max_motion){
                    if(v2 - v1  > interval[1] - interval[0]){
                        interval[0] = v1;
                        interval[1] = v2;
                    }

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
        const double blockW = 10, blockH = 10;
        const int gridW = width / blockW;
        const int gridH = height / blockH;

        auto cvPtToVec = [](const cv::Point2f& pt){
            return Vector2d(pt.x, pt.y);
        };

        output.resize(input.size());
        output[0] = input[0].clone();

        constexpr int cross_fade_r = 3;
        for(auto v = 1; v < input.size(); ++v){
            LOG(INFO) << "From frame " << v << " to frame " << v-1;

            vector<Eigen::Vector2d> src, tgt;
            vector<double> pW;
            for(auto tid=0; tid < kTrack; ++tid){
                if(optIntervals[tid][0] <= v -1 && optIntervals[tid][1] >= v){
                    src.push_back(cvPtToVec(trackMat.tracks[tid][v-offset[tid]]));
                    tgt.push_back(cvPtToVec(trackMat.tracks[tid][v-1-offset[tid]]));
                    pW.push_back(1.0);
                    const int dis_start = v - offset[tid];
                    const int dis_end = offset[tid] + trackMat.tracks[tid].size() - v;

                    if(dis_start <= cross_fade_r) {
                        pW.back() = static_cast<double>(dis_start) * 1.0 / static_cast<double>(cross_fade_r);
                    }
                    if(dis_end <= cross_fade_r){
                        pW.back() = static_cast<double>(dis_end) * 1.0 / static_cast<double>(cross_fade_r);
                    }
                }
            }

            if(src.empty()) {
                output[v] = input[v].clone();
                continue;
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
            gridWarping.computeWarpingField(src, tgt, 0.01, &pW);
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
                    trackMat.tracks[tid][v-offset[tid]].x = newLoc[0];
                    trackMat.tracks[tid][v-offset[tid]].y = newLoc[1];
                }
            }

            bool debugMode = true;
            if(debugMode){
//                //inspect point warping
                Vector2d testPt(127, 154);

                //debug visualize result
                int index = 0;
                bool reddot = true;
                bool gridline = false;
                while(true){
                    if(index == 0)
                        LOG(INFO) << v;
                    else
                        LOG(INFO) << v - 1;

                    Mat inputVis;
                    if(index % 2 == 0)
                        inputVis = input[v].clone();
                    else
                        inputVis = output[v-1].clone();

                    Mat outputVis = output[v-index%2].clone();

                    if(gridline){
                        gridWarping.visualizeGrid(gridWarping.getBaseGrid(), inputVis);
                        if(index == 0){
                            gridWarping.visualizeGrid(gridWarping.getWarpedGrid(), outputVis);
                        }else{
                            gridWarping.visualizeGrid(gridWarping.getBaseGrid(), outputVis);
                        }
                    }

                    const int scale = 3;
                    Mat largeInput, largeOutput;
                    cv::resize(inputVis, largeInput, cv::Size(width * scale, height * scale));
                    cv::resize(outputVis, largeOutput, cv::Size(width * scale, height * scale));

                    Vector2d warpedTestPt = gridWarping.warpPoint(testPt / (double)scale);
                    if(reddot) {
                        for (auto tid = 0; tid < src.size(); ++tid) {
                            if (index == 0) {
                                cv::circle(largeInput, cv::Point2f(src[tid][0] * scale, src[tid][1] * scale), 1,
                                           Scalar(0, 0, 255), 2);
                            } else {
                                cv::circle(largeInput, cv::Point2f(tgt[tid][0] * scale, tgt[tid][1] * scale), 1,
                                           Scalar(0, 0, 255), 2);
                            }
                        }

                        for (auto tid = 0; tid < src.size(); ++tid) {
                            if (index == 0) {
                                Vector2d newLoc = gridWarping.warpPoint(src[tid]);
                                cv::circle(largeOutput, cv::Point2f(newLoc[0] * scale, newLoc[1] * scale), 1,
                                           Scalar(0, 0, 255), 2);
                                cv::circle(largeOutput, cv::Point2f(warpedTestPt[0] * scale, warpedTestPt[1] * scale),
                                           1, Scalar(255, 0, 0), 2);
                            } else {
                                cv::circle(largeOutput, cv::Point2f(tgt[tid][0] * scale, tgt[tid][1] * scale), 1,
                                           Scalar(0, 0, 255), 2);
                                cv::circle(largeOutput, cv::Point2f(testPt[0], testPt[1]), 1, Scalar(255, 0, 0), 2);
                            }
                        }
                    }

                    Mat cont;
                    hconcat(largeInput, largeOutput, cont);
                    imshow("compare", cont);

                    index = (index + 1) % 2;
                    char key = (char)waitKey(500);
                    if(key == 'q') {
                        break;
                    }else if(key == 's'){
                        LOG(INFO) << "stab_input.png written";
                        imwrite("stab_input.png", largeInput);
                        LOG(INFO) << "stab_output.png written";
                        imwrite("stab_output.png", largeOutput);
                    }
                    else if(key == 'r')
                        reddot = !reddot;
                    else if(key == 'g')
                        gridline = !gridline;

                }
            }
        }
    }
}

