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

    bool findOptimalInterval(const std::vector<cv::Point2f> &pos, Vector2i &interval,
                             const double threshold, const int minInterval = 5) {
        constexpr double max_motion = 2;

        if (pos.size() < minInterval)
            return false;
        vector<double> ax(pos.size()), ay(pos.size());
        interval[0] = -1;
        interval[1] = -1;
        for (auto v1 = 0; v1 < pos.size() - minInterval; ++v1) {
            for (auto v2 = v1 + minInterval; v2 < pos.size(); ++v2) {
                int index = 0;
                double meanx = 0.0, meany = 0.0;
                for (auto i = v1; i <= v2; ++i, ++index) {
                    ax[index] = pos[i].x;
                    ay[index] = pos[i].y;
                    meanx += ax[index];
                    meany += ay[index];
                }
                meanx /= static_cast<double>(v2 - v1 + 1);
                meany /= static_cast<double>(v2 - v1 + 1);

                double vx = 0.0, vy = 0.0;
                for (auto i = 0; i < index; ++i) {
                    vx += (ax[i] - meanx) * (ax[i] - meanx);
                    vy += (ay[i] - meany) * (ay[i] - meany);
                }
                vx /= static_cast<double>(index - 1);
                vy /= static_cast<double>(index - 1);

                double curVar = std::sqrt(vx * vx + vy * vy);

                double cur_max_motion = 0.0;
                for (auto i = 1; i < index; ++i) {
                    Vector2d motion_vector = Vector2d(ax[i], ay[i]) - Vector2d(ax[i - 1], ay[i - 1]);
                    double motion = motion_vector.norm();
                    cur_max_motion = std::max(cur_max_motion, motion);
                }
                if (curVar <= threshold && cur_max_motion <= max_motion) {
                    if (v2 - v1 > interval[1] - interval[0]) {
                        interval[0] = v1;
                        interval[1] = v2;
                    }

                }
            }
        }
        if (interval[0] >= 0 && interval[1] < pos.size())
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

        const vector<size_t> &offset = trackMat.offset;
        const int kTrack = (int) offset.size();

        vector<Vector2i> optIntervals(offset.size(), Vector2i(0, 0));
        for (auto tid = 0; tid < kTrack; ++tid) {
            Vector2i interval(0, 0);
            bool is_inliner = findOptimalInterval(trackMat.tracks[tid], interval, threshold, tWindow);
            if (is_inliner) {
                interval += Vector2i(offset[tid], offset[tid]);
                optIntervals[tid] = interval;
            }
        }

        //warp frame by frame
        const double blockW = 10, blockH = 10;
        const int gridW = width / blockW;
        const int gridH = height / blockH;

        auto cvPtToVec = [](const cv::Point2f &pt) {
            return Vector2d(pt.x, pt.y);
        };

        output.resize(input.size());
        output[0] = input[0].clone();

        constexpr int cross_fade_r = 3;
        for (auto v = 1; v < input.size(); ++v) {
            LOG(INFO) << "From frame " << v << " to frame " << v - 1;

            vector<Eigen::Vector2d> src, tgt;
            vector<double> pW;
            for (auto tid = 0; tid < kTrack; ++tid) {
                if (optIntervals[tid][0] <= v - 1 && optIntervals[tid][1] >= v) {
                    src.push_back(cvPtToVec(trackMat.tracks[tid][v - offset[tid]]));
                    tgt.push_back(cvPtToVec(trackMat.tracks[tid][v - 1 - offset[tid]]));
                    pW.push_back(1.0);
                    const int dis_start = v - offset[tid];
                    const int dis_end = offset[tid] + trackMat.tracks[tid].size() - v;

                    if (dis_start <= cross_fade_r) {
                        pW.back() = static_cast<double>(dis_start) * 1.0 / static_cast<double>(cross_fade_r);
                    }
                    if (dis_end <= cross_fade_r) {
                        pW.back() = static_cast<double>(dis_end) * 1.0 / static_cast<double>(cross_fade_r);
                    }
                }
            }

            if (src.empty()) {
                output[v] = input[v].clone();
                continue;
            }

            //visualize track after filtering
            substab::GridWarpping gridWarping(input[v].cols, input[v].rows, gridW, gridH);

            //Since we have to update feature locations after warping, we do a forward warp here.
            //This may cause blury with gaussian splatting...
            double residual1 = 0.0;
            for (auto i = 0; i < src.size(); ++i) {
                residual1 += (tgt[i] - src[i]).norm();
            }
            LOG(INFO) << "Residual before optimization: " << residual1;
            gridWarping.computeWarpingField(src, tgt, 0.01, &pW);
            //gridWarping.computeWarpingField(tgt, src, 0.1);

            double residual2 = 0;
            for (auto i = 0; i < src.size(); ++i) {
                Vector2d newLoc = gridWarping.warpPoint(src[i]);
                residual2 += (newLoc - tgt[i]).norm();
            }
            LOG(INFO) << "Residual after optimization: " << residual2;
            LOG(INFO) << "Residual change: " << residual1 - residual2;

            gridWarping.warpImageForward(input[v], output[v], 0.0);
            //gridWarping.warpImageBackward(input[v], output[v]);

            //update the feature location in frame v
            for (auto tid = 0; tid < kTrack; ++tid) {
                if (offset[tid] <= v && trackMat.tracks[tid].size() + offset[tid] > v) {
                    Vector2d newLoc = gridWarping.warpPoint(cvPtToVec(trackMat.tracks[tid][v - offset[tid]]));
                    trackMat.tracks[tid][v - offset[tid]].x = newLoc[0];
                    trackMat.tracks[tid][v - offset[tid]].y = newLoc[1];
                }
            }

            bool debugMode = true;
            if (debugMode) {
//                //inspect point warping
                Vector2d testPt(127, 154);

                //debug visualize result
                int index = 0;
                bool reddot = true;
                bool gridline = false;
                while (true) {
                    if (index == 0)
                        LOG(INFO) << v;
                    else
                        LOG(INFO) << v - 1;

                    Mat inputVis;
                    if (index % 2 == 0)
                        inputVis = input[v].clone();
                    else
                        inputVis = output[v - 1].clone();

                    Mat outputVis = output[v - index % 2].clone();

                    if (gridline) {
                        gridWarping.visualizeGrid(gridWarping.getBaseGrid(), inputVis);
                        if (index == 0) {
                            gridWarping.visualizeGrid(gridWarping.getWarpedGrid(), outputVis);
                        } else {
                            gridWarping.visualizeGrid(gridWarping.getBaseGrid(), outputVis);
                        }
                    }

                    const int scale = 3;
                    Mat largeInput, largeOutput;
                    cv::resize(inputVis, largeInput, cv::Size(width * scale, height * scale));
                    cv::resize(outputVis, largeOutput, cv::Size(width * scale, height * scale));

                    Vector2d warpedTestPt = gridWarping.warpPoint(testPt / (double) scale);
                    if (reddot) {
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
                    char key = (char) waitKey(500);
                    if (key == 'q') {
                        break;
                    } else if (key == 's') {
                        LOG(INFO) << "stab_input.png written";
                        imwrite("stab_input.png", largeInput);
                        LOG(INFO) << "stab_output.png written";
                        imwrite("stab_output.png", largeOutput);
                    } else if (key == 'r')
                        reddot = !reddot;
                    else if (key == 'g')
                        gridline = !gridline;

                }
            }
        }
    }

    static void TrackFeatureDoubleDirection(const std::vector<cv::Mat> &grays,
                                            const std::vector<std::vector<cv::Mat> > &pyramid, const int anchor,
                                            const std::vector<Eigen::Vector2d> &existing_tracks,
                                            const substab::TrackingOption &tracking_option,
                                            std::vector<std::vector<Eigen::Vector2d> > &new_tracks) {
        CHECK_LT(anchor, pyramid.size());
        vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(grays[anchor], corners, tracking_option.max_num_corners, tracking_option.quality_level,
                                tracking_option.min_distance);

        vector<cv::Point2f> new_corners;
        for (const auto &c: corners) {
            bool is_new = true;
            for (const auto &ec: existing_tracks) {
                cv::Point2f ecv(ec[0], ec[1]);
                double dis = cv::norm(ecv - c);
                if (dis <= tracking_option.max_diff_distance) {
                    is_new = false;
                    break;
                }
            }
            if (is_new)
                new_corners.push_back(c);
        }

        if(new_corners.empty())
            return;

        new_tracks.resize(new_corners.size());
        for (auto tid = 0; tid < new_tracks.size(); ++tid) {
            new_tracks[tid].resize(grays.size());
            new_tracks[tid][anchor] = Vector2d(new_corners[tid].x, new_corners[tid].y);
        }

        //track forward
        vector<cv::Point2f> previous_corners = new_corners;
        vector<bool> is_valid(new_corners.size(), true);
        for (auto v = anchor + 1; v < grays.size(); ++v) {
            vector<cv::Point2f> next_corners;
            vector<uchar> track_status;
            Mat track_error;
            cv::calcOpticalFlowPyrLK(pyramid[v - 1], pyramid[v], previous_corners, next_corners, track_status,
                                     track_error, tracking_option.win_size);
            for (auto tid = 0; tid < track_status.size(); ++tid) {
                if (is_valid[tid]) {
                    if (track_status[tid] == (uchar) 1) {
                        new_tracks[tid][v] = Vector2d(next_corners[tid].x, next_corners[tid].y);
                    } else {
                        is_valid[tid] = 0.0;
                    }
                }
            }
            previous_corners.swap(next_corners);
        }

        //track backward
        previous_corners = new_corners;
        for (auto tid = 0; tid < new_corners.size(); ++tid) {
            is_valid[tid] = true;
        }

        for (auto v = anchor - 1; v >= 0; --v) {
            vector<cv::Point2f> next_corners;
            vector<uchar> track_status;
            Mat track_error;
            cv::calcOpticalFlowPyrLK(pyramid[v + 1], pyramid[v], previous_corners, next_corners, track_status,
                                     track_error, tracking_option.win_size);
            for (auto tid = 0; tid < track_status.size(); ++tid) {
                if (is_valid[tid]) {
                    if (track_status[tid] == (uchar) 1) {
                        new_tracks[tid][v] = Vector2d(next_corners[tid].x, next_corners[tid].y);
                    } else {
                        is_valid[tid] = 0.0;
                    }
                }
            }
            previous_corners.swap(next_corners);
        }

        //keep tracks which appear in the first frame
        vector<vector<Eigen::Vector2d> > filtered_tracks;
        for (auto tid = 0; tid < new_tracks.size(); ++tid) {
            if (new_tracks[tid].front()[0] >= 0 && new_tracks[tid].front()[1] >= 0) {
                filtered_tracks.push_back(new_tracks[tid]);
            }
        }

        new_tracks.swap(filtered_tracks);
    }

    static bool IsTrackValid(std::vector<Vector2d> &track, const double max_variance,
                             const int min_length = 10) {
        if (track.size() < min_length)
            return false;

        constexpr double max_motion = 2;

        vector<Vector2d> sum_pos(track.size(), Vector2d(0, 0));
        sum_pos[0] = track[0];
        for (auto i = 1; i < track.size(); ++i) {
            sum_pos[i] = sum_pos[i - 1] + track[i];
        }

        int best_end_frame = -1;

        for (int end_frame = min_length; end_frame < track.size(); ++end_frame) {
            //variance constraint
            Vector2d mean_pos = sum_pos[end_frame] / (end_frame + 1);
            Vector2d variance_pos(0.0, 0.0);
            for (int v = 0; v <= end_frame; ++v) {
                variance_pos[0] += (track[v][0] - mean_pos[0]) * (track[v][0] - mean_pos[0]);
                variance_pos[1] += (track[v][1] - mean_pos[1]) * (track[v][1] - mean_pos[1]);
            }
            variance_pos /= static_cast<double>(end_frame);
            if (variance_pos.norm() > max_variance) {
                break;
            }

            //motion constraint
            double cur_max_motion = -1;
            for(int v=1; v <= end_frame; ++v){
                Vector2d cur_motion = track[v] - track[v-1];
                cur_max_motion = std::max(cur_motion.norm(), cur_max_motion);
            }
            if(cur_max_motion > max_motion){
                break;
            }

            best_end_frame = end_frame;
        }
        if (best_end_frame < 0) {
            for (auto v = 0; v < track.size(); ++v) {
                track[v] = Vector2d(-1, -1);
            }
            return false;
        } else {
            for (auto v = best_end_frame; v < track.size(); ++v) {
                track[v] = Vector2d(-1, -1);
            }
            return true;
        }
    }

    void trackStabilizationGlobal(const std::vector<cv::Mat> &input, std::vector<cv::Mat> &output,
                                 const double threshold, int tWindow, const bool use_homography) {
        CHECK(!input.empty());
        output.resize(input.size());
        for (auto v = 0; v < input.size(); ++v) {
            output[v] = input[v].clone();
        }

        if(input.size() < tWindow){
            return;
        }

        const int width = input[0].cols;
        const int height = input[0].rows;

        constexpr int kMinTrack = 300;

        constexpr bool debug_mode = false;

        //to obtain more tracks from the first frame, also compute features in the first ${kTrackFrame} frames, take
        //all features that can be tracked back to the first frame
        constexpr int kTrackFrame = 5;

        //track from the first frame
        substab::TrackingOption tracking_option;
        tracking_option.max_num_corners = 1000;

        vector<Mat> grays(input.size());
        vector<vector<Mat> > pyramid(input.size());

        for (auto v = 0; v < input.size(); ++v) {
            cvtColor(input[v], grays[v], CV_BGR2GRAY);
            cv::buildOpticalFlowPyramid(grays[v], pyramid[v], tracking_option.win_size, tracking_option.n_level);
        }

        int start_frame = 0;
        while (start_frame < (int)input.size() - tWindow) {
            output[start_frame] = input[start_frame].clone();
            //LOG(INFO) << "Stabilization restart at frame: " << start_frame << "/" << (int)input.size() - tWindow;
            vector<vector<Eigen::Vector2d> > all_tracks;
            //LOG(INFO) << "Generating tracks";
            for (auto v = 0; v < kTrackFrame; ++v) {
                if (v + start_frame == input.size()) {
                    break;
                }

                vector<vector<Eigen::Vector2d> > new_tracks;
                vector<Eigen::Vector2d> existing_tracks;

                if (!all_tracks.empty()) {
                    for (const auto &track: all_tracks) {
                        if (track[v][0] >= 0 && track[v][1] >= 0) {
                            existing_tracks.push_back(track[v]);
                        }
                    }
                }
                TrackFeatureDoubleDirection(vector<Mat>(grays.begin()+start_frame, grays.end()),
                                            vector<vector<Mat> >(pyramid.begin() + start_frame, pyramid.end()),
                                            v, existing_tracks, tracking_option, new_tracks);
                for (auto &nt: new_tracks) {
                    if (IsTrackValid(nt, threshold, tWindow)) {
                        all_tracks.push_back(nt);
                    }
                }
            }

            constexpr int blockW = 20, blockH = 20;
            const int gridW = width / blockW, gridH = height / blockH;
            constexpr double weight_similarity = 0.1;
            substab::GridWarpping grid_warping(input[0].cols, input[0].rows, gridW, gridH);

            //register all frames to the first frame
            int terminate_frame = start_frame + 1;
            for (auto v = start_frame + 1; v < input.size(); ++v, ++terminate_frame) {
                //source: frame v, target: frame 0
                vector<Vector2d> src_point, tgt_point;
                for (auto tid = 0; tid < all_tracks.size(); ++tid) {
                    if (all_tracks[tid][v - start_frame][0] >= 0 && all_tracks[tid][v - start_frame][1] >= 0) {
                        src_point.push_back(all_tracks[tid][v - start_frame]);
                        tgt_point.push_back(all_tracks[tid][0]);
                    }
                }
                //LOG(INFO) << "Number of pairs: " << src_point.size();
                if (src_point.size() < kMinTrack) {
                    break;
                }


                if(use_homography){
                    //compute homography warping
                    vector<cv::Point2f> cv_src_pt(src_point.size()), cv_tgt_pt(src_point.size());
                    for(auto i=0; i<src_point.size(); ++i){
                        cv_src_pt[i].x = src_point[i][0];
                        cv_src_pt[i].y = src_point[i][1];
                        cv_tgt_pt[i].x = tgt_point[i][0];
                        cv_tgt_pt[i].y = tgt_point[i][1];
                    }
                    Mat homography = cv::findHomography(cv_src_pt, cv_tgt_pt);
                    cv::warpPerspective(input[v], output[v], homography, input[v].size(), cv::INTER_CUBIC);

                }else{
                    //compute grid warping from frame 0 to frame v and apply backward warping
                    grid_warping.computeWarpingField(tgt_point, src_point, weight_similarity);
                    grid_warping.warpImageBackward(input[v], output[v]);
                }


                if(debug_mode){
                    //debug: visualize stabilization
                    LOG(INFO) << "Debug mode";
                    int index = 0;
                    bool reddot = true;
                    bool gridline = false;
                    while(true){
                        if(index == 0)
                            LOG(INFO) << 0;
                        else
                            LOG(INFO) << v;

                        Mat inputVis, outputVis;
                        if(index == 0){
                            inputVis = input[0].clone();
                            outputVis = input[0].clone();
                        }else{
                            inputVis = input[v].clone();
                            outputVis = output[v].clone();
                        }

                        if(gridline){
                            grid_warping.visualizeGrid(grid_warping.getBaseGrid(), inputVis);
                            if(index == 0){
                                grid_warping.visualizeGrid(grid_warping.getBaseGrid(), outputVis);
                            }else{
                                grid_warping.visualizeGrid(grid_warping.getWarpedGrid(), outputVis);
                            }
                        }

                        const int scale = 3;
                        Mat largeInput, largeOutput;
                        cv::resize(inputVis, largeInput, cv::Size(width * scale, height * scale));
                        cv::resize(outputVis, largeOutput, cv::Size(width * scale, height * scale));

                        if(reddot) {
                            for (auto tid = 0; tid < src_point.size(); ++tid) {
                                if (index == 0) {
                                    cv::circle(largeInput, cv::Point2f(tgt_point[tid][0] * scale, tgt_point[tid][1] * scale), 1,
                                               Scalar(0, 0, 255), 2);
                                } else {
                                    cv::circle(largeInput, cv::Point2f(src_point[tid][0] * scale, src_point[tid][1] * scale), 1,
                                               Scalar(0, 0, 255), 2);
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

            //warp following frames according to current warping field
            start_frame = terminate_frame + 1;
        }
    }
}

