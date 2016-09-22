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

    double findOptimalInterval(const std::vector<cv::Point2f>& pos, Vector2i& interval, const int minInterval = 5) {
        double score = 0.0;
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
        for (auto v1 = 0; v1 < pos.size() - minInterval; ++v1) {
            for (auto v2 = v1 + minInterval; v2 < pos.size(); ++v2) {
                int index = 0;
                for(auto i=v1; i<=v2; ++i, ++index){
                    ax[index] = pos[i].x;
                    ay[index] = pos[i].y;
                }
                double meanx = (sumX[v2] - sumX[v1] + ax[0]) / (double)(v2-v1+1);
                double meany = (sumX[v2] - sumY[v1] + ay[0]) / (double)(v2-v1+1);
                double vx = math_util::variance(vector<double>(ax.begin(), ax.begin()+index), meanx);
                double vy = math_util::variance(vector<double>(ay.begin(), ay.begin()+index), meany);
                double curVar = std::sqrt(vx*vx + vy*vy);
                if(curVar < minVar){
                    minVar = curVar;
                    interval[0] = v1;
                    interval[1] = v2;
                }
            }
        }

        return score;
    }

    void trackStabilization(const std::vector<cv::Mat> &input, std::vector<cv::Mat> &output,
                                 const double threshold, const int tWindow) {
        substab::FeatureTracks trackMat;
        substab::Tracking::genTrackMatrix(input, trackMat, tWindow, -1);

        const vector<size_t>& offset = trackMat.offset;
        const int kTrack = (int)offset.size();

        vector<Vector2i> optIntervals(offset.size(), Vector2i(0,0));
        for(auto tid=0; tid<kTrack; ++tid){
            Vector2i interval;
            double trackScore = findOptimalInterval(trackMat.tracks[tid], interval);
            if(trackScore >= threshold){
                interval += Vector2i(offset[tid], offset[tid]);
                optIntervals[tid] = interval;
            }
        }

        //warp frame by frame
        const int gridW = 16;
        const int gridH = 16;

        auto cvPtToVec = [](const cv::Point2f& pt){
            return Vector2d(pt.x, pt.y);
        };

        output.resize(input.size());
        for(auto v = 1; v < input.size(); ++v){
            vector<Eigen::Vector2d> src, tgt;
            for(auto tid=0; tid < kTrack; ++tid){
                if(optIntervals[tid][0] <= v -1 && optIntervals[tid][1] >= v){
                    src.push_back(cvPtToVec(trackMat.tracks[tid][v-offset[tid]]));
                    tgt.push_back(cvPtToVec(trackMat.tracks[tid][v-1-offset[tid]]));
                }
            }
            substab::GridWarpping gridWarping(input[v].cols, input[v].rows, gridW, gridH);
            //Since we have to update feature locations after warping, we do a forward warp here.
            //This may cause blury with gaussian splatting...
            gridWarping.computeWarpingField(src, tgt);
            gridWarping.warpImageForward(input[v], output[v]);
            //update the feature location in frame v
            for(auto tid=0; tid < kTrack; ++tid){
                if(offset[tid] <= v && trackMat.tracks[tid].size() + offset[tid] > v){
                    Vector2d newLoc = gridWarping.warpPoint(cvPtToVec(trackMat.tracks[tid][v-offset[tid]]));
                    trackMat.tracks[tid][v-offset[tid]].x = newLoc[0];
                    trackMat.tracks[tid][v-offset[tid]].x = newLoc[1];
                }
            }
        }
    }
}

