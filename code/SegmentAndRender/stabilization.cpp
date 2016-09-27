//
// Created by yanhang on 9/13/16.
//

#include "stabilization.h"
#include "../SubspaceStab/subspacestab.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{

    void stabilizeSegments(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output,
                           const std::vector<std::vector<Eigen::Vector2i> >& segments,
                           const std::vector<Eigen::Vector2i>& ranges,
                           const double lambda, const StabAlg alg) {

        CHECK(!input.empty());
        output.resize(input.size());
        for (auto i = 0; i < output.size(); ++i)
            output[i] = input[i].clone();

        const int width = input[0].cols;
        const int height = input[0].rows;

        const int margin = 5;
        int index = 0;

        const int dseg = 5;
        for (const auto &segment: segments) {
            if (dseg >= 0 && index != dseg) {
                index++;
                continue;
            }
            printf("Segment %d/%d\n", index, (int) segments.size());
            cv::Point2i tl(width + 1, height + 1);
            cv::Point2i br(-1, -1);
            for (const auto &pt: segment) {
                tl.x = std::min(pt[0], tl.x);
                tl.y = std::min(pt[1], tl.y);
                br.x = std::max(pt[0], br.x);
                br.y = std::max(pt[1], br.y);
            }
            tl.x = std::max(tl.x - margin, 0);
            tl.y = std::max(tl.y - margin, 0);
            br.x = std::min(br.x + margin, width - 1);
            br.y = std::min(br.y + margin, height - 1);
            cv::Point2i cpt = (tl + br) / 2;
            printf("center: (%d,%d), size:%d\n", cpt.x, cpt.y, (int) segment.size());
            cv::Rect roi(tl, br);
            vector<Mat> warp_input((size_t) (ranges[index][1] - ranges[index][0] + 1)), warp_output;
            for (auto v = ranges[index][0]; v <= ranges[index][1]; ++v)
                warp_input[v - ranges[index][0]] = input[v](roi).clone();

            if (alg == GRID)
                gridStabilization(warp_input, warp_output, lambda);
            else if (alg == FLOW)
                flowStabilization(warp_input, warp_output, lambda);
            else if (alg == SUBSTAB) {
                substab::SubSpaceStabOption option;
                option.output_crop = false;
                substab::subSpaceStabilization(warp_input, warp_output, option);
            } else if (alg == TRACK) {
                trackStabilization(warp_input, warp_output, lambda, 10);
            }
            for (auto v = ranges[index][0]; v <= ranges[index][1]; ++v) {
                warp_output[v - ranges[index][0]].copyTo(output[v](roi));
            }
            index++;
        }
    }
}//namespace dynamic_stereo
