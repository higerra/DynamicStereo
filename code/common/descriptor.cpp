//
// Created by yanhang on 5/21/16.
//

#include <random>
#include "descriptor.h"
#include "../external/segment_ms/msImageProcessor.h"
#include "../external/segment_gb/segment-image.h"
#include "../external/line_util/line_util.h"
#include "../base/thread_guard.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
    namespace Feature {
        void normalizel2(std::vector<float> &array){
            const float epsilon = 1e-3;
            float sqsum = 0.0;
            for (auto f: array)
                sqsum += f * f;
            if (sqsum < epsilon)
                return;
            for (auto &f: array)
                f /= std::sqrt(sqsum);
        }

        void normalizeSum(std::vector<float> &array){
            const float epsilon = 1e-3;
            float sum = std::accumulate(array.begin(), array.end(), 0.0f);
            if(sum < epsilon)
                return;
            for(auto &f: array)
                f /= sum;
        }

        void RGBHist::constructFeature(const std::vector<float> &array, std::vector<float> &feat) const {
            CHECK_EQ((int) array.size() % 3, 0);
            vector<float> feat_diff((size_t) kBin * 3, 0.0f);
            vector<float> feat_intensity((size_t) kBinIntensity * 3, 0.0f);

            Vector3f RGB2Gray(0.299, 0.587, 0.114);
            //compare first half with the second half
            const int stride = array.size() / 3 / 2;

            for (auto t = 0; t < stride; ++t) {
                Vector3f pix1(array[t * 3], array[t * 3 + 1], array[t * 3 + 2]);
                Vector3f pix2(array[(t + stride) * 3], array[(t + stride) * 3 + 1], array[(t + stride) * 3 + 2]);

                //intensity
//                float intensity = pix1.dot(RGB2Gray);
//                int bidInt = floor(intensity / binUnitIntensity);
//                CHECK_LT(bidInt, kBinIntensity);
//                feat_intensity[bidInt] += 1.0;
                for(auto c=0; c<3; ++c){
                    int bid = floor(pix1[c] / binUnitIntensity);
                    CHECK_LT(kBinIntensity * c + bid, feat_intensity.size());
                    feat_intensity[kBinIntensity*c + bid] += 1.0;
                }

                //color change
                Vector3f diff = pix2 - pix1;
                if (diff.norm() >= min_diff) {
                    for (auto c = 0; c < 3; ++c) {
                        int bid = floor((diff[c] + 256) / binUnit);
                        CHECK_LT(kBin * c + bid, feat_diff.size());
                        feat_diff[kBin * c + bid] += 1.0;
                    }
                }
            }
            //normalize, cut and renormalize
            normalizel2(feat_intensity);
            normalizel2(feat_diff);
//            normalizeSum(feat_diff);
//            normalizeSum(feat_intensity);
            for (auto &f: feat_intensity) {
                if (f < cut_thres)
                    f = 0;
            }
            for (auto &f: feat_diff) {
                if (f < cut_thres)
                    f = 0;
            }
//            normalizeSum(feat_diff);
//            normalizeSum(feat_intensity);
            normalizel2(feat_intensity);
            normalizel2(feat_diff);

            feat.insert(feat.end(), feat_diff.begin(), feat_diff.end());
            feat.insert(feat.end(), feat_intensity.begin(), feat_intensity.end());
        }

        void ColorHist::constructFeature(const std::vector<float> &array, std::vector<float> &feat) const {
            const int& kChn = colorSpace.channel;
            CHECK_EQ((int) array.size() % kChn, 0);
            vector<float> feat_diff((size_t) kBins[0]+kBins[1]+kBins[2], 0.0f);
            vector<float> feat_intensity((size_t) kBinsIntensity[0]+kBinsIntensity[1]+kBinsIntensity[2], 0.0f);
            const int stride = array.size() / kChn / 2;
            vector<int> binOffset((size_t)kChn, 0);
            vector<int> binIntensityOffset((size_t)kChn, 0);
            for(auto c=1; c<kChn; ++c){
                binOffset[c] = binOffset[c-1] + kBins[c-1];
                binIntensityOffset[c] = binIntensityOffset[c-1] + kBinsIntensity[c-1];
            }
            for (auto t = 0; t < stride; ++t) {
                VectorXf pix1(kChn), pix2(kChn);
                for(auto c=0; c<kChn; ++c) {
                    pix1[c] = array[t * kChn + c];
                    pix2[c] = array[(t + stride) * kChn + c];
                }
                //intensity
//                float intensity = pix1.dot(RGB2Gray);
//                int bidInt = floor(intensity / binUnitIntensity);
//                CHECK_LT(bidInt, kBinIntensity);
//                feat_intensity[bidInt] += 1.0;
                for(auto c=0; c<kChn; ++c){
//                    printf("%d, %d %.3f, %.3f\n",t, c, pix1[c], binUnitsIntensity[c]);
                    int bid = floor((pix1[c]-colorSpace.offset[c]) / binUnitsIntensity[c]);
                    CHECK_GE(binIntensityOffset[c] + bid, 0) << binIntensityOffset[c] << ' ' << bid;
                    CHECK_LT(binIntensityOffset[c] + bid, feat_intensity.size()) << binIntensityOffset[c] << ' ' << bid;
                    feat_intensity[binIntensityOffset[c] + bid] += 1.0;
                }


                //color change
                VectorXf diff = pix2 - pix1;
                for (auto c = 0; c < kChn; ++c) {
//                    printf("%d, %d %.3f, %.3f, %.3f\n",t, c, diff[c], colorSpace.range[c], binUnits[c]);
                    int bid = floor((diff[c] + colorSpace.range[c]) / binUnits[c]);
                    CHECK_GE(binOffset[c] + bid, 0) << binOffset[c] << ' ' << bid;
                    CHECK_LT(binOffset[c] + bid, feat_diff.size()) << binOffset[c] << ' ' << bid;
                    feat_diff[binOffset[c] + bid] += 1.0;
                }
            }
            //normalize, cut and renormalize
            normalizel2(feat_intensity);
            normalizel2(feat_diff);
//            normalizeSum(feat_diff);
//            normalizeSum(feat_intensity);
            for (auto &f: feat_intensity) {
                if (f < cut_thres)
                    f = 0;
            }
            for (auto &f: feat_diff) {
                if (f < cut_thres)
                    f = 0;
            }
//            normalizeSum(feat_diff);
//            normalizeSum(feat_intensity);
            normalizel2(feat_intensity);
            normalizel2(feat_diff);

            feat.insert(feat.end(), feat_diff.begin(), feat_diff.end());
            feat.insert(feat.end(), feat_intensity.begin(), feat_intensity.end());

        }

        cv::Mat visualizeSegment(const cv::Mat& labels){
            CHECK_EQ(labels.type(), CV_32S);
            const int& width = labels.cols;
            const int& height = labels.rows;
            Mat output(height, width, CV_8UC3, Scalar::all(0));
            std::vector<cv::Vec3b> colorTable((size_t) width * height);
            std::default_random_engine generator;
            std::uniform_int_distribution<int> distribution(0, 255);
            for (int i = 0; i < width * height; i++) {
                for (int j = 0; j < 3; ++j)
                    colorTable[i][j] = (uchar) distribution(generator);
            }

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++)
                    output.at<cv::Vec3b>(y, x) = colorTable[labels.at<int>(y,x)];
            }
            return output;
        }

    }//namespace Feature
}//namespace dynamic_stereo