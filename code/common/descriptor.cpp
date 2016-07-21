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


        void compute3DGradient(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& gradient){
            CHECK_GE(input.size(), 2);
            gradient.resize(input.size());
            vector<Mat> grays(input.size());
            for(auto v=0; v<input.size(); ++v) {
                cvtColor(input[v], grays[v], CV_BGR2GRAY);
            }
            for(auto v=0; v<input.size(); ++v) {
                gradient[v].create(input[v].size(), CV_32FC3);
                vector<Mat> curG(3);
                cv::Sobel(grays[v], curG[0], CV_32F, 1, 0);
                cv::Sobel(grays[v], curG[1], CV_32F, 0, 1);
                curG[2].create(input[v].size(), CV_32FC1);
                for (auto y = 0; y < input[v].rows; ++y) {
                    for (auto x = 0; x < input[v].cols; ++x) {
                        if (v == 0) {
                            curG[2].at<float>(y, x) =
                                    (float) grays[v + 1].at<uchar>(y, x) - (float) grays[v].at<uchar>(y, x);
                        } else if (v == input.size() - 1) {
                            curG[2].at<float>(y, x) =
                                    (float) grays[v].at<uchar>(y, x) - (float) grays[v - 1].at<uchar>(y, x);
                        } else {
                            curG[2].at<float>(y, x) =
                                    ((float) grays[v + 1].at<uchar>(y, x) - (float) grays[v - 1].at<uchar>(y, x)) / 2;
                        }
                    }
                }
                cv::merge(curG, gradient[v]);
            }
        }


        HoG3D::HoG3D(const int M_, const int N_, const int kSubBlock_): M(M_), N(N_), kSubBlock(kSubBlock_) {
            dim = 20 * M * M * N;
            const double fi = (1.0 + sqrt(5.0)) / 2;
            P.resize(20, 3);
            P << 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1,
                    0, 1 / fi, fi, 0, 1 / fi, -fi, 0, -1 / fi, fi, 0, -1 / fi, -fi,
                    1 / fi, fi, 0, 1 / fi, -fi, 0, -1 / fi, fi, 0, -1 / fi, -fi, 0,
                    fi, 0, 1 / fi, fi, 0, -1 / fi, -fi, 0, 1 / fi, -fi, 0, -1 / fi;
        }

        void HoG3D::constructFeature(const std::vector<cv::Mat> &images, std::vector<float> &feat) const {
            CHECK_GE(images.size(), N*kSubBlock);
            CHECK_GE(images[0].cols, M*kSubBlock);
            CHECK_GE(images[0].rows, M*kSubBlock);
            CHECK_EQ(images[0].type(), CV_32FC3);
            auto computeBlock = [&](int x0, int y0, int z0, int x1, int y1, int z1) {
                Vector3f res(0, 0, 0);
                for (auto x = x0; x <= x1; ++x) {
                    for (auto y = y0; y <= y1; ++y) {
                        for (auto z = z0; z <= z1; ++z) {
                            Vec3f gb = images[z].at<Vec3f>(y, x);
                            res += Vector3f(gb[0], gb[1], gb[2]);
                        }
                    }
                }
                float count = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1);
                res /= count;
                return res;
            };

            auto computeCell = [&](int x0, int y0, int z0, int x1, int y1, int z1, vector<float>& res) {
                int bx = (x1 - x0 + 1) / kSubBlock;
                int by = (y1 - y0 + 1) / kSubBlock;
                int bz = (z1 - z0 + 1) / kSubBlock;
                const float t = 1.29107f;
                res.resize((size_t) P.rows(), 0.0f);
                for (auto x = 0; x < kSubBlock; ++x) {
                    for (auto y = 0; y < kSubBlock; ++y) {
                        for (auto z = 0; z < kSubBlock; ++z) {
                            Vector3f gb = computeBlock(x0+ x * bx, y0 + y * by, z0 + z * bz,
                                                       x0 + (x + 1) * bx - 1, y0 + (y + 1) * by - 1, z0 + (z + 1) * bz - 1);
                            float mag = gb.norm();
                            if (mag < std::numeric_limits<float>::min())
                                continue;
                            VectorXf proj = P * gb / mag;
                            for (auto i = 0; i < proj.rows(); ++i) {
                                proj[i] = (proj[i] >= t) ? proj[i] - t : 0.0f;
                            }

                            float projNorm = proj.norm();
                            if (projNorm < std::numeric_limits<float>::min())
                                continue;
                            VectorXf qb = mag * proj / (proj.norm());
                            for (auto i = 0; i < res.size(); ++i)
                                res[i] += qb[i];
                        }
                    }
                }
            };

            int cx = images[0].cols / M;
            int cy = images[0].rows / M;
            int cz = (int)images.size() / N;
            for(auto x=0; x<M; ++x){
                for(auto y=0; y<M; ++y){
                    for(auto z=0; z<N; ++z){
                        vector<float> hist;
                        computeCell(x*cx, y*cy, z*cz,
                                    (x+1)*cx-1, (y+1)*cy-1, (z+1)*cz-1, hist);
                        feat.insert(feat.end(), hist.begin(), hist.end());
                    }
                }
            }

            const float cut_thres = 0.25;
            normalizel2(feat);
            for(auto& v: feat){
                if(v < cut_thres)
                    v = 0.0f;
            }
            normalizel2(feat);
        }

    }//namespace Feature
}//namespace dynamic_stereo