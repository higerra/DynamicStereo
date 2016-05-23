//
// Created by yanhang on 5/21/16.
//

#include "descriptor.h"
#include "../external/segment_ms/msImageProcessor.h"
#include "../base/thread_guard.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
    namespace Feature {
        void FeatureConstructor::normalizel2(std::vector<float> &array) const {
            const float epsilon = 1e-3;
            float sqsum = 0.0;
            for (auto f: array)
                sqsum += f * f;
            if (sqsum < epsilon)
                return;
            for (auto &f: array)
                f /= std::sqrt(sqsum);
        }

        void FeatureConstructor::normalizeSum(std::vector<float> &array) const {
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

        void clusterRGBHist(const std::vector<cv::Mat>& input, std::vector<std::vector<Eigen::Vector2i> >& cluster, const int kBin){

        }

        void clusterRGBStat(const std::vector<cv::Mat>& input, std::vector<std::vector<Eigen::Vector2i> >& cluster){
            //cluster the mean median color using meanshift
            CHECK(!input.empty());
            const int width = input[0].cols;
            const int height = input[0].rows;
            const int chn = input[0].channels();

            Mat medImage(height, width, CV_8UC3, Scalar::all(0));
            printf("Computing median image...\n");
            const size_t kth = input.size() / 2;
            auto threadFun = [&](const int tid, const int nt){
                for(auto y=tid; y < height; y+=nt){
                    for(auto x=0; x < width; ++x){
                        vector<vector<uchar> > pv(3, vector<uchar>(input.size(),0));
                        for(auto c=0; c<chn; ++c){
                            for(auto v=0; v<input.size(); ++v)
                                pv[c][v] = input[v].at<Vec3b>(y,x)[c];
                            nth_element(pv[c].begin(), pv[c].begin()+kth, pv[c].end());
                            medImage.at<Vec3b>(y,x)[c] = pv[c][kth];
                        }
                    }
                }
            };

            const int num_thread = 6;
            vector<thread_guard> threads(num_thread);
            for(auto tid=0; tid < num_thread; ++tid){
                std::thread t(threadFun, tid, num_thread);
                threads[tid].bind(t);
            }
            for(auto &t: threads)
                t.join();

            Mat vis;
            cvtColor(medImage, vis, CV_RGB2BGR);
            imshow("Median image", vis);
            waitKey(0);
        }

    }//namespace Feature
}//namespace dynamic_stereo
