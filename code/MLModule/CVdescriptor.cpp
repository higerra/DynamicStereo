//
// Created by yanhang on 7/20/16.
//

#include "CVdescriptor.h"
#include "mlutility.h"

using namespace std;
using namespace Eigen;

namespace dynamic_stereo{
    namespace ML {

        HoG3D::HoG3D(const int M_, const int N_, const int kSubBlock_) : M(M_), N(N_), kSubBlock(kSubBlock_) {
            dim = 20 * M * M * N;
            const double fi = (1.0 + sqrt(5.0)) / 2;
            P.resize(20, 3);
            P << 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1,
                    0, 1 / fi, fi, 0, 1 / fi, -fi, 0, -1 / fi, fi, 0, -1 / fi, -fi,
                    1 / fi, fi, 0, 1 / fi, -fi, 0, -1 / fi, fi, 0, -1 / fi, -fi, 0,
                    fi, 0, 1 / fi, fi, 0, -1 / fi, -fi, 0, 1 / fi, -fi, 0, -1 / fi;
        }

        void HoG3D::constructFeature(const std::vector<cv::Mat> &images, std::vector<float> &feat) const {
            CHECK_GE(images.size(), N * kSubBlock);
            CHECK_GE(images[0].cols, M * kSubBlock);
            CHECK_GE(images[0].rows, M * kSubBlock);
            CHECK_EQ(images[0].type(), CV_32FC3);
            auto computeBlock = [&](int x0, int y0, int z0, int x1, int y1, int z1) {
                Vector3f res(0, 0, 0);
                for (auto x = x0; x <= x1; ++x) {
                    for (auto y = y0; y <= y1; ++y) {
                        for (auto z = z0; z <= z1; ++z) {
                            cv::Vec3f gb = images[z].at<cv::Vec3f>(y, x);
                            res += Vector3f(gb[0], gb[1], gb[2]);
                        }
                    }
                }
                float count = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1);
                res /= count;
                return res;
            };

            auto computeCell = [&](int x0, int y0, int z0, int x1, int y1, int z1, vector<float> &res) {
                int bx = (x1 - x0 + 1) / kSubBlock;
                int by = (y1 - y0 + 1) / kSubBlock;
                int bz = (z1 - z0 + 1) / kSubBlock;
                const float t = 1.29107f;
                res.resize((size_t) P.rows(), 0.0f);
                for (auto x = 0; x < kSubBlock; ++x) {
                    for (auto y = 0; y < kSubBlock; ++y) {
                        for (auto z = 0; z < kSubBlock; ++z) {
                            Vector3f gb = computeBlock(x0 + x * bx, y0 + y * by, z0 + z * bz,
                                                       x0 + (x + 1) * bx - 1, y0 + (y + 1) * by - 1,
                                                       z0 + (z + 1) * bz - 1);
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
            int cz = (int) images.size() / N;
            for (auto x = 0; x < M; ++x) {
                for (auto y = 0; y < M; ++y) {
                    for (auto z = 0; z < N; ++z) {
                        vector<float> hist;
                        computeCell(x * cx, y * cy, z * cz,
                                    (x + 1) * cx - 1, (y + 1) * cy - 1, (z + 1) * cz - 1, hist);
                        feat.insert(feat.end(), hist.begin(), hist.end());
                    }
                }
            }

            const float cut_thres = 0.25;
            MLUtility::normalizel2(feat);
//            for(auto& v: feat){
//                if(v < cut_thres)
//                    v = 0.0f;
//            }
//            normalizel2(feat);
        }


        void Color3D::constructFeature(const std::vector<cv::Mat> &images, std::vector<float> &feat) const {
            CHECK_GE(images.size(), N);
            CHECK_GE(images[0].cols, M);
            CHECK_GE(images[0].rows, M);
            CHECK_EQ(images[0].type(), CV_32FC3);

            //average color inside one cell
            auto computeCell = [&](int x0, int y0, int z0, int x1, int y1, int z1, vector<float> &res) {
                res.resize((size_t) images[0].channels(), 0.0f);
                for (auto x = x0; x <= x1; ++x) {
                    for (auto y = y0; y <= y1; ++y) {
                        for (auto z = z0; z <= z1; ++z) {
                            cv::Vec3f pix = images[z].at<cv::Vec3f>(y, x);
                            for (auto i = 0; i < res.size(); ++i)
                                res[i] += pix[i];
                        }
                    }
                }
                const int count = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1);
                for (auto &r: res)
                    r /= (float) count;
            };

            int cx = images[0].cols / M;
            int cy = images[0].rows / M;
            int cz = (int) images.size() / N;
            for (auto x = 0; x < M; ++x) {
                for (auto y = 0; y < M; ++y) {
                    for (auto z = 0; z < N; ++z) {
                        vector<float> hist;
                        computeCell(x * cx, y * cy, z * cz,
                                    (x + 1) * cx - 1, (y + 1) * cy - 1, (z + 1) * cz - 1, hist);
                        feat.insert(feat.end(), hist.begin(), hist.end());
                    }
                }
            }
        }
    }//namespace ML
}//namespace dynamic_stereo

namespace cv {
    CVHoG3D::CVHoG3D(const int ss_, const int sr_, const int M_, const int N_, const int kSubBlock_)
            : M(M_), N(N_), kSubBlock(kSubBlock_), sigma_s(ss_),  sigma_r(sr_){}

    void CVHoG3D::prepareImage(const std::vector<cv::Mat> &input, std::vector<cv::Mat> &output) const {
        CHECK(!input.empty());
        dynamic_stereo::ML::MLUtility::compute3DGradient(input, output);
    }
    void CVHoG3D::compute(const _InputArray &image, std::vector<KeyPoint> &keypoints,
                          const _OutputArray &descriptors) {
        CHECK(!keypoints.empty());
        vector<Mat> input;
        image.getMatVector(input);
        CHECK_GE(input.size(), sigma_r);
        dynamic_stereo::ML::HoG3D hog3D(M,N,kSubBlock);

        descriptors.create((int)keypoints.size(), hog3D.getDim(), CV_32FC1);
        Mat& descriptors_ = descriptors.getMatRef();

        for(auto i=0; i<keypoints.size(); ++i){
            vector<Mat> subVideo((size_t)sigma_r);
            const Point2f& pt = keypoints[i].pt;
            cv::Rect roi((int)pt.x - sigma_s/2, (int)pt.y-sigma_s/2, sigma_s, sigma_s);

            const int startId = keypoints[i].octave - sigma_r / 2;
            const int endId = startId + sigma_r - 1;
            //boundary check
            CHECK_GE(roi.x, 0);
            CHECK_GE(roi.y, 0);
            CHECK_LT(roi.br().x, input[0].cols);
            CHECK_LT(roi.br().y, input[0].rows);
            CHECK_GE(startId, 0);
            CHECK_LT(endId, input.size());

            for(auto v=startId; v<=endId; ++v)
                subVideo[v - startId] = input[v](roi);

            vector<float> feat;
            hog3D.constructFeature(subVideo, feat);

            for(auto j=0; j<descriptors.cols(); ++j) {
                descriptors_.at<float>(i, j) = feat[j];
            }
        }

    }


    CVColor3D::CVColor3D(const int ss_, const int sr_, const int M_, const int N_)
            :sigma_s(ss_), sigma_r(sr_), M(M_), N(N_){}

    void CVColor3D::prepareImage(const std::vector<cv::Mat> &input, std::vector<cv::Mat> &output) const {
        CHECK(!input.empty());
        output.resize(input.size());
        for(auto v=0; v<input.size(); ++v){
            input[v].convertTo(output[v], CV_32F);
        }
    }
    void CVColor3D::compute(InputArray image,
                         CV_OUT CV_IN_OUT std::vector<KeyPoint> &keypoints,
                         OutputArray descriptors){
        CHECK(!keypoints.empty());
        vector<Mat> input;
        image.getMatVector(input);
        CHECK_GE(input.size(), sigma_r);

        const int kChannel = M*M*N*input[0].channels();

        descriptors.create((int)keypoints.size(), kChannel, CV_32FC1);
        Mat& descriptors_ = descriptors.getMatRef();

        dynamic_stereo::ML::Color3D color3d(M, N);
        for(auto i=0; i<keypoints.size(); ++i){
            vector<Mat> subVideo((size_t)sigma_r);
            const Point2f& pt = keypoints[i].pt;
            cv::Rect roi((int)pt.x - sigma_s/2, (int)pt.y-sigma_s/2, sigma_s, sigma_s);

            const int startId = keypoints[i].octave - sigma_r / 2;
            const int endId = startId + sigma_r - 1;
            //boundary check
            CHECK_GE(roi.x, 0);
            CHECK_GE(roi.y, 0);
            CHECK_LT(roi.br().x, input[0].cols);
            CHECK_LT(roi.br().y, input[0].rows);
            CHECK_GE(startId, 0);
            CHECK_LT(endId, input.size());

            for(auto v=startId; v<=endId; ++v)
                subVideo[v - startId] = input[v](roi);

            vector<float> feat;
            color3d.constructFeature(subVideo, feat);

            for(auto j=0; j<descriptors.cols(); ++j) {
                descriptors_.at<float>(i, j) = feat[j];
            }
        }
    }
}//namespace cv