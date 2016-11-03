//
// Created by yanhang on 11/1/16.
//

#include "cinemagraph.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{

    namespace Cinemagraph {

        void CreatePixelMat(const std::vector<cv::Mat> &warped, const std::vector<Eigen::Vector2i> &segment,
                            const Eigen::Vector2i &range, cv::Mat &output) {
            CHECK(!warped.empty());
            CHECK(!segment.empty());
            output.create(range[1] - range[0] + 1, (int) segment.size(), warped[0].type());
            for (auto i = 0; i < segment.size(); ++i) {
                for (auto v = range[0]; v <= range[1]; ++v) {
                    output.at<Vec3b>(v - range[0], i) = warped[v].at<Vec3b>(segment[i][1], segment[i][0]);
                }
            }
        }

        void ComputeBlendMap(const std::vector<std::vector<Eigen::Vector2i> >& segments,
                             const int width, const int height, const int blend_R, const int min_size,
                             cv::Mat& blend_map){
            Mat mask_all_segments(height, width, CV_8UC1, Scalar::all(0));
            blend_map.create(height, width, CV_32FC1);
            blend_map.setTo(cv::Scalar::all(0));
            for(const auto& segment: segments){
                for(const auto& pid: segment){
                    CHECK_GE(pid[0], 0);
                    CHECK_GE(pid[1], 0);
                    CHECK_LT(pid[0], width);
                    CHECK_LT(pid[1], height);
                    mask_all_segments.at<uchar>(pid[1],pid[0]) = (uchar)255;
                    blend_map.at<float>(pid[1], pid[0]) = 1.0f;
                }
            }

            for(auto i=0; i<blend_R; ++i){
                Mat eroded;
                cv::erode(mask_all_segments, eroded, cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3)));
                Mat contour = mask_all_segments - eroded;
                for(auto y=0; y<height; ++y){
                    for(auto x=0; x<width; ++x){
                        if(contour.at<uchar>(y,x) > (uchar)200){
                            blend_map.at<float>(y,x) = (float)i * 1.0 / (float)blend_R;
                        }
                    }
                }
                mask_all_segments.release();
                mask_all_segments = eroded.clone();
            }

            for(const auto& segment: segments){
                if(segment.size() < min_size) {
                    for (const auto &pid: segment) {
                        blend_map.at<float>(pid[1], pid[0]) = 1.0f;
                    }
                }
            }
        }

        void RenderCinemagraph(const Cinemagraph& cinemagraph, std::vector<cv::Mat> &output, const int kOutputFrame,
                               const bool start_from_reference) {
            CHECK_EQ(cinemagraph.pixel_mat_display.size(), cinemagraph.ranges_display.size());
            CHECK_EQ(cinemagraph.pixel_mat_flashy.size(), cinemagraph.ranges_flashy.size());
            CHECK_EQ(cinemagraph.pixel_mat_display.size(), cinemagraph.pixel_loc_display.size());
            CHECK_EQ(cinemagraph.pixel_mat_flashy.size(), cinemagraph.pixel_loc_flashy.size());

            const int width = cinemagraph.background.cols;
            const int height = cinemagraph.background.rows;

            output.resize((size_t) kOutputFrame);
            for (auto i = 0; i < kOutputFrame; ++i) {
                output[i] = cinemagraph.background.clone();
            }

            //render display: back-forth
            for (auto sid = 0; sid < cinemagraph.pixel_loc_display.size(); ++sid) {
                const int kSegLength = cinemagraph.ranges_display[sid][1] - cinemagraph.ranges_display[sid][0];
                for (auto output_index = 0; output_index < output.size(); ++output_index) {
                    int fid = output_index % (kSegLength * 2);
                    if (fid >= kSegLength) {
                        fid = 2 * kSegLength - fid;
                    }
                    for (auto pid = 0; pid < cinemagraph.pixel_loc_display[sid].size(); ++pid) {
                        const Vector2i& loc = cinemagraph.pixel_loc_display[sid][pid];
                        float alpha = 1.0;
                        if(cinemagraph.blend_map.data){
                            alpha = cinemagraph.blend_map.at<float>(loc[1], loc[0]);
                        }
                        Vec3f pix = (Vec3f) cinemagraph.background.at<Vec3b>(loc[1], loc[0]) * (1 - alpha) +
                                    (Vec3f) cinemagraph.pixel_mat_display[sid].at<Vec3b>(fid, pid) * alpha;
                        output[output_index].at<Vec3b>(loc[1], loc[0]) = (Vec3b) pix;

                    }
                }
            }

            //render flashy: direct
            for (auto sid = 0; sid < cinemagraph.pixel_loc_flashy.size(); ++sid) {
                const int kSegLength = cinemagraph.ranges_flashy[sid][1] - cinemagraph.ranges_flashy[sid][0] + 1;
                for (auto output_index = 0; output_index < output.size(); ++output_index) {
                    int fid = output_index % kSegLength;
                    for (auto pid = 0; pid < cinemagraph.pixel_loc_flashy[sid].size(); ++pid) {
                        const Vector2i& loc = cinemagraph.pixel_loc_flashy[sid][pid];
                        float alpha = 1.0;
                        if(cinemagraph.blend_map.data){
                            alpha = cinemagraph.blend_map.at<float>(loc[1], loc[0]);
                        }
                        Vec3f pix = (Vec3f) cinemagraph.background.at<Vec3b>(loc[1], loc[0]) * (1 - alpha) +
                                    (Vec3f) cinemagraph.pixel_mat_flashy[sid].at<Vec3b>(fid, pid) * alpha;
                        output[output_index].at<Vec3b>(loc[1], loc[0]) = (Vec3b) pix;
                    }
                }
            }
        }

        bool ReadCinemagraph(const std::string &path, Cinemagraph& output) {

        }

        void SaveCinemagraph(const std::string &path, const Cinemagraph &output) {
            CHECK(check_cinemagraph(output));

            const int width = output.background.cols;
            const int height = output.background.rows;
            ofstream fout(path.c_str(), ios::binary);
            CHECK(fout.is_open()) << "Can not open file to write " << path;
            fout << output.reference << endl;
            fout << width << ' ' << height << endl;
            fout << output.pixel_loc_display.size() << ' ' << output.pixel_loc_flashy.size() << endl;
            fout << "location" << endl;
            for(int i=0; i<output.pixel_loc_display.size(); ++i){
                fout << output.pixel_loc_display[i].size() << endl;
                for(const auto& loc: output.pixel_loc_display[i]){
                    fout << loc[1] * width + loc[0] << ' ';
                }
                fout << endl;
            }
            for(auto i=0; i<output.pixel_loc_flashy.size(); ++i){
                fout << output.pixel_loc_flashy[i].size() << endl;
                for(const auto& loc: output.pixel_loc_flashy){
                    fout << loc[1] * width + loc[0] << ' ';
                }
                fout << endl;
            }

            fout << "range" << endl;
            for(auto i=0; i<output.ranges_display.size(); ++i){
                fout << output.ranges_display[i][0] << ' ' << output.ranges_display[i][1] << endl;
            }
            for(auto i=0; i<output.ranges_flashy.size(); ++i){
                fout << output.ranges_flashy[i][0] << ' ' << output.ranges_flashy[i][1] << endl;
            }

            fout << "pixel" << endl;
            for(auto i=0; i<output.pixel_mat_display.size(); ++i){
                Mat cur_mat = output.pixel_mat_display[i];
                CHECK(cur_mat.isContinuous());
                CHECK_EQ(cur_mat.type(), CV_8UC3);
                fout << cur_mat.cols << ' ' << cur_mat.rows << endl;
                fout.write((char*) cur_mat.data, cur_mat.cols * cur_mat.rows * cur_mat.channels() * sizeof(uchar));
                fout << endl;
            }

            for(auto i=0; i<output.pixel_mat_flashy.size(); ++i){
                Mat cur_mat = output.pixel_mat_flashy[i];
                CHECK(cur_mat.isContinuous());
                CHECK_EQ(cur_mat.type(), CV_8UC3);
                fout << cur_mat.cols << ' ' << cur_mat.rows << endl;
                fout.write((char*) cur_mat.data, cur_mat.cols * cur_mat.rows * cur_mat.channels() * sizeof(uchar));
                fout << endl;
            }
        }

    }//namespace Cinemagraph
}//namespace dynamic_stereo