//
// Created by yanhang on 11/7/16.
//

#include "auto_cinemagraph.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaoptflow.hpp>

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    void ComputeOpticalFlow(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& opt_flow){
        CHECK_GE(input.size(), 2);
        cv::Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f,50.0f,0.8f,10,77,10);
        for(int i=0; i<input.size() - 1; ++i){
            LOG(INFO) << "Flow from " << i << " to " << i + 1;
            Mat gray1, gray2;
            cvtColor(input[i], gray1, CV_BGR2GRAY);
            cvtColor(input[i+1], gray2, CV_BGR2GRAY);
            cuda::GpuMat img1GPU(gray1);
            cuda::GpuMat img2GPU(gray2);
            cuda::GpuMat flowGPU(input[i].size(), CV_32FC2);
            cuda::GpuMat img1GPUf, img2GPUf;
            img1GPU.convertTo(img1GPUf, CV_32F, 1.0 / 255.0);
            img2GPU.convertTo(img2GPUf, CV_32F, 1.0 / 255.0);
            brox->calc(img1GPUf, img2GPUf, flowGPU);
            Mat flowCPU(flowGPU.clone());
            opt_flow.push_back(flowCPU);
        }
    }

    void LoadVideo(const std::string& path, std::vector<cv::Mat>& images, int max_frame){
        VideoCapture cap(path);
        CHECK(cap.isOpened()) << "Can not open video " << path;

        while(true){
            Mat frame;
            if(!cap.read(frame)){
                break;
            }
            Mat small;
            pyrDown(frame, small);
            images.push_back(small);
            if(max_frame >=0 && images.size() == max_frame){
                break;
            }
        }
    }
    void GetPixelScore(const std::vector<cv::Mat>& opt_flow, std::vector<cv::Mat>& scores){
        scores.resize(opt_flow.size());
        for(auto v=0; v<opt_flow.size(); ++v){
            scores[v].create(opt_flow[v].size(), CV_32FC1);
            scores[v].setTo(cv::Scalar::all(0));
            for(auto y=0; y<scores[v].rows; ++y){
                for(auto x=0; x<scores[v].cols; ++x){
                    Vec2f fv = opt_flow[v].at<Vec2f>(y,x);
                    scores[v].at<float>(y,x) = (float)cv::norm(fv);
                }
            }
        }
    }

    void OptimalSpatialInterval(const std::vector<cv::Mat>& scores, std::vector<std::vector<int> >& spatial){
        CHECK(!scores.empty());
        Mat accumulated_score = scores[0].clone();
        for(auto i=1; i<scores.size(); ++i){
            accumulated_score += scores[i];
        }

        const int width = accumulated_score.cols;
        const int height = accumulated_score.rows;

        accumulated_score /= (float)scores.size();
        float constC = 1;

//        {
//            Mat vis = accumulated_score.clone();
//            double minv, maxv;
//            cv::minMaxLoc(vis, &minv, &maxv);
//            vis /= (float) maxv;
//            imshow("accumualte_score", vis);
//            waitKey(0);
//        }
//        {
            vector<float> accu_array;
            Mat temp_array = accumulated_score.reshape(1,1);
            temp_array.copyTo(accu_array);
            //nth_element(accu_array.begin(), accu_array.begin() + accu_array.size() / 2, accu_array.end());
            //constC = accu_array[accu_array.size() / 2];
            constC = std::accumulate(accu_array.begin(), accu_array.end(), 0.0f) / (float)accu_array.size();
//        }
        LOG(INFO) << "constant C: " << constC;

        for(auto y=0; y<height; ++y){
            for(auto x=0; x<width; ++x){
                accumulated_score.at<float>(y,x) = accumulated_score.at<float>(y,x) - constC;
            }
        }

        const int min_dim = 25, step = 2;
        LOG(INFO) << "Start searching";

        while(true) {
            Mat integral_score;
            cv::integral(accumulated_score, integral_score, CV_32F);

            float max_score = -100000;
            vector<int> cur_spatial(4, -1);
            for (auto x1 = 0; x1 < width - min_dim; x1 += step) {
                for (auto x2 = x1 + min_dim; x2 < width; x2 += step) {
                    for (auto y1 = 0; y1 < height - min_dim; y1 += step) {
                        for (auto y2 = y1 + min_dim; y2 < height; y2 += step) {
                            float s = integral_score.at<float>(y2, x2) -
                                      integral_score.at<float>(y2, x1) -
                                      integral_score.at<float>(y1, x2) +
                                      integral_score.at<float>(y1, x1);
                            //printf("(%d,%d)->(%d,%d), score %.3f\n", x1, y1, x2, y2, s);
                            if (s > max_score) {
                                cur_spatial[0] = x1;
                                cur_spatial[1] = y1;
                                cur_spatial[2] = x2;
                                cur_spatial[3] = y2;
                                max_score = s;
                            }
                        }
                    }
                }
            }
            for(auto y=cur_spatial[1]; y<=cur_spatial[3]; ++y){
                for(auto x=cur_spatial[0]; x<=cur_spatial[2]; ++x){
                    accumulated_score.at<float>(y,x) = -constC;
                }
            }
            spatial.push_back(cur_spatial);
            printf("Region %d, (%d,%d)->(%d,%d), score %.3f\n",
                   (int)spatial.size(), cur_spatial[0], cur_spatial[1], cur_spatial[2], cur_spatial[3], max_score);
            if(max_score < 0){
                break;
            }
        }

    }
}//namespace dynamic_stereo