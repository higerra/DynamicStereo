//
// Created by yanhang on 4/29/16.
//

#include "dynamicsegment.h"
#include "../base/thread_guard.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
    void DynamicSegment::computeColorConfidence(const std::vector<cv::Mat> &input, Depth &result) const {
        //compute color difference pattern
        cout << "Computing dynamic confidence..." << endl;
        const int width = input[0].cols;
        const int height = input[0].rows;
        result.initialize(width, height, 0.0);

        //search in a 7 by 7 window
        const int pR = 2;
        auto threadFunc = [&](int tid, int numt){
            for(auto y=tid; y<height; y+=numt) {
                for (auto x = 0; x < width; ++x) {
                    double count = 0.0;
                    vector<double> colorDiff;
                    colorDiff.reserve(input.size());

                    for (auto i = 0; i < input.size(); ++i) {
                        if (i == anchor - offset)
                            continue;
                        Vec3b curPix = input[i].at<Vec3b>(y, x);
                        if (curPix == Vec3b(0, 0, 0)) {
                            continue;
                        }
                        double min_dis = numeric_limits<double>::max();
                        for (auto dx = -1 * pR; dx <= pR; ++dx) {
                            for (auto dy = -1 * pR; dy <= pR; ++dy) {
                                const int curx = x + dx;
                                const int cury = y + dy;
                                if (curx < 0 || cury < 0 || curx >= width || cury >= height)
                                    continue;
                                Vec3b refPix = input[anchor - offset].at<Vec3b>(cury, curx);
                                min_dis = std::min(cv::norm(refPix - curPix), min_dis);
                            }
                        }
                        colorDiff.push_back(min_dis);
                        count += 1.0;
                    }
                    if (count < 1)
                        continue;
                    const size_t kth = colorDiff.size() * 0.95;
//				sort(colorDiff.begin(), colorDiff.end(), std::less<double>());
                    nth_element(colorDiff.begin(), colorDiff.begin() + kth, colorDiff.end());
//				dynamicness(x,y) = accumulate(colorDiff.begin(), colorDiff.end(), 0.0) / count;
                    result(x, y) = colorDiff[kth];
                }
            }
        };


        const int num_thread = 6;
        vector<thread_guard> threads((size_t)num_thread);
        for(auto tid=0; tid<num_thread; ++tid){
            std::thread t(threadFunc, tid, num_thread);
            threads[tid].bind(t);
        }
        for(auto &t: threads)
            t.join();
    }

    void DynamicSegment::getHistogram(const std::vector<cv::Vec3b> &samples, std::vector<double> &hist,
                                      const int nBin) const {
        CHECK_EQ(256%nBin, 0);
        hist.resize((size_t)nBin*3, 0.0);
        const int rBin = 256 / nBin;
        for(auto& s: samples){
            vector<int> ind{(int)s[0]/rBin, (int)s[1]/rBin, (int)s[2]/rBin};
            hist[ind[0]] += 1.0;
            hist[ind[1]+nBin] += 1.0;
            hist[ind[2]+2*nBin] += 1.0;
        }
        //normalize
        const double sum = std::accumulate(hist.begin(), hist.end(), 0.0);
        //const double sum = *max_element(hist.begin(), hist.end());
        CHECK_GT(sum, 0);
        for(auto &h: hist)
            h /= sum;
    }

    //estimate nLog value based on masked region (pedestrian...)
    double DynamicSegment::computeNlogThreshold(const std::vector<cv::Mat> &input, const cv::Mat &inputMask, const int K) const {
        CHECK(!input.empty());
        const int width = input[0].cols;
        const int height = input[0].rows;
        CHECK_EQ(inputMask.cols, width);
        CHECK_EQ(inputMask.rows, height);
        CHECK_EQ(inputMask.channels(), 1);

        Mat mask = 255 - inputMask;
        const uchar* pMask = mask.data;
        //process inside patches. For each patch, randomly choose half pixels for negative samples
        const int pR = 50;
        const int fR = 5;
        const double nRatio = 0.5;

        double threshold = 0.0;
        double count = 0.0;
        Mat labels, stats, centroids;
        const int nLabel = cv::connectedComponentsWithStats(mask, labels, stats, centroids);
        const int min_area = 150;
        const int min_kSample = 1000;

        for(auto l=1; l<nLabel; ++l){
            printf("Component %d/%d\n", l, nLabel-1);
            const int area = stats.at<int>(l,CC_STAT_AREA);
            if(area < min_area)
                continue;
            const int left = stats.at<int>(l,CC_STAT_LEFT);
            const int top = stats.at<int>(l, CC_STAT_TOP);
            const int comW = stats.at<int>(l, CC_STAT_WIDTH);
            const int comH = stats.at<int>(l, CC_STAT_HEIGHT);

            for(auto cy=top+pR; cy<top+comH-pR; cy+=pR){
                for(auto cx=left+pR; cx<left+comW-pR; cx+=pR){
                    vector<Vec3b> samples;

                    for(auto x=cx-pR; x<=cx+pR; ++x){
                        for(auto y=cy-pR; y<=cy+pR; ++y){
                            if(x<0 || y<0 || x >=width || y>=height)
                                continue;
                            if(pMask[y*width+x] > 200){
                                for(auto v=-1*fR; v<=fR; ++v)
                                    samples.push_back(input[anchor-offset+v].at<Vec3b>(y,x));
                            }
                        }
                    }
                    if(samples.size() < min_kSample)
                        continue;
                    std::random_shuffle(samples.begin(), samples.end());
                    const int boundry = (int)(samples.size() * nRatio);
                    cv::Ptr<cv::ml::EM> gmmbg = cv::ml::EM::create();
                    CHECK(gmmbg.get());
                    gmmbg->setClustersNumber(K);
                    Mat nSamples(boundry, 3, CV_64F);
                    for(auto i=0; i<boundry; ++i){
                        nSamples.at<double>(i,0) = (double)samples[i][0];
                        nSamples.at<double>(i,1) = (double)samples[i][1];
                        nSamples.at<double>(i,2) = (double)samples[i][2];
                    }
                    printf("Training at (%d,%d)... Number of samples: %d\n", cx, cy, boundry);
                    gmmbg->trainEM(nSamples);

                    double curnLog = 0.0;
                    double curCount = 0.0;
                    for(auto i=boundry; i<samples.size(); ++i){
                        Mat s(1,3,CV_64F);
                        for(auto j=0; j<3; ++j)
                            s.at<double>(0,j) = (double)samples[i][j];
                        Mat prob(1, gmmbg->getClustersNumber(), CV_64F);
                        Vec2d pre = gmmbg->predict2(s, prob);
                        curnLog -= pre[0];
                        curCount += 1.0;
                    }
                    printf("Done, nLog: %.3f\n", curnLog/curCount);
                    threshold += curnLog / curCount;
                    count += 1.0;
                }
            }
        }
        CHECK_GT(count,0.9);
        return threshold / count;
    }

    void DynamicSegment::segmentDisplay(const std::vector<cv::Mat> &input, const cv::Mat& inputMask, cv::Mat& displayLabels,
                                        std::vector<std::vector<Eigen::Vector2d> >& segmentsDisplay) const {
        CHECK(!input.empty());
        CHECK(inputMask.data);
        CHECK_EQ(inputMask.channels(), 1);
        char buffer[1024] = {};

        const int width = input[0].cols;
        const int height = input[0].rows;

        Mat segnetMask;
        cv::resize(inputMask, segnetMask, cv::Size(width, height), INTER_NEAREST);

        Depth dynamicness;
        computeColorConfidence(input, dynamicness);
        dynamicness.updateStatics();

        sprintf(buffer, "%s/temp/conf_dynamicness%05d.jpg", file_io.getDirectory().c_str(), anchor);
        dynamicness.saveImage(string(buffer));

        const double dynamic_thres = dynamicness.getAverageDepth() + 2 * dynamicness.getDepthVariance();
        const double static_thres = dynamicness.getAverageDepth();

        printf("Dynamic threshold: %.3f\n", dynamic_thres);
        Mat regionCan(height, width, CV_8UC1, Scalar::all(0));
        Mat staticCan(height, width, CV_8UC1, Scalar::all(0));

        for(auto i=0; i<width * height; ++i){
            if(dynamicness[i] >= dynamic_thres)
                regionCan.data[i] = 255;
            if(dynamicness[i] <= static_thres)
                staticCan.data[i] = 255;
        }
        //morphological operation
        const int r1 = 3, r2 = 9, r3 = 11;
        cv::erode(regionCan,regionCan,cv::getStructuringElement(MORPH_ELLIPSE,cv::Size(r1,r1)));
        cv::dilate(regionCan,regionCan,cv::getStructuringElement(MORPH_ELLIPSE,cv::Size(r2,r2)));

        cv::erode(staticCan,staticCan,cv::getStructuringElement(MORPH_ELLIPSE,cv::Size(r3,r3)));
        //cv::dilate(staticCan,staticCan,cv::getStructuringElement(MORPH_ELLIPSE,cv::Size(r2,r2)));

        sprintf(buffer, "%s/temp/conf_dynRegion%05d.jpg", file_io.getDirectory().c_str(), anchor);
        imwrite(buffer, regionCan);

        sprintf(buffer, "%s/temp/conf_staRegion%05d.jpg", file_io.getDirectory().c_str(), anchor);
        imwrite(buffer, staticCan);

        //connect component analysis
        Mat labels, stats, centroid;
        int nLabel = cv::connectedComponentsWithStats(regionCan, labels, stats, centroid);
        const int min_area = 300;
        const int fR = 10;
        const int* pLabel = (int*) labels.data;
        const int min_multi = 2;
        const int kComponent = 5;
        const int min_nSample = 1000;
        const double max_areagain = 3.0;
        const double maxRatioOcclu = 0.3;

        displayLabels = Mat(height, width, labels.type(), Scalar::all(0));
        int kOutputLabel = 0;

        //compute nLog threshold
        printf("Computing nLog threshold...\n");
        //const double nLogThres = computeNlogThreshold(input, segnetMask, kComponent);
        //const double nLogThres = 20;
        //const double probThres = 0.2;

        const int testL = -1;

        Depth regionConfidence(width, height, 0.0);
        for(auto l=1; l<nLabel; ++l){
            if(testL > 0 && l != testL)
                continue;

            const int area = stats.at<int>(l, CC_STAT_AREA);
            //search for bounding box.
            //The number of static samples inside the window should be at least twice of of area

            const int cx = stats.at<int>(l,CC_STAT_LEFT) + stats.at<int>(l,CC_STAT_WIDTH) / 2;
            const int cy = stats.at<int>(l,CC_STAT_TOP) + stats.at<int>(l,CC_STAT_HEIGHT) / 2;


            printf("========================\n");
            printf("label:%d/%d, centroid:(%d,%d), area:%d\n", l, nLabel, cx, cy, area);
            if(segnetMask.at<uchar>(cy,cx) < 200)
                continue;
            if(area < min_area) {
                printf("Area too small\n");
                continue;
            }

            int nOcclu = 0;
            for(auto y=0; y<height; ++y){
                for(auto x=0; x<width; ++x){
                    if(pLabel[y*width+x] != l)
                        continue;
                    int pixOcclu = 0;
                    for(auto v=0; v<input.size(); ++v){
                        if(input[v].at<Vec3b>(y,x) == Vec3b(0,0,0))
                            pixOcclu++;
                    }
                    if(pixOcclu > (int)input.size() / 3)
                        nOcclu++;
                }
            }
            if(testL == l){
                printf("nOcclu:%d\n", nOcclu);
            }
            if(nOcclu > maxRatioOcclu * area) {
                printf("Violate occlusion constraint\n");
                continue;
            }

            if(l == testL){
                Mat tempMat = input[anchor-offset].clone();
                for(auto y=0; y<height; ++y){
                    for(auto x=0; x<width; ++x){
                        if(pLabel[y*width+x] == l)
                            tempMat.at<Vec3b>(y,x) = tempMat.at<Vec3b>(y,x) * 0.5 + Vec3b(0,0,128);
                        else
                            tempMat.at<Vec3b>(y,x) = tempMat.at<Vec3b>(y,x) * 0.5 + Vec3b(128,0,0);
                    }
                }
                sprintf(buffer, "%s/temp/component%05d_%03d.jpg", file_io.getDirectory().c_str(), anchor, l);
                imwrite(buffer, tempMat);
            }


            //refine segmentation



            Mat segRes = input[anchor-offset].clone();
            for(auto y=0; y<height; ++y){
                for(auto x=0; x<width; ++x){
                    if(labels.at<int>(y,x) == l){
                        displayLabels.at<int>(y,x) = kOutputLabel;
                        segRes.at<Vec3b>(y,x) = segRes.at<Vec3b>(y,x) * 0.5 + Vec3b(0,0,255) * 0.5;
                    }else
                        segRes.at<Vec3b>(y,x) = segRes.at<Vec3b>(y,x) * 0.5 + Vec3b(255,0,0) * 0.5;
                }
            }
            kOutputLabel++;
//            sprintf(buffer, "%s/temp/segmask_b%05d_com%03d.jpg", file_io.getDirectory().c_str(), anchor, l);
//            imwrite(buffer, segRes);
        }

        Mat labelsLarge;
        cv::resize(displayLabels, labelsLarge, cv::Size(width * downsample, height * downsample), 0, 0, INTER_NEAREST);
        segmentsDisplay.resize((size_t)kOutputLabel);
        for(auto y=0; y<labelsLarge.rows; ++y){
            for(auto x=0; x<labelsLarge.cols; ++x){
                int l = labelsLarge.at<int>(y,x);
                CHECK_LE(l, segmentsDisplay.size());
                if(l > 0){
                    segmentsDisplay[l-1].push_back(Vector2d(x,y));
                }
            }
        }
    }
}//namespace dynamic_stereo