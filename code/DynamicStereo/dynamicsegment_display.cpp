//
// Created by yanhang on 4/29/16.
//

#include "dynamicsegment.h"
#include "../base/thread_guard.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    void DynamicSegment::computeColorConfidence(const std::vector<cv::Mat> &input, Depth &result) const {
        //compute color difference pattern
        cout << "Computing dynamic confidence..." << endl;
        const int width = input[0].cols;
        const int height = input[0].rows;
        result.initialize(width, height, 0.0);

        //search in a 7 by 7 window
        const int pR = 3;
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
                    const size_t kth = colorDiff.size() * 0.9;
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

    void DynamicSegment::segmentDisplay(const std::vector<cv::Mat> &input, cv::Mat &result) const {
        CHECK(!input.empty());
        char buffer[1024] = {};

        const int width = input[0].cols;
        const int height = input[0].rows;
        result = Mat(height, width, CV_8UC1, Scalar::all(0));

        Depth dynamicness;
        computeColorConfidence(input, dynamicness);
        dynamicness.updateStatics();

        sprintf(buffer, "%s/temp/conf_dynamicness%05d.jpg", file_io.getDirectory().c_str(), anchor);
        dynamicness.saveImage(string(buffer));

        const double dynamic_thres = dynamicness.getAverageDepth() + 3 * dynamicness.getDepthVariance();
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
        const int r1 = 5, r2 = 5;
        cv::erode(regionCan,regionCan,cv::getStructuringElement(MORPH_ELLIPSE,cv::Size(r1,r1)));
        cv::dilate(regionCan,regionCan,cv::getStructuringElement(MORPH_ELLIPSE,cv::Size(r2,r2)));

        cv::erode(staticCan,staticCan,cv::getStructuringElement(MORPH_ELLIPSE,cv::Size(r1,r1)));
        cv::dilate(staticCan,staticCan,cv::getStructuringElement(MORPH_ELLIPSE,cv::Size(r2,r2)));

        sprintf(buffer, "%s/temp/conf_dynRegion%05d.jpg", file_io.getDirectory().c_str(), anchor);
        imwrite(buffer, regionCan);

        sprintf(buffer, "%s/temp/conf_staRegion%05d.jpg", file_io.getDirectory().c_str(), anchor);
        imwrite(buffer, staticCan);

        //connect component analysis
        Mat labels, stats, centroid;
        int nLabel = cv::connectedComponentsWithStats(regionCan, labels, stats, centroid);
        const int min_area = 150;
        const int* pLabel = (int*) labels.data;
        const int min_multi = 2;

        Depth regionConfidence(width, height, 0.0);
        for(auto l=1; l<nLabel; ++l){
            printf("component %d\n", l);
            const int area = stats.at<int>(l, CC_STAT_AREA);
            if(area < min_area)
                continue;
            //search for bounding box.
            //The number of static samples inside the window should be at least twice of of area
            int br = std::max(stats.at<int>(l,CC_STAT_WIDTH), stats.at<int>(l,CC_STAT_HEIGHT));
            const int cx = stats.at<int>(l,CC_STAT_LEFT) + stats.at<int>(l,CC_STAT_WIDTH) / 2;
            const int cy = stats.at<int>(l,CC_STAT_TOP) + stats.at<int>(l,CC_STAT_HEIGHT) / 2;
            printf("label:%d, centroid:(%d,%d), area:%d, ", l, cx, cy, area);
            while(br < 500){
                double nStatic = 0.0;
                for(auto x=cx-br; x<=cx+br; ++x){
                    for(auto y=cy-br; y<=cy+br; ++y){
                        if(x >= 0 && x < width && y >= 0 && y < height) {
                            if (dynamicness(x, y) < static_thres)
                                nStatic += 1.0;
                        }
                    }
                }
                if(nStatic > min_multi * area)
                    break;
                br += 50;
            }
            printf("br:%d\n", br);

            //estimate foreground histogram and background histogram
            vector<Vec3b> psample, nsample;
            const int fR = 5;
            for(auto x=cx-br; x<=cx+br; ++x){
                for(auto y=cy-br; y<=cy+br; ++y) {
                    if (x >= 0 && y >= 0 && x < width && y < height) {
                        if (regionCan.at<uchar>(y, x) > 200) {
                            for (auto v = 0; v < input.size(); ++v)
                                psample.push_back(input[v].at<Vec3b>(y, x));
                        }
                        if (staticCan.at<uchar>(y, x) > 200) {
                            //for (auto v = 0; v < input.size(); ++v)
                            for(auto v=-1*fR; v<=fR; ++v)
                                nsample.push_back(input[anchor-offset+v].at<Vec3b>(y, x));
                        }
                    }
                }
            }

//            const int nBin = 4;
//            vector<double> histfg, histbg;
//            getHistogram(psample, histfg, nBin);
//            getHistogram(nsample, histbg, nBin);
//            CHECK_EQ(histfg.size(), histbg.size());
//            //calculate distance between histogram
//            double disHist = 0.0;
//            for(auto i=0; i<histfg.size(); ++i)
//                disHist += (histfg[i]-histbg[i]) * (histfg[i]-histbg[i]);
//
//            printf("disHist: %.3f\n", disHist);
//            cout << "Histogram:" << endl;
//            for(auto i=0; i<histfg.size(); ++i)
//                printf("%.3f ", histfg[i]);
//            cout << endl;
//            for(auto i=0; i<histfg.size(); ++i)
//                printf("%.3f ", histbg[i]);
//            cout << endl;

            Ptr<cv::ml::EM> gmmbg = cv::ml::EM::create();
            Mat nsampleMat((int)nsample.size(), 3, CV_64F);
            for(auto i=0; i<nsample.size(); ++i){
                nsampleMat.at<double>(i,0) = nsample[i][0];
                nsampleMat.at<double>(i,1) = nsample[i][1];
                nsampleMat.at<double>(i,2) = nsample[i][2];
            }
            printf("Training local background color model, number of samples:%d...\n", nsampleMat.rows);
            gmmbg->trainEM(nsampleMat);
            printf("Done\n");
            double pbg = 0.0;
            for(auto i=0; i<psample.size(); ++i){
                Mat sample(1,3, CV_64F);
                Mat prob(1, gmmbg->getClustersNumber(), CV_64F);
                sample.at<double>(0,0) = (double)psample[i][0];
                sample.at<double>(0,1) = (double)psample[i][1];
                sample.at<double>(0,2) = (double)psample[i][2];
                Vec2d pre = gmmbg->predict2(sample, prob);
                pbg += pre[0];
            }

            pbg /= (double)psample.size();
            printf("Probability of being background: %.3f\n", pbg);

            if(pbg < -15){
                for(auto i=0; i<width * height; ++i){
                    if(pLabel[i] == l)
                        result.data[i] = 255;
                }
            }
        }
    }
}//namespace dynamic_stereo