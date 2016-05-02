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
                    const size_t kth = colorDiff.size() * 0.8;
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

    void DynamicSegment::segmentDisplay(const std::vector<cv::Mat> &input, const cv::Mat& inputMask, cv::Mat &result) const {
        CHECK(!input.empty());
        CHECK(inputMask.data);
        CHECK_EQ(inputMask.channels(), 1);
        char buffer[1024] = {};

        const int width = input[0].cols;
        const int height = input[0].rows;

        Mat segnetMask;
        cv::resize(inputMask, segnetMask, cv::Size(width, height), INTER_NEAREST);

        result = Mat(height, width, CV_8UC1, Scalar::all(0));

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
        const int r1 = 3, r2 = 7, r3 = 11;
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
        const int* pLabel = (int*) labels.data;
        const int min_multi = 2;
        const int kComponent = 5;
        //compute nLog threshold
        printf("Computing nLog threshold...\n");
        //const double nLogThres = computeNlogThreshold(input, segnetMask, kComponent);
        //const double nLogThres = 17;
        const double probThres = 0.2;

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
            if(area < min_area)
                continue;

            int br = std::max(stats.at<int>(l,CC_STAT_WIDTH)/2, stats.at<int>(l,CC_STAT_HEIGHT)/2);
            printf("Init br: %d\n", br);
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
                br += 5;
            }
            printf("br:%d\n", br);

            //estimate foreground histogram and background histogram
            vector<Vec3b> nsample;
            vector<vector<Vec3b> > psample(input.size());
            const int fR = 5;
            for(auto x=cx-br; x<=cx+br; ++x){
                for(auto y=cy-br; y<=cy+br; ++y) {
                    if (x >= 0 && y >= 0 && x < width && y < height) {
                        if (regionCan.at<uchar>(y, x) > 200) {
                            for (auto v = 0; v < input.size(); ++v)
                                psample[v].push_back(input[v].at<Vec3b>(y, x));
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

            Mat tempMat = input[anchor-offset].clone();
            cv::rectangle(tempMat, cv::Point(cx-br,cy-br), cv::Point(cx+br, cy+br), cv::Scalar(0,0,255));
            sprintf(buffer, "%s/temp/sample_region%05d_com%05d.jpg", file_io.getDirectory().c_str(), anchor, l);
            imwrite(buffer, tempMat);

            if(l == testL){
		        printf("Dummping out samples...\n");
                Mat tempMat = input[anchor-offset].clone();
                cv::rectangle(tempMat, cv::Point(cx-br,cy-br), cv::Point(cx+br, cy+br), cv::Scalar(0,0,255));
                sprintf(buffer, "%s/temp/sample_region%05d_com%05d.jpg", file_io.getDirectory().c_str(), anchor, l);
                imwrite(buffer, tempMat);

		        sprintf(buffer, "%s/temp/sample_train%05d_com%05d.txt", file_io.getDirectory().c_str(), anchor, l);
		        ofstream fout(buffer);
		        CHECK(fout.is_open());
		        for(auto i=0; i<nsample.size(); ++i)
			        fout << (int)nsample[i][0] << ' ' << (int)nsample[i][1] << ' ' << (int)nsample[i][2] << endl;
		        fout.close();
		        sprintf(buffer, "%s/temp/sample_test%05d_com%05d.txt", file_io.getDirectory().c_str(), anchor, l);
		        fout.open(buffer);
		        CHECK(fout.is_open());
		        for(auto i=0; i<psample.size(); ++i)
                    for(auto j=0; j<psample[i].size(); ++j)
                        fout << (int)psample[i][j][0] << ' ' << (int)psample[i][j][1] << ' ' << (int)psample[i][j][2] << endl;
		        fout.close();
	        }

            Ptr<cv::ml::EM> gmmbg = cv::ml::EM::create();
            gmmbg->setClustersNumber(kComponent);
            Mat nsampleMat((int)nsample.size(), 3, CV_64F);
            for(auto i=0; i<nsample.size(); ++i){
                nsampleMat.at<double>(i,0) = nsample[i][0];
                nsampleMat.at<double>(i,1) = nsample[i][1];
                nsampleMat.at<double>(i,2) = nsample[i][2];
            }
            printf("Training local background color model, number of samples:%d...\n", nsampleMat.rows);
            gmmbg->trainEM(nsampleMat);
	        printf("Done. Means of component gaussian models:\n");
	        Mat means = gmmbg->getMeans();
            Mat weights = gmmbg->getWeights();
            const double* pGmmWeights = (double*) weights.data;

	        for(auto i=0; i<means.rows; ++i){
		        printf("(%.2f,%.2f,%.2f)\n", means.at<double>(i,0), means.at<double>(i,1), means.at<double>(i,2));
	        }
            printf("Done\n");
            vector<double> pnLogs(psample.size());
            for(auto i=0; i<psample.size(); ++i){
                double pbg = 0.0;
                for(auto j=0; j<psample[i].size(); ++j) {
                    Mat sample(1, 3, CV_64F);
                    Mat prob(1, gmmbg->getClustersNumber(), CV_64F);
                    sample.at<double>(0, 0) = (double) psample[i][j][0];
                    sample.at<double>(0, 1) = (double) psample[i][j][1];
                    sample.at<double>(0, 2) = (double) psample[i][j][2];
                    Vec2d pre = gmmbg->predict2(sample, prob);
                    double curProb = 0.0;
                    for(auto clu=0; clu<gmmbg->getClustersNumber(); ++clu){
                        curProb += prob.at<double>(0, clu) * pGmmWeights[clu];
                    }
                    //pbg -= pre[0];
                    pbg += curProb;
                }
                pnLogs[i] = pbg / (double)psample[i].size();
            }
            const size_t pProbth = pnLogs.size() * 0.8;
            nth_element(pnLogs.begin(), pnLogs.begin()+pProbth, pnLogs.end());
            printf("Probability: %.3f\n", pnLogs[pProbth]);
            if(pnLogs[pProbth] < probThres ){
                for(auto i=0; i<width * height; ++i){
                    if(pLabel[i] == l)
                        result.data[i] = 255;
                }
            }
        }
    }
}//namespace dynamic_stereo