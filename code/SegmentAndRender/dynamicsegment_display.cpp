//
// Created by yanhang on 4/29/16.
//

#include <unordered_set>
#include "dynamicsegment.h"
#include "../base/thread_guard.h"
#include "../VisualWord/visualword.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{

    void segmentDisplay(const FileIO& file_io, const int anchor,
                        const std::vector<cv::Mat> &input,
                        const string& classifierPath, const string& codebookPath, cv::Mat& result){
        CHECK(!input.empty());
        char buffer[1024] = {};
        const int width = input[0].cols;
        const int height = input[0].rows;

        //display region
        sprintf(buffer, "%s/midres/classification%05d.png", file_io.getDirectory().c_str(), anchor);
        Mat preSeg = imread(buffer, false);

        if(!preSeg.data) {
            const vector<float> levelList{0.60, 0.70, 0.80};
            cv::Ptr<ml::StatModel> classifier;
            Mat codebook;
            VisualWord::VisualWordOption vw_option;
            cv::FileStorage codebookIn(codebookPath, FileStorage::READ);
            CHECK(codebookIn.isOpened()) << "Can not open code book: " << codebookPath;
            codebookIn["codebook"] >> codebook;

            int pixeldesc = (int)codebookIn["pixeldesc"];
            int classifiertype = (int)codebookIn["classifiertype"];
            vw_option.pixDesc = (VisualWord::PixelDescriptor) pixeldesc;
            vw_option.classifierType = (VisualWord::ClassifierType) classifiertype;
            if(classifiertype == VisualWord::RANDOM_FOREST) {
                classifier = ml::RTrees::load<ml::RTrees>(classifierPath);
                CHECK(classifier.get());
                cout << "Tree depth: " << classifier.dynamicCast<ml::RTrees>()->getMaxDepth() << endl;
            }
            else if(classifiertype == VisualWord::BOOSTED_TREE)
                classifier = ml::Boost::load<ml::Boost>(classifierPath);
            else if(classifiertype == VisualWord::SVM)
                classifier = ml::SVM::load(classifierPath);
            CHECK(classifier.get()) << "Can not open classifier: " << classifierPath;

            //load or compute segmentation
            vector<Mat> segments;
            sprintf(buffer, "%s/midres/segment%05d.yml", file_io.getDirectory().c_str(), anchor);
            if(!video_segment::LoadHierarchicalSegmentation(string(buffer), segments)){
                VisualWord::detectVideo(input, classifier, codebook, levelList, preSeg, vw_option, cv::noArray(), segments);
                video_segment::SaveHierarchicalSegmentation(string(buffer), segments);
            }else{
                VisualWord::detectVideo(input, classifier, codebook, levelList, preSeg, vw_option, segments, cv::noArray());
            }
            cout << "Done" << endl;
            //dump out segmentation result
            for(auto i=0; i<levelList.size(); ++i){
                int lid = segments.size() * levelList[i];
                CHECK_GE(lid, 0);
                CHECK_LT(lid, segments.size());
                Mat segVis = video_segment::visualizeSegmentation(segments[lid]);
                Mat blended;
                cv::addWeighted(input[input.size()/2], 0.1, segVis, 0.9, 0.0, blended);
                sprintf(buffer, "%s/temp/videosegment%05d_%.3f.png", file_io.getDirectory().c_str(), anchor, levelList[i]);
                imwrite(buffer, blended);
            }

            sprintf(buffer, "%s/midres/classification%05d.png", file_io.getDirectory().c_str(), anchor);
            imwrite(buffer, preSeg);
        }

        Mat display_mask(preSeg.rows, preSeg.cols, CV_8UC3, cv::Scalar(255,0,0));
        for(auto y=0; y<preSeg.rows; ++y){
            for(auto x=0; x<preSeg.cols; ++x){
                if(preSeg.at<uchar>(y,x) > (uchar)200){
                    display_mask.at<Vec3b>(y,x) = Vec3b(0,0,255);
                }
            }
        }
        Mat display_overlay;
        cv::addWeighted(input[input.size()/2], 0.2, display_mask, 0.8, 0.0, display_overlay);
        sprintf(buffer, "%s/temp/segment_display%05d.jpg", file_io.getDirectory().c_str(), anchor);
        imwrite(buffer, display_overlay);
        result = video_segment::localRefinement(input, 1, 5, 200, preSeg);
    }

    void groupPixel(const cv::Mat& labels, std::vector<std::vector<Eigen::Vector2i> >& segments){
        CHECK_NOTNULL(labels.data);
        CHECK_EQ(labels.type(), CV_32S);
        double minl, maxl;
        cv::minMaxLoc(labels, &minl, &maxl);
        CHECK_LT(minl, std::numeric_limits<double>::epsilon());
        const int nLabel = (int)maxl;
        segments.clear();
        segments.resize((size_t)nLabel);
        for(auto y=0; y<labels.rows; ++y){
            for(auto x=0; x<labels.cols; ++x){
                int l = labels.at<int>(y,x);
                if(l > 0)
                    segments[l-1].push_back(Vector2i(x,y));
            }
        }
    }



}//namespace dynamic_stereo
