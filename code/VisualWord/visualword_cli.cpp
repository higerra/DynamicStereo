//
// Created by yanhang on 7/19/16.
//

#include <gflags/gflags.h>
#include "visualword.h"
#include "../external/video_segmentation/segment_util/segmentation_io.h"
#include "../external/video_segmentation/segment_util/segmentation_util.h"

using namespace std;
using namespace cv;
using namespace dynamic_stereo;

DEFINE_string(mode, "train", "train, test or detect");
DEFINE_string(cache, "", "cache path for training data");
DEFINE_string(model, "", "path to trained model");
DEFINE_string(codebook, "", "path to code book");
DEFINE_int32(kCluster, 200, "number of clusters");
DEFINE_string(classifier, "rf", "random forest(rf) or boosted tree(bt), or SVM(svm)");
DEFINE_int32(numTree, -1, "number of trees");
DEFINE_int32(treeDepth, -1, "max depth of trees");

void run_train(int argc, char** argv);
void run_test(int argc, char** argv);
void run_detect(int argc, char** argv);

static vector<float> levelList{0.1,0.2,0.3};

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    if(FLAGS_mode == "train"){
        run_train(argc, argv);
    }else if(FLAGS_mode == "test"){
        run_test(argc, argv);
    }else if(FLAGS_mode == "detect"){
        run_detect(argc, argv);
    }else{
        cerr << "Unsupported mode" << endl;
        return 1;
    }
    return 0;
}

void run_train(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: ./VisualWord --mode=train <path-to-list>" << endl;
        return;
    }
    char buffer[256] = {};
    cv::Ptr<ml::TrainData> traindata = ml::TrainData::loadFromCSV(FLAGS_cache, 0);

    if(!traindata.get()) {
        string list_path = string(argv[1]);
        ifstream listIn(list_path.c_str());
        string dir = list_path.substr(0, list_path.find_last_of("/"));
        dir.append("/");
        CHECK(listIn.is_open()) << "Can not open list file: " << argv[1];

        string filename, gtname;

        VisualWordOption option;

        cv::CVHoG3D hog3D(24, 24);

        Mat descriptors;

        vector<vector<float> > segmentsFeature;
        vector<int> response;
        //descriptorMap: the id of descriptor contained by each segment
        vector<vector<int> > descriptorMap;

        while (listIn >> filename >> gtname) {
            vector<Mat> images;
            cv::VideoCapture cap(dir + filename);
            CHECK(cap.isOpened()) << "Can not open video: " << dir + filename;
            printf("Loading %s...\n", filename.c_str());
            while (true) {
                Mat frame;
                if (!cap.read(frame))
                    break;
                images.push_back(frame);
            }
            printf("number of frames: %d\n", (int) images.size());
            vector<Mat> gradient;
            Feature::compute3DGradient(images, gradient);

            Mat gt = imread(dir + gtname, false);
            CHECK(gt.data) << "Can not read ground truth mask: " << dir + gtname;

            //load segment
            printf("Loading segments...\n");
            for (auto level: levelList) {
                vector<Mat> segments;
                sprintf(buffer, "%s/segmentation/%s.pb", dir.c_str(), filename.c_str());
                segmentation::readSegmentAsMat(string(buffer), segments, level);
                Feature::compressSegments(segments);
                vector<vector<vector<int> > > pixelGroup;
                vector<vector<int> > segmentRegion;
                printf("Level %.2f, %d segments, extracting region feature...\n", level, (int) pixelGroup.size());
                Feature::regroupSegments(segments, pixelGroup, segmentRegion);
                vector<int> curResponse;
                Feature::assignSegmentLabel(pixelGroup, gt, curResponse);
                response.insert(response.end(), curResponse.begin(), curResponse.end());
                for (const auto &pg: pixelGroup) {
                    vector<float> curRegionFeat;
                    vector<float> color, shape, position;
                    Feature::computeColor(images, pg, color);
                    Feature::computeShapeAndLength(pg, images[0].cols, images[0].rows, shape);
                    Feature::computePosition(pg, images[0].cols, images[0].rows, position);
                    curRegionFeat.insert(curRegionFeat.end(), color.begin(), color.end());
                    curRegionFeat.insert(curRegionFeat.end(), shape.begin(), shape.end());
                    curRegionFeat.insert(curRegionFeat.end(), position.begin(), position.end());
                    segmentsFeature.push_back(curRegionFeat);
                }

                //sample keypoints and extract 3D HoG
                const int kOffset = descriptors.rows; //number of descriptors so far
                printf("Extracting descriptors...\n");
                vector<KeyPoint> keypoints;
                sampleKeyPoints(gradient, keypoints, option);
                Mat curDescriptor;
                hog3D.compute(gradient, keypoints, curDescriptor);
                cv::hconcat(descriptors, curDescriptor, descriptors);

                //update descriptor map
                printf("Updating descriptor map...\n");
                vector<vector<int> > curDescMap(pixelGroup.size());
                for (auto i = 0; i < keypoints.size(); ++i) {
                    const int sid = segments[keypoints[i].octave].at<int>(keypoints[i].pt);
                    curDescMap[sid].push_back(i + kOffset);
                }
                descriptorMap.insert(descriptorMap.end(), curDescMap.begin(), curDescMap.end());
            }
        }

        //Sanity check
        CHECK_EQ(descriptorMap.size(), segmentsFeature.size());
        CHECK_EQ(descriptorMap.size(), response.size());
        CHECK(!segmentsFeature.empty());

        //construct visual word
        printf("Constructing visual words...\n");
        Mat visualWord;
        string path_codebook;
        if (FLAGS_codebook.empty()) {
            if (FLAGS_model.empty())
                path_codebook = dir + "codebook.txt";
            else
                path_codebook = FLAGS_model + "_codebook.txt";
        } else
            path_codebook = FLAGS_codebook;
        if (!loadCodebook(path_codebook, visualWord)) {
            cv::BOWKMeansTrainer bowTrainer(FLAGS_kCluster);
            visualWord = bowTrainer.cluster(descriptors);
            writeCodebook(path_codebook, visualWord);
        }

        const int kChannel = (int) segmentsFeature[0].size() + visualWord.rows;
        const int kSample = (int) segmentsFeature.size();

        //train random forest
        Mat featureMat(kSample, kChannel, CV_32FC1, Scalar::all(0)), responseMat(kSample, 1, CV_32SC1, response.data());

        //assign each descriptor to a cluster sample
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
        vector<DMatch> matches;
        matcher->match(descriptors, visualWord, matches);

        const int regionOffset = visualWord.rows;

        for (auto i = 0; i < kSample; ++i) {
            //histogram of visual words
            vector<float> vwHist((size_t) visualWord.rows, 0.0f);
            for (auto j = 0; j < descriptorMap[i].size(); ++j) {
                const int did = descriptorMap[i][j];
                vwHist[matches[did].queryIdx] += 1.0f;
            }
            Feature::normalizeSum(vwHist);
            for (auto j = 0; j < vwHist.size(); ++j)
                featureMat.at<float>(i, j) = vwHist[j];
            for (auto j = 0; j < segmentsFeature[i].size(); ++j)
                featureMat.at<float>(i, j + regionOffset) = segmentsFeature[i][j];
        }
        //release some memory
        segmentsFeature.clear();
        descriptorMap.clear();
        traindata = ml::TrainData::create(featureMat, ml::ROW_SAMPLE, responseMat);
        if(!FLAGS_cache.empty())
            writeTrainData(FLAGS_cache, traindata);
    }


    cv::Ptr<ml::StatModel> classifier;
    string classifier_path = FLAGS_model;
    if(classifier_path.empty()){
        classifier_path =  "model";
    }
    if(FLAGS_classifier == "rf"){
        if(FLAGS_model.empty())
            classifier_path.append(".rf");
        classifier = ml::RTrees::create();
        if(FLAGS_treeDepth > 0)
            classifier.dynamicCast<ml::DTrees>()->setMaxDepth(FLAGS_treeDepth);
    }else if(FLAGS_classifier == "bt") {
        if (FLAGS_model.empty())
            classifier_path.append(".bt");
        classifier = ml::Boost::create();
        if(FLAGS_treeDepth > 0)
            classifier.dynamicCast<ml::DTrees>()->setMaxDepth(FLAGS_treeDepth);
        if (FLAGS_numTree > 0)
            classifier.dynamicCast<ml::Boost>()->setWeakCount(FLAGS_numTree);
        printf("Number of trees: %d\n", classifier.dynamicCast<ml::Boost>()->getWeakCount());
    }else if(FLAGS_classifier == "svm"){
        if (FLAGS_model.empty())
            classifier_path.append(".svm");
    }else{
        cerr << "Unsupported classifier." << endl;
        return;
    }

    printf("Training...\n");
    classifier->train(traindata);

    double acc = testClassifier(traindata, classifier);
    printf("Saving %s\n", classifier_path.c_str());
    CHECK_NOTNULL(classifier.get())->save(classifier_path);
    printf("Training accuracy: %.3f\n", acc);
}

void run_test(int argc, char** argv){

}
void run_detect(int argc, char** argv){

}