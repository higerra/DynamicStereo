//
// Created by yanhang on 7/19/16.
//

#include <gflags/gflags.h>
#include "visualword.h"

using namespace std;
using namespace cv;
using namespace dynamic_stereo;
using namespace VisualWord;

DEFINE_string(mode, "train", "train, test or detect");

DEFINE_string(desc, "hog3d", "type of descriptor: hog3d or color3d");
//path
DEFINE_string(cache, "", "cache path for training data");
DEFINE_string(model, "", "path to trained model");
DEFINE_string(codebook, "", "path to code book");
DEFINE_string(classifier, "rf", "random forest(rf) or boosted tree(bt), or SVM(svm)");
DEFINE_string(validation, "", "path to validation set");

//hyperparameter
//trees
DEFINE_int32(kCluster, 50, "number of clusters");
DEFINE_int32(sigma_s, 12, "spatial window size");
DEFINE_int32(sigma_r, 24, "temporal window size");
DEFINE_int32(numTree, 30, "number of trees");
DEFINE_int32(treeDepth, -1, "max depth of trees");
//svm
DEFINE_string(svmKernel, "chi2", "svm kernel type");
DEFINE_double(svmC, 1.0, "C parameter in svm");
DEFINE_double(svmGamma, 0.5, "Gamma parameter in svm");

cv::Ptr<cv::ml::TrainData> run_extract(int argc, char** argv, const VisualWordOption& vw_option);
cv::Ptr<cv::ml::StatModel> run_train(cv::Ptr<cv::ml::TrainData> traindata, const VisualWordOption& vw_option);

void run_test(int argc, char** argv);
void run_detect(int argc, char** argv, const VisualWordOption& vw_option);
void run_multiExtract(const std::vector<int>& kClusters, int argc, char** argv, const VisualWordOption& vw_option);

VisualWordOption setOption();

static const vector<float> levelList{10.0, 20.0, 30.0};
static const int smoothSize = 9;
static const int minSize = 200;
static const float theta = 100;

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    VisualWordOption vw_option = setOption();

    if(FLAGS_mode == "extract"){
        cv::Ptr<ml::TrainData> traindata = run_extract(argc, argv, vw_option);
        CHECK(traindata.get());
        if(!FLAGS_cache.empty())
            ML::MLUtility::writeTrainData(FLAGS_cache, traindata);
    }else if(FLAGS_mode == "multiExtract"){
        vector<int> kClusters{50,100,200};
        run_multiExtract(kClusters, argc, argv, vw_option);
    }else if(FLAGS_mode == "train"){
        cv::Ptr<ml::TrainData> traindata = ml::TrainData::loadFromCSV(FLAGS_cache, 0);
        if(!traindata.get()) {
            traindata = run_extract(argc, argv, vw_option);
            if(!FLAGS_cache.empty())
                ML::MLUtility::writeTrainData(FLAGS_cache, traindata);
        }
        CHECK(traindata.get());
        cv::Ptr<ml::StatModel> classifier = run_train(traindata, vw_option);
        if(!classifier.get()){
            cerr << "Error occurs in training" << endl;
            return 1;
        }
        cv::Ptr<ml::TrainData> validationSet = ml::TrainData::loadFromCSV(FLAGS_validation, 0);
        if(validationSet.get()){
            double valid_acc = testClassifier(validationSet, classifier);
            printf("Validation accuracy: %.3f\n", valid_acc);
        }
    }else if(FLAGS_mode == "test"){
        run_test(argc, argv);
    }else if(FLAGS_mode == "detect"){
        run_detect(argc, argv, vw_option);
    }else if(FLAGS_mode == "grid"){
        cerr << "Grid search Not implemented" << endl;
        return 1;
    }else{
        cerr << "Unsupported mode" << endl;
        return 1;
    }
    return 0;
}

cv::Ptr<cv::ml::TrainData> run_extract(int argc, char** argv, const VisualWordOption& vw_option) {
    if (argc < 2) {
        cerr << "Usage: ./VisualWord <path-to-list>" << endl;
        return cv::Ptr<ml::TrainData>();
    }

    string list_path = string(argv[1]);
    ifstream listIn(list_path.c_str());
    string dir = list_path.substr(0, list_path.find_last_of("/"));
    dir.append("/");
    CHECK(listIn.is_open()) << "Can not open list file: " << argv[1];

    char buffer[256] = {};
    string filename, gtname;
    Mat descriptors;

    vector<vector<float> > segmentsFeature;
    vector<int> response;
    //descriptorMap: the id of descriptor contained by each segment
    vector<vector<int> > descriptorMap;

    cv::Ptr<cv::Feature2D> descriptorExtractor;

    if (vw_option.pixDesc == HOG3D)
        descriptorExtractor.reset(new CVHoG3D(vw_option.sigma_s, vw_option.sigma_r));
    else if (vw_option.pixDesc == COLOR3D)
        descriptorExtractor.reset(new CVColor3D(vw_option.sigma_s, vw_option.sigma_r));
    else
        CHECK(true) << "unsupported descriptor " << FLAGS_desc;

    while (listIn >> filename >> gtname) {
        vector<Mat> images;
        cv::VideoCapture cap(dir + filename);
        CHECK(cap.isOpened()) << "Can not open video: " << dir + filename;
        cout << "Loading " << dir + filename << endl;
        while (true) {
            Mat tmp;
            if (!cap.read(tmp))
                break;
            images.push_back(tmp);
        }
        printf("number of frames: %d\n", (int) images.size());
        vector<Mat> featureImage;
        descriptorExtractor.dynamicCast<CV3DDescriptor>()->prepareImage(images, featureImage);

        Mat gt = imread(dir + gtname, false);
        CHECK(gt.data) << "Can not read ground truth mask: " << dir + gtname;
        cv::resize(gt, gt, images[0].size(), INTER_NEAREST);

        vector<KeyPoint> keypoints;
        sampleKeyPoints(featureImage, keypoints, FLAGS_sigma_s, FLAGS_sigma_r);
        printf("Number of keypoints: %d\n", (int) keypoints.size());
        printf("Extracting descriptors...\n");

        const int descOffset = descriptors.rows;
        Mat curDescriptor;
        descriptorExtractor->compute(featureImage, keypoints, curDescriptor);
        CHECK(!curDescriptor.empty());
        if (descOffset == 0)
            descriptors = curDescriptor.clone();
        else
            cv::vconcat(descriptors, curDescriptor, descriptors);

        //load segment
        for (auto level: levelList) {
            Mat segments;
            video_segment::segment_video(images, segments, level);
            vector<ML::PixelGroup> pixelGroup;
            ML::regroupSegments(segments, pixelGroup);
            printf("Level %.2f, %d segments\n", level, (int) pixelGroup.size());
            vector<int> curResponse;
            printf("Assigning segment label...\n");
            ML::assignSegmentLabel(pixelGroup, gt, curResponse);
            response.insert(response.end(), curResponse.begin(), curResponse.end());
            printf("Extracting features...\n");
            vector<vector<float> > curSegFeatures;
            extractSegmentFeature(images, pixelGroup, curSegFeatures);
            segmentsFeature.insert(segmentsFeature.end(), curSegFeatures.begin(), curSegFeatures.end());
            //update descriptor map
            printf("Updating descriptor map...\n");
            vector<vector<int> > curDescMap(pixelGroup.size());
            for (auto i = 0; i < keypoints.size(); ++i) {
                const int sid = segments.at<int>(keypoints[i].pt);
                curDescMap[sid].push_back(i + descOffset);
            }
            descriptorMap.insert(descriptorMap.end(), curDescMap.begin(), curDescMap.end());
        }
    }

    //Sanity check
    CHECK_EQ(descriptorMap.size(), segmentsFeature.size());
    CHECK_EQ(descriptorMap.size(), response.size());
    CHECK(!segmentsFeature.empty());
    CHECK(!descriptors.empty());
    //construct visual word
    printf("Constructing visual words...\n");
    string path_codebook;

    if (!FLAGS_codebook.empty())
        path_codebook = FLAGS_codebook;
    else if (!FLAGS_model.empty()){
        path_codebook = FLAGS_model + "_codebook.txt";
    }
    Mat visualWord, bestLabel;
    cv::FileStorage codebookIn(path_codebook, cv::FileStorage::READ);

    if (!codebookIn.isOpened()) {
        //merge all descriptors into a mat
        printf("descriptors: %d, %d\n", descriptors.rows, descriptors.cols);
        cv::kmeans(descriptors, FLAGS_kCluster, bestLabel, cv::TermCriteria(cv::TermCriteria::COUNT, 30, 1.0), 3, KMEANS_PP_CENTERS, visualWord);
        if (!path_codebook.empty()) {
            cv::FileStorage codebookOut(path_codebook, cv::FileStorage::WRITE);
            codebookOut << "classifier" << vw_option.classifierType;
            codebookOut << "pixeldesc" << vw_option.pixDesc;
            codebookOut << "codebook" << visualWord;
        }
    }else{
        int pixdesc, classifiertype;
        codebookIn["classifier"] >> classifiertype;
        codebookIn["pixeldesc"] >> pixdesc;
        CHECK_EQ(pixdesc, (int)vw_option.pixDesc);
        CHECK_EQ(classifiertype, (int)vw_option.classifierType);

        codebookIn["codebook"] >> visualWord;
        CHECK_EQ(visualWord.cols, descriptors.cols);
    }

    const int kChannel = (int) segmentsFeature[0].size() + visualWord.rows;
    const int kSample = (int) segmentsFeature.size();

    Mat featureMat(kSample, kChannel, CV_32FC1, Scalar::all(0)), responseMat(kSample, 1, CV_32SC1);
    for (auto i = 0; i < responseMat.rows; ++i)
        responseMat.at<int>(i, 0) = response[i];

    //assign each descriptor to a cluster sample
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
    vector<DMatch> matches;
    matcher->match(descriptors, visualWord, matches);

    const int regionOffset = visualWord.rows;

    printf("Compose features...\n");
    for (auto i = 0; i < kSample; ++i) {
        //histogram of visual words
        vector<float> vwHist((size_t) visualWord.rows, 0.0f);
        for (auto j = 0; j < descriptorMap[i].size(); ++j) {
            const int did = descriptorMap[i][j];
            CHECK_LT(matches[did].trainIdx, vwHist.size());
            vwHist[matches[did].trainIdx] += 1.0f;
        }
        ML::MLUtility::normalizel1(vwHist);
        for (auto j = 0; j < vwHist.size(); ++j)
            featureMat.at<float>(i, j) = vwHist[j];
        for (auto j = 0; j < segmentsFeature[i].size(); ++j)
            featureMat.at<float>(i, j + regionOffset) = segmentsFeature[i][j];
    }
    cv::Ptr<cv::ml::TrainData> traindata = ml::TrainData::create(featureMat, ml::ROW_SAMPLE, responseMat);
    return traindata;
}

cv::Ptr<cv::ml::StatModel> run_train(cv::Ptr<ml::TrainData> traindata, const VisualWordOption& vw_option) {
    CHECK(traindata.get());
    cv::Ptr<ml::StatModel> classifier;
    string classifier_path = FLAGS_model;
    if(classifier_path.empty()){
        classifier_path =  "model";
    }
    if(vw_option.classifierType == RANDOM_FOREST){
        classifier_path.append(".rf");
        classifier = ml::RTrees::create();
        if(FLAGS_treeDepth > 0)
            classifier.dynamicCast<ml::RTrees>()->setMaxDepth(FLAGS_treeDepth);
        if(FLAGS_numTree > 0)
            classifier.dynamicCast<ml::RTrees>()->
                    setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER, FLAGS_numTree, std::numeric_limits<double>::min()));
    }else if(vw_option.classifierType == BOOSTED_TREE) {
        classifier_path.append(".bt");
        classifier = ml::Boost::create();
        if(FLAGS_treeDepth > 0)
            classifier.dynamicCast<ml::DTrees>()->setMaxDepth(FLAGS_treeDepth);
        if (FLAGS_numTree > 0)
            classifier.dynamicCast<ml::Boost>()->setWeakCount(FLAGS_numTree);
        printf("Number of trees: %d\n", classifier.dynamicCast<ml::Boost>()->getWeakCount());
    }else if(vw_option.classifierType == SVM) {
        classifier_path.append(".svm");
        classifier = ml::SVM::create();
        cv::Ptr<ml::SVM> svm = classifier.dynamicCast<ml::SVM>();
        svm->setC(FLAGS_svmC);
        svm->setGamma(FLAGS_svmGamma);
        if(FLAGS_svmKernel == "chi2")
            svm->setKernel(ml::SVM::CHI2);
        else if(FLAGS_svmKernel == "rbf")
            svm->setKernel(ml::SVM::RBF);
        else
            CHECK(true) << "Unsupported svm kernel type: " << FLAGS_svmKernel;
    }
    printf("Training...\n");

    int kPos = 0, kNeg = 0;
    Mat response = traindata->getResponses();

    for(auto i=0; i<response.rows; ++i) {
        float res = 0.0;
        if(response.depth() == CV_32F)
            res = response.at<float>(i, 0);
        else if(response.depth() == CV_32S)
            res = (float)response.at<int>(i, 0);
        if (res > 0.5)
            kPos++;
        else
            kNeg++;
    }
    printf("%d samples in total: %d potive, %d negative\n", kPos+kNeg, kPos, kNeg);
    
    CHECK_NOTNULL(classifier.get())->train(traindata);

    double acc = testClassifier(traindata, classifier);
    printf("Saving %s\n", classifier_path.c_str());
    classifier.get()->save(classifier_path);
    printf("Training accuracy: %.3f\n", acc);
    return classifier;
}

void run_test(int argc, char** argv){

}
void run_detect(int argc, char** argv, const VisualWordOption& vw_option){
    if(argc < 3){
        CHECK(true) <<  "Usage: ./VisualWord --mode=detect <path-to-video> <path-to-segment>";
    }

    char buffer[256] = {};

    cv::Ptr<ml::StatModel> classifier;
    if(vw_option.classifierType == RANDOM_FOREST){
        classifier = ml::RTrees::load<ml::RTrees>(FLAGS_model);
        CHECK(classifier.get()) << "Can not load classifier: " << FLAGS_model;
        printf("random forest, max depth: %d\n",
               classifier.dynamicCast<ml::RTrees>()->getMaxDepth());
    }else if(vw_option.classifierType == BOOSTED_TREE){
        classifier = ml::Boost::load<ml::Boost>(FLAGS_model);
    }else if(vw_option.classifierType == SVM) {
        classifier = ml::SVM::load<ml::SVM>(FLAGS_model);
    }

    printf("Reading video...\n");
    cv::VideoCapture cap(argv[1]);
    CHECK(cap.isOpened()) << "Can not open video: " << argv[1];
    vector<Mat> images;
    while(true){
        Mat frame;
        if(!cap.read(frame))
            break;
        images.push_back(frame);
    }
    CHECK(!images.empty());
    Mat refImage = images[0].clone();
    Mat codebook;
    string model_name = FLAGS_model.substr(0, FLAGS_model.find_last_of("."));
    string path_codebook;
    if(FLAGS_codebook.empty()){
        path_codebook = model_name + "_codebook.txt";
    } else
        path_codebook = FLAGS_codebook;

    cv::FileStorage codebookIn(path_codebook, FileStorage::READ);
    CHECK(codebookIn.isOpened()) << "Can not load code book: " << path_codebook;
    codebookIn["codebook"] >> codebook;

    Mat detection;
    detectVideo(images, classifier, codebook, levelList, detection, vw_option);

    Mat mask(detection.size(), CV_8UC3, Scalar(255,0,0));
    for (auto y = 0; y < mask.rows; ++y) {
        for (auto x = 0; x < mask.cols; ++x) {
            if (detection.at<uchar>(y, x) > (uchar)200)
                mask.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
        }
    }
    const double blend_weight = 0.4;
    Mat vis;
    cv::addWeighted(refImage, blend_weight, mask, 1.0 - blend_weight, 0.0, vis);

    if(argc >= 3){
        imwrite(argv[2], vis);
    }else {
        string fullPath = string(argv[1]);
        string filename = fullPath.substr(0, fullPath.find_last_of('.'));
        imwrite(filename + "_result.png", vis);
    }
}

void run_multiExtract(const vector<int>& kClusters, int argc, char** argvm, const VisualWordOption& vw_option) {
//    if (argc < 2) {
//        cerr << "Usage: ./VisualWord <path-to-list>" << endl;
//        return;
//    }
//    CHECK(!(FLAGS_model.empty() && FLAGS_codebook.empty()));
//
//    string list_path = string(argv[1]);
//    ifstream listIn(list_path.c_str());
//    string dir = list_path.substr(0, list_path.find_last_of("/"));
//    dir.append("/");
//    CHECK(listIn.is_open()) << "Can not open list file: " << argv[1];
//
//    char buffer[256] = {};
//    string filename, gtname;
//    Mat descriptors;
//
//    vector<vector<float> > segmentsFeature;
//    vector<int> response;
//    //descriptorMap: the id of descriptor contained by each segment
//    vector<vector<int> > descriptorMap;
//
//    cv::Ptr<cv::Feature2D> descriptorExtractor;
//    if (FLAGS_desc == "hog3d")
//        descriptorExtractor.reset(new CVHoG3D(FLAGS_sigma_s, FLAGS_sigma_r));
//    else if (FLAGS_desc == "color3d")
//        descriptorExtractor.reset(new CVColor3D(FLAGS_sigma_s, FLAGS_sigma_r));
//    else
//        CHECK(true) << "unsupported descriptor " << FLAGS_desc;
//
//    while (listIn >> filename >> gtname) {
//        vector<Mat> images;
//        cv::VideoCapture cap(dir + filename);
//        CHECK(cap.isOpened()) << "Can not open video: " << dir + filename;
//        cout << "Loading " << dir + filename << endl;
//        while (true) {
//            Mat tmp;
//            if (!cap.read(tmp))
//                break;
//            images.push_back(tmp);
//        }
//        printf("number of frames: %d\n", (int) images.size());
//        vector<Mat> featureImage;
//        descriptorExtractor.dynamicCast<CV3DDescriptor>()->prepareImage(images, featureImage);
//
//        Mat gt = imread(dir + gtname, false);
//        CHECK(gt.data) << "Can not read ground truth mask: " << dir + gtname;
//        cv::resize(gt, gt, images[0].size(), INTER_NEAREST);
//
//        vector<KeyPoint> keypoints;
//        sampleKeyPoints(featureImage, keypoints, FLAGS_sigma_s, FLAGS_sigma_r);
//        printf("Number of keypoints: %d\n", (int) keypoints.size());
//        printf("Extracting descriptors...\n");
//
//        const int descOffset = descriptors.rows;
//        Mat curDescriptor;
//        descriptorExtractor->compute(featureImage, keypoints, curDescriptor);
//        CHECK(!curDescriptor.empty());
//        if (descOffset == 0)
//            descriptors = curDescriptor.clone();
//        else
//            cv::vconcat(descriptors, curDescriptor, descriptors);
//
//        //load segment
//        printf("Loading segments...\n");
//        for (auto level: levelList) {
//            vector<Mat> segments;
//            sprintf(buffer, "%s/segmentation/%s.pb", dir.c_str(), filename.c_str());
//            segmentation::readSegmentAsMat(string(buffer), segments, level);
//            Feature::compressSegments(segments);
//            vector<vector<vector<int> > > pixelGroup;
//            vector<vector<int> > segmentRegion;
//            Feature::regroupSegments(segments, pixelGroup, segmentRegion);
//            printf("Level %.2f, %d segments\n", level, (int) pixelGroup.size());
//            vector<int> curResponse;
//            printf("Assigning segment label...\n");
//            Feature::assignSegmentLabel(pixelGroup, gt, curResponse);
//            response.insert(response.end(), curResponse.begin(), curResponse.end());
//            printf("Extracting features...\n");
//            for (const auto &pg: pixelGroup) {
//                vector<float> curRegionFeat;
//                vector<float> color, shape, position;
//                Feature::computeColor(images, pg, color);
//                Feature::computeShapeAndLength(pg, images[0].cols, images[0].rows, shape);
//                Feature::computePosition(pg, images[0].cols, images[0].rows, position);
//                curRegionFeat.insert(curRegionFeat.end(), color.begin(), color.end());
//                curRegionFeat.insert(curRegionFeat.end(), shape.begin(), shape.end());
//                curRegionFeat.insert(curRegionFeat.end(), position.begin(), position.end());
//                segmentsFeature.push_back(curRegionFeat);
//            }
//
//            //update descriptor map
//            printf("Updating descriptor map...\n");
//            vector<vector<int> > curDescMap(pixelGroup.size());
//            for (auto i = 0; i < keypoints.size(); ++i) {
//                const int sid = segments[keypoints[i].octave].at<int>(keypoints[i].pt);
//                curDescMap[sid].push_back(i + descOffset);
//            }
//            descriptorMap.insert(descriptorMap.end(), curDescMap.begin(), curDescMap.end());
//        }
//    }
//
//    //Sanity check
//    CHECK_EQ(descriptorMap.size(), segmentsFeature.size());
//    CHECK_EQ(descriptorMap.size(), response.size());
//    CHECK(!segmentsFeature.empty());
//    CHECK(!descriptors.empty());
//    //construct visual word
//    for(auto cluster: kClusters) {
//        printf("Constructing visual words, kcluster: %d...\n", cluster);
//        string path_codebook;
//        sprintf(buffer, "%s_cluster%05d_codebook.txt", FLAGS_model.c_str(), cluster);
//        path_codebook = string(buffer);
//
//        Mat visualWord, bestLabel;
//        if(!loadCodebook(path_codebook, visualWord)) {
//            printf("descriptors: %d, %d\n", descriptors.rows, descriptors.cols);
//            cv::kmeans(descriptors, cluster, bestLabel, cv::TermCriteria(cv::TermCriteria::COUNT, 30, 1.0), 3, KMEANS_PP_CENTERS, visualWord);
//            writeCodebook(path_codebook, visualWord);
//        }
//        const int kChannel = (int) segmentsFeature[0].size() + visualWord.rows;
//        const int kSample = (int) segmentsFeature.size();
//
//        Mat featureMat(kSample, kChannel, CV_32FC1, Scalar::all(0)), responseMat(kSample, 1, CV_32SC1);
//        for (auto i = 0; i < responseMat.rows; ++i)
//            responseMat.at<int>(i, 0) = response[i];
//
//        //assign each descriptor to a cluster sample
//        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
//        vector<DMatch> matches;
//        matcher->match(descriptors, visualWord, matches);
//
//        const int regionOffset = visualWord.rows;
//
//        printf("Compose features...\n");
//        for (auto i = 0; i < kSample; ++i) {
//            //histogram of visual words
//            vector<float> vwHist((size_t) visualWord.rows, 0.0f);
//            for (auto j = 0; j < descriptorMap[i].size(); ++j) {
//                const int did = descriptorMap[i][j];
//                CHECK_LT(matches[did].trainIdx, vwHist.size());
//                vwHist[matches[did].trainIdx] += 1.0f;
//            }
//            Feature::normalizeSum(vwHist);
//            for (auto j = 0; j < vwHist.size(); ++j)
//                featureMat.at<float>(i, j) = vwHist[j];
//            for (auto j = 0; j < segmentsFeature[i].size(); ++j)
//                featureMat.at<float>(i, j + regionOffset) = segmentsFeature[i][j];
//        }
//        cv::Ptr<cv::ml::TrainData> traindata = ml::TrainData::create(featureMat, ml::ROW_SAMPLE, responseMat);
//        sprintf(buffer, "%s_cluster%05d.csv", FLAGS_cache.c_str(), cluster);
//        writeTrainData(string(buffer), traindata);
//    }
 }

VisualWordOption setOption(){
    VisualWordOption option;
    option.sigma_r = FLAGS_sigma_r;
    option.sigma_s = FLAGS_sigma_s;
    if(FLAGS_desc == "color3d"){
        option.pixDesc = COLOR3D;
    }else if(FLAGS_desc == "hog3d"){
        option.pixDesc = HOG3D;
    }else{
        CHECK(true) << "Unsuppored pixel descriptor: " << FLAGS_desc;
    }

    if(FLAGS_classifier == "rf"){
        option.classifierType = RANDOM_FOREST;
    }else if(FLAGS_classifier == "bt"){
        option.classifierType = BOOSTED_TREE;
    }else if(FLAGS_classifier == "svm"){
        option.classifierType = SVM;
    }else{
        CHECK(true) << "Unsuppored classifier: " << FLAGS_classifier;
    }

    return option;
}