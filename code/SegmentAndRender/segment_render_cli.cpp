//
// Created by yanhang on 6/1/16.
//
#include "dynamicsegment.h"
#include <gflags/gflags.h>
#include "dynamicregularizer.h"
#include "../common/dynamicwarpping.h"
#include "stabilization.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace dynamic_stereo;

DEFINE_int32(testFrame, 60, "anchor frame");
DEFINE_int32(resolution, 128, "resolution");
DEFINE_int32(tWindow, 100, "tWindow");
DEFINE_int32(downsample, 2, "downsample ratio");
DEFINE_string(classifierPath, "/home/yanhang/Documents/research/DynamicStereo/data/traindata/visualword/model_new.rf", "Path to classifier");
DEFINE_string(codebookPath, "/home/yanhang/Documents/research/DynamicStereo/data/traindata/visualword/metainfo_new_cluster00050.yml", "path to codebook");
DEFINE_string(regularization, "RPCA", "algorithm for regularization, {median, RPCA, poisson, anisotropic}");

DEFINE_double(param_stab, 0.01, "parameter for geometric stabilization");
DECLARE_string(flagfile);

void loadData(const FileIO& file_io, vector<Mat>& images, Mat& segMask, Depth& refDepth);

int main(int argc, char** argv) {
    char buffer[1024] = {};
    if(argc < 2){
        cerr << "Usage: ./SegmentAndRender <path-to-data>" << endl;
        return 1;
    }
    FileIO file_io(argv[1]);
    CHECK_GT(file_io.getTotalNum(), 0) << "Empty dataset";

    sprintf(buffer, "%s/config.txt", file_io.getDirectory().c_str());
    ifstream flagfile(buffer);
    if(flagfile.is_open()) {
        printf("Read flag from file\n");
        FLAGS_flagfile = string(buffer);
    }

    google::InitGoogleLogging(argv[1]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    printf("testFrame:%d, tWindow:%d\n", FLAGS_testFrame, FLAGS_tWindow);

    vector<Mat> images;
    Mat segMask;
    Depth refDepth;
    int width, height;
    printf("Loading...\n");
    loadData(file_io, images, segMask, refDepth);
    width = images[0].cols;
    height = images[0].rows;
    Mat refImage = imread(file_io.getImage(FLAGS_testFrame));
    cv::resize(refImage, refImage, cv::Size(width, height), cv::INTER_CUBIC);

    ////////////////////////////////////////////
    //Segmentation
    printf("Segmenting...\n");

    Mat seg_result_display, seg_result_flashy;
    LOG(INFO) << "Segmenting display...";
    segmentDisplay(file_io, FLAGS_testFrame, images, segMask, FLAGS_classifierPath, FLAGS_codebookPath ,seg_result_display);
//    LOG(INFO) << "Segmenting flashy...";
//    segmentFlashy(file_io, FLAGS_testFrame, images, seg_result_flashy);



    //////////////////////////////////////////////////////////
    //Rendering
    //reload full resolution image, set black pixel to (1,1,1)
    const double depthSmooth = 2.0;
    std::shared_ptr<DynamicWarpping> warping(new DynamicWarpping(file_io, FLAGS_testFrame, FLAGS_tWindow, FLAGS_resolution, depthSmooth));

    int offset = CHECK_NOTNULL(warping.get())->getOffset();
    int anchor_frame = FLAGS_testFrame - offset;
    images.resize((size_t)FLAGS_tWindow);
    for(auto v=0; v<FLAGS_tWindow; ++v){
        images[v] = imread(file_io.getImage(offset+v));
        for(auto y=0; y<images[v].rows; ++y){
            for(auto x=0; x<images[v].cols; ++x){
                if(images[v].at<Vec3b>(y,x) == Vec3b(0,0,0))
                    images[v].at<Vec3b>(y,x) = Vec3b(1,1,1);
            }
        }
    }
    cv::resize(seg_result_display, seg_result_display, images[0].size(), 0, 0, INTER_NEAREST);
    //cv::resize(seg_result_flashy, seg_result_flashy, images[0].size(), 0, 0, INTER_NEAREST);

    vector<vector<Vector2i> > segmentsDisplay;
    vector<vector<Vector2i> > segmentsFlashy;
    groupPixel(seg_result_display, segmentsDisplay);
    //groupPixel(seg_result_flashy, segmentsFlashy);

    vector <Mat> mid_input, mid_output, visMaps;
    LOG(INFO) << "Full warping..";
    //warping->warpToAnchor(images, segmentsDisplay, segmentsFlashy, finalResult, FLAGS_tWindow);
    warping->preWarping(mid_input, true, &visMaps);

    cv::Size frameSize(mid_input[0].cols, mid_input[0].rows);

    sprintf(buffer, "%s/temp/warped%05d.avi", file_io.getDirectory().c_str(), FLAGS_testFrame);
    VideoWriter warpOutput;
    warpOutput.open(string(buffer), CV_FOURCC('x','2','6','4'), 30, frameSize);
    CHECK(warpOutput.isOpened()) << "Can not open video stream";

    for (auto i = 0; i < mid_input.size(); ++i) {
        warpOutput << mid_input[i];
    }
    warpOutput.release();

    vector<Vector2i> rangesDisplay, rangesFlashy;
    getSegmentRange(visMaps, segmentsDisplay, rangesDisplay);
    //getSegmentRange(visMaps, segmentsFlashy, rangesFlashy);

    //discard segments with too small ranges
    const int minFrame = static_cast<int>(mid_input.size() * 0.2);
    filterShortSegments(segmentsDisplay, rangesDisplay, minFrame);
    //filterShortSegments(segmentsFlashy, rangesFlashy, minFrame);


    //three step regularization:
    //1. Apply a small poisson smoothing, fill in holes
    //2. Geometric stablization by grid warping
    //3. Apply RPCA to smooth transition and remove high frequency noise


//   printf("Step 1: Fill holes by poisson smoothing\n");
//   const double small_poisson = 0.01;
//   regularizationPoisson(mid_input, segmentsDisplay, mid_output, small_poisson, small_poisson);
//   mid_input.swap(mid_output);
//   mid_output.clear();
//
     printf("Step 2: geometric stablization\n");
     float stab_t = (float)cv::getTickCount();
     //vector<Mat> debug_input(mid_input.begin(), mid_input.begin() + 20);
     stabilizeSegments(mid_input, mid_output, segmentsDisplay, rangesDisplay, anchor_frame, FLAGS_param_stab, StabAlg::TRACK);
     printf("Done. Time usage: %.3fs\n", ((float)getTickCount() - stab_t) / (float)getTickFrequency());
     mid_input.swap(mid_output);
     mid_output.clear();

    //Now apply mask
    segmentsDisplay.insert(segmentsDisplay.end(), segmentsFlashy.begin(), segmentsFlashy.end());
    rangesDisplay.insert(rangesDisplay.end(), rangesFlashy.begin(), rangesFlashy.end());
    renderToMask(mid_input, segmentsDisplay, rangesDisplay, mid_output);
    mid_input.swap(mid_output);
    mid_output.clear();

    sprintf(buffer, "%s/temp/stabilized%05d.avi", file_io.getDirectory().c_str(), FLAGS_testFrame);
    VideoWriter stabilizedOutput(string(buffer), CV_FOURCC('x','2','6','4'), 30, frameSize);
    CHECK(stabilizedOutput.isOpened()) << buffer;
    for (auto i = 0; i < mid_input.size(); ++i) {
        stabilizedOutput << mid_input[i];
    }
    stabilizedOutput.release();

//    printf("Step 3: Color regularization\n");
//    float reg_t = (float)cv::getTickCount();
//    if(FLAGS_regularization == "median"){
//        const int medianR = 5;
//        printf("Running regularization with median filter, r: %d\n", medianR);
//        temporalMedianFilter(mid_input, segmentsDisplay, mid_output, medianR);
//    }else if(FLAGS_regularization == "RPCA"){
//        const double regular_lambda = 0.015;
//        printf("Running regularizaion with RPCA, lambda: %.3f\n", regular_lambda);
//        regularizationRPCA(mid_input, segmentsDisplay, mid_output, regular_lambda);
//    }else if(FLAGS_regularization == "anisotropic"){
//        const double ws = 0.6;
//        printf("Running regularization with anisotropic diffusion, ws: %.3f\n", ws);
//        regularizationAnisotropic(mid_input, segmentsDisplay, mid_output, ws);
//    }else if(FLAGS_regularization == "poisson"){
//        const double ws = 0.1, wt = 0.5;
//        printf("Running regularization with poisson smoothing, ws: %.3f, wt: %.3f\n", ws, wt);
//        regularizationPoisson(mid_input, segmentsDisplay, mid_output, ws, wt);
//    }else{
//        cerr << "Invalid regularization algorithm. Choose between {median, RPCA, anisotropic, poisson}" << endl;
//        return 1;
//    }
//    printf("Done, time usage: %.2fs\n", ((float)cv::getTickCount() -reg_t)/(float)cv::getTickFrequency());
//    mid_input.swap(mid_output);
//    mid_output.clear();
//
//    sprintf(buffer, "%s/temp/finalReault_%05d.avi", file_io.getDirectory().c_str(), FLAGS_testFrame);
//    VideoWriter resultWriter(string(buffer), CV_FOURCC('x','2','6','4'), 30, frameSize);
//    CHECK(resultWriter.isOpened()) << buffer;
//    for (auto i = 0; i < mid_input.size(); ++i) {
////        cv::putText(finalResult[i], FLAGS_regularization, cv::Point(20,50), FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0,0,255), 3);
//        resultWriter << mid_input[i];
//    }
//    resultWriter.release();

    return 0;
}

void loadData(const FileIO& file_io, vector<Mat>& images, Mat& segMask, Depth& refDepth){
    //images
    char buffer[1024] = {};
    images.resize((size_t)FLAGS_tWindow);
    int width, height;
    const int offset = FLAGS_testFrame - FLAGS_tWindow / 2;
    CHECK_GE(offset, 0);
    for(auto i=0; i<FLAGS_tWindow; ++i){
        sprintf(buffer, "%s/midres/prewarp/prewarpb%05d_%05d.jpg", file_io.getDirectory().c_str(), FLAGS_testFrame, i);
        images[i] = imread(buffer);
        CHECK(images[i].data) << buffer;
    }
    CHECK(!images.empty());
//	width = images[0].cols;
//	height = images[0].rows;
    //segnet mask
//	sprintf(buffer, "%s/segnet/seg%05d.png", file_io.getDirectory().c_str(), FLAGS_testFrame);
//	Mat segMaskImg = imread(buffer);
//	CHECK(segMaskImg.data) << buffer;
//	cv::resize(segMaskImg, segMaskImg, cv::Size(width, height), 0,0, INTER_NEAREST);
//	//vector<Vec3b> validColor{Vec3b(0,0,128), Vec3b(128,192,192), Vec3b(128,128,192), Vec3b(128,128,128), Vec3b(0,128,128)};
//	vector<Vec3b> invalidColor{Vec3b(128,0,64), Vec3b(128,64,128), Vec3b(0,64,64), Vec3b(222,40,60)};
//
//	segMask.create(height, width, CV_8UC1);
//	for(auto y=0; y<height; ++y){
//		for(auto x=0; x<width; ++x){
//			Vec3b pix = segMaskImg.at<Vec3b>(y,x);
//			if(std::find(invalidColor.begin(), invalidColor.end(), pix) < invalidColor.end())
//				segMask.at<uchar>(y,x) = 0;
//			else
//				segMask.at<uchar>(y,x) = 255;
//		}
//	}
    //depth
    sprintf(buffer, "%s/midres/depth%05d.depth", file_io.getDirectory().c_str(), FLAGS_testFrame);
    CHECK(refDepth.readDepthFromFile(string(buffer))) << "Can not read depth file";

}
