//
// Created by yanhang on 6/1/16.
//

#include <gflags/gflags.h>
#include <external/segment_gb/segment-image.h>

#include "dynamicsegment.h"
#include "dynamicregularizer.h"
#include "../GeometryModule/dynamicwarpping.h"
#include "../Cinemagraph/cinemagraph.h"
#include "stabilization.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace dynamic_stereo;

DEFINE_int32(testFrame, 60, "anchor frame");
DEFINE_int32(resolution, 128, "resolution");
DEFINE_int32(tWindow, 100, "tWindow");
DEFINE_int32(kFrames, 300, "Number of frames in cinemagraph");

DEFINE_int32(downsample, 2, "downsample ratio");
DEFINE_string(classifierPath, "/home/yanhang/Documents/research/DynamicStereo/data/traindata/visualword/model_new.rf", "Path to classifier");
DEFINE_string(codebookPath, "/home/yanhang/Documents/research/DynamicStereo/data/traindata/visualword/metainfo_new_cluster00050.yml", "path to codebook");

//DEFINE_string(classifierPath, "/home/yanhang/Documents/research/DynamicStereo/code/build/VisualWord/model_hier2.rf", "Path to classifier");
//DEFINE_string(codebookPath, "/home/yanhang/Documents/research/DynamicStereo/code/build/VisualWord/metainfo_hier_cluster00050.yml", "path to codebook");

DEFINE_string(regularization, "RPCA", "algorithm for regularization, {median, RPCA, poisson, anisotropic, none}");

DEFINE_double(param_stab, 2.0, "parameter for geometric stabilization");
DECLARE_string(flagfile);

int main(int argc, char** argv) {
    char buffer[1024] = {};
    if(argc < 2){
        cerr << "Usage: ./SegmentAndRender <path-to-data>" << endl;

        return 1;
    }
    FileIO file_io(argv[1]);
    CHECK_GT(file_io.getTotalNum(), 0) << "Empty dataset";

    google::InitGoogleLogging(argv[1]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    LOG(INFO) << "testFrame:" << FLAGS_testFrame <<  " tWindow:" << FLAGS_tWindow;

    const double depthSmooth = 0.1;
    std::shared_ptr<DynamicWarpping> warping(new DynamicWarpping(file_io, FLAGS_testFrame, FLAGS_tWindow, FLAGS_resolution, depthSmooth));
    const int anchor_frame = FLAGS_testFrame - warping->getOffset();

    LOG(INFO) << "Warping..";
    vector <Mat> mid_input, mid_output, visMaps;
    warping->preWarping(mid_input, true, &visMaps);
    const cv::Size kFrameSize(mid_input[0].cols, mid_input[0].rows);

    sprintf(buffer, "%s/temp/warped%05d.avi", file_io.getDirectory().c_str(), FLAGS_testFrame);
    VideoWriter warpOutput;
    warpOutput.open(string(buffer), CV_FOURCC('x','2','6','4'), 30, kFrameSize);
    CHECK(warpOutput.isOpened()) << "Can not open video stream";

    for (auto i = 0; i < mid_input.size(); ++i) {
        warpOutput << mid_input[i];
    }
    warpOutput.release();

    ////////////////////////////////////////////
    Cinemagraph::Cinemagraph cinemagraph;
    cinemagraph.reference = FLAGS_testFrame;

    //Segmentation
    LOG(INFO) <<"Segmenting...";
    Mat seg_result_display(kFrameSize, CV_32SC1, Scalar::all(0)), seg_result_flashy(kFrameSize, CV_32SC1, Scalar::all(0));
    LOG(INFO) << "Segmenting display...";
    segmentDisplay(file_io, FLAGS_testFrame, mid_input, FLAGS_classifierPath, FLAGS_codebookPath ,seg_result_display);
//    LOG(INFO) << "Segmenting flashy...";
//    segmentFlashy(file_io, FLAGS_testFrame, mid_input, cinemagraph.pixel_loc_flashy, cinemagraph.ranges_flashy);

    CHECK_EQ(seg_result_display.cols, kFrameSize.width);
    CHECK_EQ(seg_result_display.rows, kFrameSize.height);
    CHECK_EQ(seg_result_flashy.cols, kFrameSize.width);
    CHECK_EQ(seg_result_flashy.rows, kFrameSize.height);

    //////////////////////////////////////////////////////////

    //Rendering
    groupPixel(seg_result_display, cinemagraph.pixel_loc_display);
    getSegmentRange(visMaps, cinemagraph.pixel_loc_display, cinemagraph.ranges_display);
    //discard segments with too small ranges
    const int minFrame = static_cast<int>((double)mid_input.size() * 0.2);
    filterShortSegments(cinemagraph.pixel_loc_display, cinemagraph.ranges_display, minFrame);
    filterShortSegments(cinemagraph.pixel_loc_flashy, cinemagraph.ranges_flashy, minFrame);

    vector<vector<Vector2i> > segments_all = cinemagraph.pixel_loc_display;
    segments_all.insert(segments_all.end(), cinemagraph.pixel_loc_flashy.begin(), cinemagraph.pixel_loc_flashy.end());

    //three step regularization:
    //1. Apply a small poisson smoothing, fill in holes
    //2. Geometric stablization by grid warping
    //3. Apply RPCA to smooth transition and remove high frequency noise

    LOG(INFO) << "Step 1: Fill holes by poisson smoothing";
    const double small_poisson = 0.01;
    regularizationPoisson(mid_input, cinemagraph.pixel_loc_display, mid_output, small_poisson, small_poisson);
    mid_input.swap(mid_output);
    mid_output.clear();

    //The flashy segments will not pass stabilization and regularization, so create the pixel mat now
    cinemagraph.pixel_mat_flashy.resize(cinemagraph.pixel_loc_flashy.size());
    for(auto i=0; i<cinemagraph.pixel_loc_flashy.size(); ++i){
        Cinemagraph::CreatePixelMat(mid_input, cinemagraph.pixel_loc_flashy[i], cinemagraph.ranges_flashy[i], cinemagraph.pixel_mat_flashy[i]);
    }

    LOG(INFO) << "Step 2: geometric stablization";
    float stab_t = (float)cv::getTickCount();
    stabilizeSegments(mid_input, mid_output, cinemagraph.pixel_loc_display, cinemagraph.ranges_display, anchor_frame, FLAGS_param_stab, StabAlg::HOMOGRAPHY);
    LOG(INFO) << "Done. Time usage: " << ((float)getTickCount() - stab_t) / (float)getTickFrequency() << "s";
    mid_input.swap(mid_output);
    mid_output.clear();

    sprintf(buffer, "%s/temp/stabilized%05d.avi", file_io.getDirectory().c_str(), FLAGS_testFrame);
    VideoWriter stabilizedOutput(string(buffer), CV_FOURCC('x','2','6','4'), 30, kFrameSize);
    CHECK(stabilizedOutput.isOpened()) << buffer;
    for (auto i = 0; i < mid_input.size(); ++i) {
        stabilizedOutput << mid_input[i];
    }
    stabilizedOutput.release();

    cinemagraph.background = imread(file_io.getImage(FLAGS_testFrame));

    cinemagraph.pixel_mat_display.resize(cinemagraph.pixel_loc_display.size());
    for(auto i=0; i<cinemagraph.pixel_loc_display.size(); ++i){
        Cinemagraph::CreatePixelMat(mid_input, cinemagraph.pixel_loc_display[i], cinemagraph.ranges_display[i], cinemagraph.pixel_mat_display[i]);
    }
    vector<Mat> cinemagraph_unregulared;
    Cinemagraph::RenderCinemagraph(cinemagraph, cinemagraph_unregulared, FLAGS_kFrames);
    sprintf(buffer, "%s/temp/unregulared%05d.avi", file_io.getDirectory().c_str(), FLAGS_testFrame);
    VideoWriter unregulared_vw(string(buffer), CV_FOURCC('x','2','6','4'), 30, kFrameSize);
    CHECK(unregulared_vw.isOpened());
    for(const auto& img: cinemagraph_unregulared){
        unregulared_vw << img;
    }
    unregulared_vw.release();

    LOG(INFO) << "Step 3: Color regularization";
    float reg_t = (float)cv::getTickCount();
    if(FLAGS_regularization == "median"){
        const int medianR = 10;
        printf("Running regularization with median filter, r: %d\n", medianR);
        temporalMedianFilter(mid_input, cinemagraph.pixel_loc_display, mid_output, medianR);
    }else if(FLAGS_regularization == "RPCA"){
        //use adaptive weighting
        vector<float> adaptive_lambdas(cinemagraph.pixel_loc_display.size(), 0.005);
//        for(auto i=0; i<cinemagraph.pixel_mat_display.size(); ++i){
//            adaptive_lambdas[i] = GetRPCAWeight(cinemagraph.pixel_mat_display[i]);
//        }

        printf("Running regularizaion with RPCA with adaptive lambda\n");
        regularizationRPCA(mid_input, cinemagraph.pixel_loc_display, adaptive_lambdas, mid_output);
    }else if(FLAGS_regularization == "anisotropic"){
        const double ws = 10;
        printf("Running regularization with anisotropic diffusion, ws: %.5f\n", ws);
        regularizationAnisotropic(mid_input, cinemagraph.pixel_loc_display, mid_output, ws);
    }else if(FLAGS_regularization == "poisson"){
        const double ws = 0.1, wt = 0.5;
        printf("Running regularization with poisson smoothing, ws: %.3f, wt: %.5f\n", ws, wt);
        regularizationPoisson(mid_input, cinemagraph.pixel_loc_display, mid_output, ws, wt);
    }else if(FLAGS_regularization == "none"){
        mid_output = mid_input;
        printf("No regularization\n");
    }else{
        cerr << "Invalid regularization algorithm. Choose between {median, RPCA, anisotropic, poisson}" << endl;
        return 1;
    }
    printf("Done, time usage: %.2fs\n", ((float)cv::getTickCount() -reg_t)/(float)cv::getTickFrequency());
    mid_input.swap(mid_output);
    mid_output.clear();

    //create pixel mat for display
    cinemagraph.pixel_mat_display.resize(cinemagraph.pixel_loc_display.size());
    for(auto i=0; i<cinemagraph.pixel_loc_display.size(); ++i){
        Cinemagraph::CreatePixelMat(mid_input, cinemagraph.pixel_loc_display[i],
                                    cinemagraph.ranges_display[i], cinemagraph.pixel_mat_display[i]);
    }

    //release unused memory
    mid_input.clear();
    warping.reset();

    //final rendering
    CHECK(cinemagraph.background.data);
    //compute blending weight
    constexpr int blend_R = 5;
    constexpr int min_segment_size = 50;
//    Cinemagraph::ComputeBlendMap(cinemagraph.pixel_loc_display, cinemagraph.background.cols, cinemagraph.background.rows,
//                                 blend_R, cinemagraph.blend_map);
    Cinemagraph::ComputeBlendMap(segments_all, cinemagraph.background.cols, cinemagraph.background.rows,
                                 blend_R, min_segment_size, cinemagraph.blend_map);

    vector<Mat> rendered;
    LOG(INFO) << "Rendering cinemagraph";
    Cinemagraph::RenderCinemagraph(cinemagraph, rendered, FLAGS_kFrames);
    sprintf(buffer, "%s/temp/finalResult_%s_%05d.avi", file_io.getDirectory().c_str(), FLAGS_regularization.c_str(), FLAGS_testFrame);
    VideoWriter resultWriter(string(buffer), CV_FOURCC('x','2','6','4'), 30, kFrameSize);
    CHECK(resultWriter.isOpened()) << buffer;
    for (auto i = 0; i < rendered.size(); ++i) {
        resultWriter << rendered[i];
    }
    resultWriter.release();

    LOG(INFO) << "Writing cenimagraph to file";
    sprintf(buffer, "%s/temp/cinemagraph_%05d_%s.cg", file_io.getDirectory().c_str(), FLAGS_testFrame, FLAGS_regularization.c_str());
    Cinemagraph::SaveCinemagraph(string(buffer), cinemagraph);

    return 0;
}
