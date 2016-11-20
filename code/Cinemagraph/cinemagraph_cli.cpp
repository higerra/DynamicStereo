//
// Created by yanhang on 11/1/16.
//

#include "cinemagraph.h"
#include "cinemagraph_util.h"

#include <gflags/gflags.h>

#include "../base/file_io.h"
using namespace std;
using namespace cv;
using namespace dynamic_stereo;

DEFINE_int32(testFrame, 0, "test frame");
DEFINE_int32(kFrame, 200, "output frames");
DEFINE_string(alg, "RPCA", "algorithm for regularization");

int main(int argc, char** argv){
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    CHECK_GE(argc, 3);

    char buffer[128] = {};
    FileIO file_io(argv[1]);
    sprintf(buffer, "%s/temp/cinemagraph_%05d_%s.cg", file_io.getDirectory().c_str(), FLAGS_testFrame, FLAGS_alg.c_str());
    Cinemagraph::Cinemagraph cinemagraph;
    printf("Reading %s\n", buffer);
    Cinemagraph::ReadCinemagraph(string(buffer), cinemagraph);

    Mat ref_img = imread(file_io.getImage(FLAGS_testFrame));
    cinemagraph.background = ref_img.clone();
    Cinemagraph::ComputeBlendMap(cinemagraph.pixel_loc_display, cinemagraph.background.cols, cinemagraph.background.rows, 5, 300, cinemagraph.blend_map_display);
    CHECK(Cinemagraph::check_cinemagraph(cinemagraph));

    //detect display regions
    for(auto i=0; i<cinemagraph.pixel_loc_display.size(); ++i) {
        Cinemagraph::ApproximateQuad(cinemagraph.pixel_loc_display[i], ref_img.cols, ref_img.rows, cinemagraph.corners[i], false);
    }

    sprintf(buffer, "%s/temp/%s", file_io.getDirectory().c_str(), argv[2]);
    Cinemagraph::SaveCinemagraph(string(buffer), cinemagraph);

//    vector<Mat> render;
//    printf("Rendering %s\n", argv[2]);
//    Cinemagraph::RenderCinemagraph(cinemagraph, render, FLAGS_kFrame);
//    VideoWriter resultWriter(argv[2], CV_FOURCC('x', '2', '6', '4'), 24, cinemagraph.background.size());
//    CHECK(resultWriter.isOpened()) << argv[2];
//    for(const auto& img: render){
//        resultWriter << img;
//    }

    return 0;
}

