// Copyright (C) 2015 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <time.h>
#include <theia/theia.h>
#include <algorithm>
#include <chrono>  // NOLINT
#include <string>
#include <vector>
#include <stlplus3/file_system.hpp>
#include <opencv2/opencv.hpp>
#include "../base/file_io.h"

#include "command_line_helpers.h"

// Input/output files.
DEFINE_string(matches_file, "", "Filename of the matches file.");
DEFINE_string(calibration_file, "",
              "Calibration file containing image calibration data.");
DEFINE_string(
        output_matches_file, "",
        "File to write the two-view matches to. This file can be used in "
                "future iterations as input to the reconstruction builder. Leave empty if "
                "you do not want to output matches.");

// Multithreading.
DEFINE_int32(num_threads, 6,
             "Number of threads to use for feature extraction and matching.");

DEFINE_int32(image_intervals, 1, "Use image every ${image_intervals}");

// Feature and matching options.
DEFINE_string(
        descriptor, "SIFT",
        "Type of feature descriptor to use. Must be one of the following: "
                "SIFT");
DEFINE_string(matching_strategy, "CASCADE_HASHING",
              "Strategy used to match features. Must be BRUTE_FORCE "
                      " or CASCADE_HASHING");
DEFINE_bool(match_out_of_core, true,
            "Perform matching out of core by saving features to disk and "
                    "reading them as needed. Set to false to perform matching all in "
                    "memory.");
DEFINE_int32(matching_max_num_images_in_cache, 128,
             "Maximum number of images to store in the LRU cache during "
                     "feature matching. The higher this number is the more memory is "
                     "consumed during matching.");
DEFINE_double(lowes_ratio, 0.8, "Lowes ratio used for feature matching.");
DEFINE_double(
        max_sampson_error_for_verified_match, 4.0,
        "Maximum sampson error for a match to be considered geometrically valid.");
DEFINE_int32(min_num_inliers_for_valid_match, 30,
             "Minimum number of geometrically verified inliers that a pair on "
                     "images must have in order to be considered a valid two-view "
                     "match.");
DEFINE_bool(bundle_adjust_two_view_geometry, true,
            "Set to false to turn off 2-view BA.");
DEFINE_bool(keep_only_symmetric_matches, true,
            "Performs two-way matching and keeps symmetric matches.");

// Reconstruction building options.
DEFINE_string(reconstruction_estimator, "GLOBAL",
              "Type of SfM reconstruction estimation to use.");
DEFINE_bool(reconstruct_largest_connected_component, true,
            "If set to true, only the single largest connected component is "
                    "reconstructed. Otherwise, as many models as possible are "
                    "estimated.");
DEFINE_bool(only_calibrated_views, false,
            "Set to true to only reconstruct the views where calibration is "
                    "provided or can be extracted from EXIF");
DEFINE_int32(max_track_length, 50, "Maximum length of a track.");
DEFINE_string(intrinsics_to_optimize,
              "FOCAL_LENGTH|PRINCIPAL_POINTS|RADIAL_DISTORTION",
              "Set to control which intrinsics parameters are optimized during "
                      "bundle adjustment.");
DEFINE_double(max_reprojection_error_pixels, 4.0,
              "Maximum reprojection error for a correspondence to be "
                      "considered an inlier after bundle adjustment.");

// Global SfM options.
DEFINE_string(global_rotation_estimator, "ROBUST_L1L2",
              "Type of global rotation estimation to use for global SfM.");
DEFINE_string(global_position_estimator, "NONLINEAR",
              "Type of global position estimation to use for global SfM.");
DEFINE_bool(refine_relative_translations_after_rotation_estimation, true,
            "Refine the relative translation estimation after computing the "
                    "absolute rotations. This can help improve the accuracy of the "
                    "position estimation.");
DEFINE_double(post_rotation_filtering_degrees, 15.0,
              "Max degrees difference in relative rotation and rotation "
                      "estimates for rotation filtering.");
DEFINE_bool(extract_maximal_rigid_subgraph, false,
            "If true, only cameras that are well-conditioned for position "
                    "estimation will be used for global position estimation.");
DEFINE_bool(filter_relative_translations_with_1dsfm, true,
            "Filter relative translation estimations with the 1DSfM algorithm "
                    "to potentially remove outlier relativep oses for position "
                    "estimation.");
DEFINE_int32(num_retriangulation_iterations, 1,
             "Number of times to retriangulate any unestimated tracks. Bundle "
                     "adjustment is performed after retriangulation.");

// Nonlinear position estimation options.
DEFINE_int32(
        position_estimation_min_num_tracks_per_view, 0,
        "Minimum number of point to camera constraints for position estimation.");
DEFINE_double(position_estimation_robust_loss_width, 0.1,
              "Robust loss width to use for position estimation.");

// Incremental SfM options.
DEFINE_double(absolute_pose_reprojection_error_threshold, 8.0,
              "The inlier threshold for absolute pose estimation.");
DEFINE_int32(min_num_absolute_pose_inliers, 30,
             "Minimum number of inliers in order for absolute pose estimation "
                     "to be considered successful.");
DEFINE_double(full_bundle_adjustment_growth_percent, 5.0,
              "Full BA is only triggered for incremental SfM when the "
                      "reconstruction has growth by this percent since the last time "
                      "full BA was used.");
DEFINE_int32(partial_bundle_adjustment_num_views, 20,
             "When full BA is not being run, partial BA is executed on a "
                     "constant number of views specified by this parameter.");


// Triangulation options.
DEFINE_double(min_triangulation_angle_degrees, 4.0,
              "Minimum angle between views for triangulation.");
DEFINE_double(
        triangulation_reprojection_error_pixels, 15.0,
        "Max allowable reprojection error on initial triangulation of points.");
DEFINE_bool(bundle_adjust_tracks, true,
            "Set to true to optimize tracks immediately upon estimation.");

// Bundle adjustment parameters.
DEFINE_string(bundle_adjustment_robust_loss_function, "NONE",
              "By setting this to an option other than NONE, a robust loss "
                      "function will be used during bundle adjustment which can "
                      "improve robustness to outliers. Options are NONE, HUBER, "
                      "SOFTLONE, CAUCHY, ARCTAN, and TUKEY.");
DEFINE_double(bundle_adjustment_robust_loss_width, 10.0,
              "If the BA loss function is not NONE, then this value controls "
                      "where the robust loss begins with respect to reprojection error "
                      "in pixels.");

// Sift parameters.
DEFINE_int32(sift_num_octaves, -1, "Number of octaves in the scale space. "
        "Set to a value less than 0 to use the maximum  ");
DEFINE_int32(sift_num_levels, 3, "Number of levels per octave.");
DEFINE_int32(sift_first_octave, -1, "The index of the first octave");
DEFINE_double(sift_edge_threshold, 10.0f,
              "The edge threshold value is used to remove spurious features."
                      " Reduce threshold to reduce the number of keypoints.");
// The default value is calculated using the following formula:
// 255.0 * 0.02 / num_levels.
DEFINE_double(sift_peak_threshold, 1.7f,
              "The peak threshold value is used to remove features with weak "
                      "responses. Increase threshold value to reduce the number of "
                      "keypoints");
DEFINE_bool(root_sift, true, "Enables the usage of Root SIFT.");

using theia::Reconstruction;
using theia::ReconstructionBuilder;
using theia::ReconstructionBuilderOptions;
using namespace dynamic_stereo;
// Sets the feature extraction, matching, and reconstruction options based on
// the command line flags. There are many more options beside just these located
// in //theia/vision/sfm/reconstruction_builder.h
ReconstructionBuilderOptions SetReconstructionBuilderOptions(const FileIO& file_io) {
    ReconstructionBuilderOptions options;
    options.num_threads = FLAGS_num_threads;
    options.output_matches_file = file_io.getSfMMatchFile();

    options.descriptor_type = StringToDescriptorExtractorType(FLAGS_descriptor);
    // Setting sift parameters.
    if (options.descriptor_type == DescriptorExtractorType::SIFT) {
        options.sift_parameters.num_octaves = FLAGS_sift_num_octaves;
        options.sift_parameters.num_levels = FLAGS_sift_num_levels;
        CHECK_GT(options.sift_parameters.num_levels, 0)
            << "The number of levels must be positive";
        options.sift_parameters.first_octave = FLAGS_sift_first_octave;
        options.sift_parameters.edge_threshold = FLAGS_sift_edge_threshold;
        options.sift_parameters.peak_threshold = FLAGS_sift_peak_threshold;
        options.sift_parameters.root_sift = FLAGS_root_sift;
    }

    options.matching_options.match_out_of_core = FLAGS_match_out_of_core;
    options.matching_options.keypoints_and_descriptors_output_dir =
            file_io.getMvgDirectory();


    options.matching_options.cache_capacity =
            FLAGS_matching_max_num_images_in_cache;
    options.matching_strategy =
            StringToMatchingStrategyType(FLAGS_matching_strategy);
    options.matching_options.lowes_ratio = FLAGS_lowes_ratio;
    options.matching_options.keep_only_symmetric_matches =
            FLAGS_keep_only_symmetric_matches;
    options.min_num_inlier_matches = FLAGS_min_num_inliers_for_valid_match;
    options.geometric_verification_options.estimate_twoview_info_options
            .max_sampson_error_pixels = FLAGS_max_sampson_error_for_verified_match;
    options.geometric_verification_options.bundle_adjustment =
            FLAGS_bundle_adjust_two_view_geometry;

    options.max_track_length = FLAGS_max_track_length;

    // Reconstruction Estimator Options.
    theia::ReconstructionEstimatorOptions& reconstruction_estimator_options =
            options.reconstruction_estimator_options;
    reconstruction_estimator_options.min_num_two_view_inliers =
            FLAGS_min_num_inliers_for_valid_match;
    reconstruction_estimator_options.num_threads = FLAGS_num_threads;
    reconstruction_estimator_options.intrinsics_to_optimize =
            StringToOptimizeIntrinsicsType(FLAGS_intrinsics_to_optimize);

    options.reconstruct_largest_connected_component =
            FLAGS_reconstruct_largest_connected_component;
    options.only_calibrated_views = FLAGS_only_calibrated_views;
    reconstruction_estimator_options.max_reprojection_error_in_pixels =
            FLAGS_max_reprojection_error_pixels;

    // Which type of SfM pipeline to use (e.g., incremental, global, etc.);
    reconstruction_estimator_options.reconstruction_estimator_type =
            StringToReconstructionEstimatorType(FLAGS_reconstruction_estimator);

    // Global SfM Options.
    reconstruction_estimator_options.global_rotation_estimator_type =
            StringToRotationEstimatorType(FLAGS_global_rotation_estimator);
    reconstruction_estimator_options.global_position_estimator_type =
            StringToPositionEstimatorType(FLAGS_global_position_estimator);
    reconstruction_estimator_options.num_retriangulation_iterations =
            FLAGS_num_retriangulation_iterations;
    reconstruction_estimator_options
            .refine_relative_translations_after_rotation_estimation =
            FLAGS_refine_relative_translations_after_rotation_estimation;
    reconstruction_estimator_options.extract_maximal_rigid_subgraph =
            FLAGS_extract_maximal_rigid_subgraph;
    reconstruction_estimator_options.filter_relative_translations_with_1dsfm =
            FLAGS_filter_relative_translations_with_1dsfm;
    reconstruction_estimator_options
            .rotation_filtering_max_difference_degrees =
            FLAGS_post_rotation_filtering_degrees;
    reconstruction_estimator_options.nonlinear_position_estimator_options
            .min_num_points_per_view =
            FLAGS_position_estimation_min_num_tracks_per_view;

    // Incremental SfM Options.
    reconstruction_estimator_options
            .absolute_pose_reprojection_error_threshold =
            FLAGS_absolute_pose_reprojection_error_threshold;
    reconstruction_estimator_options.min_num_absolute_pose_inliers =
            FLAGS_min_num_absolute_pose_inliers;
    reconstruction_estimator_options
            .full_bundle_adjustment_growth_percent =
            FLAGS_full_bundle_adjustment_growth_percent;
    reconstruction_estimator_options.partial_bundle_adjustment_num_views =
            FLAGS_partial_bundle_adjustment_num_views;

    // Triangulation options (used by all SfM pipelines).
    reconstruction_estimator_options.min_triangulation_angle_degrees =
            FLAGS_min_triangulation_angle_degrees;
    reconstruction_estimator_options
            .triangulation_max_reprojection_error_in_pixels =
            FLAGS_triangulation_reprojection_error_pixels;
    reconstruction_estimator_options.bundle_adjust_tracks =
            FLAGS_bundle_adjust_tracks;

    // Bundle adjustment options (used by all SfM pipelines).
    reconstruction_estimator_options.bundle_adjustment_loss_function_type =
            StringToLossFunction(FLAGS_bundle_adjustment_robust_loss_function);
    reconstruction_estimator_options.bundle_adjustment_robust_loss_width =
            FLAGS_bundle_adjustment_robust_loss_width;
    return options;
}

void AddImagesToReconstructionBuilder(const FileIO& file_io,
                                      ReconstructionBuilder* reconstruction_builder) {
    std::vector<std::string> image_files;
    CHECK_GT(FLAGS_image_intervals, 0);
    char buffer[1024] = {};
    int idx = 0;
    while(true){
        sprintf(buffer, "%s/images_input/image%05d.jpg", file_io.getDirectory().c_str(), idx);
        if(!stlplus::file_exists(buffer))
            break;
	    image_files.push_back(std::string(buffer));
	    std::cout << buffer << std::endl;
        idx += FLAGS_image_intervals;
    }

    CHECK_GT(image_files.size(), 0) << "No images found in: " << file_io.getImageDirectory();

    // Load calibration file if it is provided.
    std::unordered_map<std::string, theia::CameraIntrinsicsPrior>
            camera_intrinsics_prior;
    if (FLAGS_calibration_file.size() != 0) {
        CHECK(theia::ReadCalibration(FLAGS_calibration_file,
                                     &camera_intrinsics_prior))
        << "Could not read calibration file.";
    }

    // Add images with possible calibration.
    for (const std::string& image_file : image_files) {
        std::string image_filename;
        CHECK(theia::GetFilenameFromFilepath(image_file, true, &image_filename));

        const theia::CameraIntrinsicsPrior* image_camera_intrinsics_prior =
                FindOrNull(camera_intrinsics_prior, image_filename);
        if (image_camera_intrinsics_prior != nullptr) {
            CHECK(reconstruction_builder->AddImageWithCameraIntrinsicsPrior(
                    image_file, *image_camera_intrinsics_prior));
        } else {
            CHECK(reconstruction_builder->AddImage(image_file));
        }
    }

    // Extract and match features.
    CHECK(reconstruction_builder->ExtractAndMatchFeatures());
}

int main(int argc, char *argv[]) {
    if(argc < 2){
        std::cerr << "Usage: SfM_cli <path-to-data>" << std::endl;
        return 1;
    }
    THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    CHECK(stlplus::folder_exists(std::string(argv[1]))) << "Invalid path";
    dynamic_stereo::FileIO file_io(argv[1]);

    char buffer[1024] = {};
    sprintf(buffer, "%s/images_input", file_io.getDirectory().c_str());
    CHECK(stlplus::folder_exists(buffer)) << "Please put all image inside images_input folder";

    if(!stlplus::folder_exists(file_io.getMvgDirectory()))
        stlplus::folder_create(file_io.getMvgDirectory());

    if(!stlplus::folder_exists(file_io.getSfMDirectory()))
        stlplus::folder_create(file_io.getSfMDirectory());

    if(!stlplus::folder_exists(file_io.getImageDirectory()))
        stlplus::folder_create(file_io.getImageDirectory());

    std::string output_ply = file_io.getSfMDirectory() + "/reconstruction.ply";


    int totalNum = 0;

    //count totalNum of images
    while(true){
        sprintf(buffer, "%s/images_input/image%05d.jpg", file_io.getDirectory().c_str(), totalNum * FLAGS_image_intervals);
        if(!stlplus::file_exists(buffer))
            break;
        totalNum++;
    }

    theia::Reconstruction *res = new theia::Reconstruction();
    if(!theia::ReadReconstruction(file_io.getReconstruction(), res)) {

        const ReconstructionBuilderOptions options =
                SetReconstructionBuilderOptions(file_io);
        ReconstructionBuilder reconstruction_builder(options);

        AddImagesToReconstructionBuilder(file_io, &reconstruction_builder);

        std::vector<Reconstruction *> reconstructions;
        CHECK(reconstruction_builder.BuildReconstruction(&reconstructions)) << "Could not create a reconstruction.";

        //colorized and write ply file
        CHECK(!reconstructions.empty());
        res = reconstructions.front();
        //colorize reconstruction
        CHECK(theia::WriteReconstruction(*res, file_io.getReconstruction())) << "Cannot write reconstruction file";
    }
    theia::ColorizeReconstruction(file_io.getDirectory()+"/images_input/", FLAGS_num_threads, res);
    CHECK(theia::WritePlyFile(output_ply, *(res), 3)) << "Cannot write ply file";


    //std::vector<theia::Matrix3x4d> pMatrix(file_io.getTotalNum());

    printf("%d out of %d images are registered. Unregistered images are discarded\n", res->NumViews(), totalNum);
    //re-index images, ignore un-registered images
    //view ids returned by theia might be unordered
    std::vector<theia::ViewId> viewIds = res->ViewIds();
    std::vector<std::pair<int, theia::ViewId> > orderedId(viewIds.size());
    int index = 0;
    for(auto vid: viewIds){
        const theia::View* v = res->View(vid);
        std::string nstr = v->Name().substr(5,5);
        int idx = atoi(nstr.c_str());
        orderedId[index].first = idx;
        orderedId[index].second = vid;
        index++;
    }
    std::sort(orderedId.begin(), orderedId.end(),
              [](const std::pair<int, theia::ViewId>& v1, const std::pair<int, theia::ViewId>& v2){return v1.first < v2.first;});

    index = 0;
    for(auto vpair: orderedId){
        const theia::ViewId& vid = vpair.second;
        const theia::View* v = res->View(vid);
        CHECK(v) << "View " << vid << " is Null";
        const theia::Camera cam = v->Camera();
        std::string nstr = v->Name().substr(5,5);
        int idx = atoi(nstr.c_str());
        sprintf(buffer, "%s/images_input/image%05d.jpg", file_io.getDirectory().c_str(), idx);
        cv::Mat img = cv::imread(buffer);
        sprintf(buffer, "%s/images/image%05d.jpg", file_io.getDirectory().c_str(), index);
        printf("Saving %s as %d\n", nstr.c_str(), index);
        cv::imwrite(buffer, img);
//        theia::Matrix3x4d p;
//        cam.GetProjectionMatrix(&p);
//        sprintf(buffer, "%s/pose%05d.txt", file_io.getSfMDirectory().c_str(), index);
//        std::ofstream fout(buffer);
//        CHECK(fout.is_open()) << "Can not open " << buffer << " to write";
//        for(auto y=0; y<3; ++y){
//            for(auto x=0; x<4; ++x)
//                fout << p(y,x) << ' ';
//            fout << std::endl;
//        }
        index++;
    }

    return 0;
}
