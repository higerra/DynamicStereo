//
// Created by yanhang on 4/10/16.
//

#include "dynamicwarpping.h"
#include "../base/utility.h"

using namespace std;
using namespace Eigen;
using namespace cv;

namespace dynamic_stereo {
    DynamicWarpping::DynamicWarpping(const FileIO &file_io_, const int anchor_, const int tWindow_,
                                     const int downsample_,
                                     const std::vector<Depth> &depths, const std::vector<int> &depthind) :
            file_io(file_io_), anchor(anchor_), downsample(downsample_) {
        if (anchor - tWindow_ / 2 < 0) {
            offset = 0;
            CHECK_LT(offset + tWindow_, file_io.getTotalNum());
        } else if (anchor + tWindow_ / 2 >= file_io.getTotalNum()) {
            offset = file_io.getTotalNum() - 1 - tWindow_;
            CHECK_GE(offset, 0);
        } else
            offset = anchor - tWindow_ / 2;

        //reading images. Set (0,0,0) to (1,1,1);
        images.resize((size_t) tWindow_);
        for (auto i = 0; i < images.size(); ++i) {
            images[i] = imread(file_io.getImage(i + offset));
            for (auto y = 0; y < images[i].rows; ++y) {
                for (auto x = 0; x < images[i].cols; ++x) {
                    if (images[i].at<Vec3b>(y, x) == Vec3b(0, 0, 0))
                        images[i].at<Vec3b>(y, x) = Vec3b(1, 1, 1);
                }
            }
        }
        CHECK(!images.empty());

        width = images[0].cols;
        height = images[0].rows;
        //initialize reconstruction
        sfmModel.init(file_io.getReconstruction());

        for (auto i = 0; i < depthind.size(); ++i) {
            if (depthind[i] == anchor) {
                refDepth = depths[i];
                break;
            }
        }
        CHECK_EQ(refDepth.getWidth(), width / downsample);
        CHECK_EQ(refDepth.getHeight(), height / downsample);

        initZBuffer(depths, depthind);
    }

    void DynamicWarpping::updateZBuffer(const Depth &depth, Depth &zb, const theia::Camera &cam1,
                                        const theia::Camera &cam2) const {
        CHECK_EQ(depth.getWidth(), zb.getWidth());
        CHECK_EQ(depth.getHeight(), zb.getHeight());
        const int w = depth.getWidth();
        const int h = depth.getHeight();
        for (auto y = 0; y < h; ++y) {
            for (auto x = 0; x < w; ++x) {
                Vector3d spt = cam1.PixelToUnitDepthRay(Vector2d(x * downsample, y * downsample)) * depth(x, y) +
                               cam1.GetPosition();
                Vector2d imgpt;
                double curd = cam2.ProjectPoint(spt.homogeneous(), &imgpt);
                int intx = (int) imgpt[0];
                int inty = (int) imgpt[1];
                if (intx >= 0 && inty >= 0 && intx < w && inty < h) {
                    if (zb(intx, inty) < 0 || (zb(intx, inty) >= 0 && curd < zb(intx, inty)))
                        zb(intx, inty) = curd;
                }
            }
        }
    }

    void DynamicWarpping::initZBuffer(const std::vector<Depth> &depths, const std::vector<int> &depthind) {
        CHECK_EQ(depths.size(), depthind.size());
        //for each frame, find nearest depth
        zBuffers.resize(images.size());
        for (auto i = 0; i < images.size(); ++i) {
            zBuffers[i].initialize(width / downsample, height / downsample, -1);
            if (i + offset <= depthind[0]) {
                updateZBuffer(depths[0], zBuffers[i], sfmModel.getCamera(depthind[0]), sfmModel.getCamera(i + offset));
                continue;
            }
            if (i + offset >= depthind.back()) {
                updateZBuffer(depths.back(), zBuffers[i], sfmModel.getCamera(depthind.back()),
                              sfmModel.getCamera(i + offset));
                continue;
            }
            for (auto j = 1; j < depthind.size(); ++j) {
                if (i + offset > depthind[j - 1] && i + offset <= depthind[j]) {
                    updateZBuffer(depths[j - 1], zBuffers[i], sfmModel.getCamera(depthind[j - 1]),
                                  sfmModel.getCamera(i + offset));
                    updateZBuffer(depths[j], zBuffers[i], sfmModel.getCamera(depthind[j]),
                                  sfmModel.getCamera(i + offset));
                }
            }
        }
    }

    void DynamicWarpping::warpToAnchor(const cv::Mat &mask, std::vector<cv::Mat> &warpped,
                                       const bool earlyTerminate) const {
        cout << "Warpping..." << endl;
        warpped.resize(images.size());
        CHECK_EQ(mask.cols, width);
        CHECK_EQ(mask.rows, height);
        CHECK_EQ(mask.channels(), 1);

        const theia::Camera &cam1 = sfmModel.getCamera(anchor);
        //compute zBuffer
        vector<double> zMargin(zBuffers.size(), 0.0);
        for (auto i = 0; i < images.size(); ++i) {
            zBuffers[i].updateStatics();
            zMargin[i] = zBuffers[i].getMedianDepth() / 5.0;
        }
        for (auto i = 0; i < images.size(); ++i) {
            cout << i + offset << ' ' << flush;
            if (i == anchor - offset) {
                warpped[i] = images[i].clone();
                continue;
            } else {
                warpped[i] = Mat(height, width, CV_8UC3, Scalar(0, 0, 0));
            }
            //const theia::Camera cam2 = reconstruction.View(orderedId[i+offset].second)->Camera();
            const theia::Camera &cam2 = sfmModel.getCamera(i + offset);
            for (auto y = downsample; y < height - downsample; ++y) {
                for (auto x = downsample; x < width - downsample; ++x) {
                    if (mask.at<uchar>(y, x) < 200) {
                        warpped[i].at<Vec3b>(y, x) = images[anchor - offset].at <Vec3b>(y, x);
                        continue;
                    }
                    Vector3d ray = cam1.PixelToUnitDepthRay(Vector2d(x, y));
                    //ray.normalize();
                    Vector3d spt = cam1.GetPosition() + ray * refDepth((int) x / downsample, (int) y / downsample);
                    Vector2d imgpt;
                    double curdepth = cam2.ProjectPoint(spt.homogeneous(), &imgpt);
                    int dptx = (int) imgpt[0] / downsample;
                    int dpty = (int) imgpt[1] / downsample;
                    if (dptx >= 0 && dptx < width && dpty >= 0 && dpty < height) {
                        if ((zBuffers[i](dptx, dpty) > 0) && (curdepth > zBuffers[i](dptx, dpty) + zMargin[i])) {
                            warpped[i].at<Vec3b>(y, x) = images[anchor - offset].at <Vec3b>(y, x);
                            continue;
                        }
                    }
                    if (imgpt[0] >= 1 && imgpt[1] >= 1 && imgpt[0] < width - 1 && imgpt[1] < height - 1) {
                        Vector3d pix2 = interpolation_util::bilinear<uchar, 3>(images[i].data, width, height,
                                                                               imgpt);
                        warpped[i].at<Vec3b>(y, x) = Vec3b(pix2[0], pix2[1], pix2[2]);
                    }
                }
            }
        }
        cout << endl;
    }
}//namespace dynamic_stereo