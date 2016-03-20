//
// Created by yanhang on 3/20/16.
//

#include "proposal.h"
#include "base/plane3D.h"
#include "external/segment_ms/msImageProcessor.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
    ProposalSfM::ProposalSfM(const FileIO& file_io_, const cv::Mat& img_, const theia::Reconstruction& r_,
                             const int anchor_, const int nLabel_, const double min_disp_, const double max_disp_,
                             const double downsample_, const int segNum_):
            file_io(file_io_), image(img_), anchor(anchor_), nLabel(nLabel_), min_disp(min_disp_), max_disp(max_disp_),
            downsample(downsample_), w(img_.cols), h(img_.rows), reconstruction(r_), segNum(segNum_), method("SfMMeanshift"){
        mults = vector<double>{1,2,3,4,5,6,7};
        params = vector<double>{1,1.5,10,100};
    }

    void ProposalSfM::segment(std::vector<std::vector<std::vector<int> > > &seg) {
        //segment with meanshift
        for(auto pid=0; pid < segNum; ++pid){
            vector<vector<int> > curseg;
            segment_ms::msImageProcessor segmentor;
            segmentor.DefineImage(image.data, segment_ms::COLOR, image.rows, image.cols);
            segmentor.Segment((int)(params[0]*mults[pid]), (float)(params[1]*mults[pid]), (int)(params[2]*mults[pid]), meanshift::MED_SPEEDUP);

            int* labels = NULL;
            labels = segmentor.GetLabels();
            CHECK(labels != NULL);
            curseg.resize((size_t) segmentor.GetRegionCount());

            cv::Vec3b colorTable[] = {cv::Vec3b(255, 0, 0), cv::Vec3b(0, 255, 0), cv::Vec3b(0, 0, 255), cv::Vec3b(255, 255, 0),
                                      cv::Vec3b(255, 0, 255), cv::Vec3b(0, 255, 255), cv::Vec3b(128,128,0), cv::Vec3b(128,0,128), cv::Vec3b(0,128,128)};
            cv::Mat segTestRes(h, w, image.type());
            for (auto y = 0; y < h; ++y) {
                for (auto x = 0; x < w; ++x) {
                    int l = labels[y * w + x];
                    segTestRes.at<cv::Vec3b>(y,x) = colorTable[l%9];
                }
            }
            char buffer[1024] = {};
            sprintf(buffer, "%s/temp/%s%03d.jpg", file_io.getDirectory().c_str(), method.c_str(), pid);
            imwrite(buffer, segTestRes);

            for(auto i=0; i<w*h; ++i){
                int l = labels[i];
                CHECK_LT(l, curseg.size());
                curseg[l].push_back(i);
            }
            seg.push_back(curseg);
        }
    }

    void ProposalSfM::genProposal(std::vector<Depth>& proposals){
        char buffer[1024] = {};
        const int nPixels = w * h;
        const double epsilon = (double)1e-05;
        const double min_depth = 1.0 / max_disp;
        const double max_depth = 1.0 / min_disp;
        const double dis_thres = 0.3 * (max_depth - min_depth);
        printf("dis_thres: %.3f\n", dis_thres);
        //if one segmentation contains less than min_track_num points, invalidate this segment
        const int min_track_num = 5;

        vector<vector<vector<int> > > segs;
        segment(segs);

        Depth zBuffer;
        zBuffer.initialize(w,h,numeric_limits<double>::max());
        vector<theia::TrackId> trackIds = reconstruction.TrackIds();
        const theia::Camera& cam = reconstruction.View(anchor)->Camera();
        for(auto tid: trackIds){
            const theia::Track* t = reconstruction.Track(tid);
            const Vector4d& sptH = t->Point();
            Vector2d imgpt;
            double d = cam.ProjectPoint(sptH, &imgpt);
            if(d > max_depth || d < min_depth)
                continue;
            imgpt /= downsample;
            if(imgpt[0] <= 0 || imgpt[1] <= 0 || imgpt[0] >= w || imgpt[1] >= h)
                continue;
            zBuffer((int)imgpt[0], (int)imgpt[1]) = std::min(zBuffer((int)imgpt[0], (int)imgpt[1]), d);
            //printf("zBuffer(%d,%d)=%.3f\n", (int)imgpt[0], (int)imgpt[1], zBuffer((int)imgpt[0], (int)imgpt[1]));
        }
        for(auto i=0; i<nPixels; ++i){
            if(zBuffer[i] > max_depth)
                zBuffer[i] = -1;
        }
        sprintf(buffer, "%s/temp/zbuffer.jpg", file_io.getDirectory().c_str());
        zBuffer.saveImage(string(buffer), -1);


        proposals.resize(segs.size());
        for(auto i=0; i<segs.size(); ++i) {
            proposals[i].initialize(w, h, -1);
            int index = 0;
            for (const auto &seg: segs[i]) {
                vector<Vector3d> pts;
                pts.reserve(seg.size());
                for (auto idx: seg) {
                    CHECK_LT(idx, nPixels);
                    const int x = idx % w;
                    const int y = idx / w;
                    if (zBuffer[idx] > 0) {
                        Vector3d ray = cam.PixelToUnitDepthRay(Vector2d(x * downsample, y * downsample));
                        ray.normalize();
                        Vector3d spt = cam.GetPosition() + zBuffer[idx] * ray;
                        pts.push_back(spt);
                    }
                }
                if (pts.size() < min_track_num)
                    continue;

                //check if points are colinear
                Eigen::MatrixXd A(pts.size(), 3);
                for (auto i = 0; i < pts.size(); ++i) {
                    A(i, 0) = pts[i][0];
                    A(i, 1) = pts[i][1];
                    A(i, 2) = 1.0;
                }
                Eigen::Matrix3d A2 = A.transpose() * A;
                double A2det = A2.determinant();
                if (A2det < epsilon)
                    continue;
                Plane3D segPln;
                if (!plane_util::planeFromPointsRANSAC(pts, segPln, dis_thres, 5000))
                    continue;

                double offset = segPln.getOffset();
                Vector3d n = segPln.getNormal();
                if (std::abs(n[2]) < epsilon)
                    continue;

                for (const auto idx: seg) {
                    int x = idx % w;
                    int y = idx / w;
                    double newdepth = (-1 * offset - n[0] * x - n[1] * y) / n[2];
                    //double d = std::max(std::min(newdepth, max_depth), 0.0);
                    //proposals[i][idx] = std::max(std::min(depthToDisp(d), (double) nLabel - 1), 0.0);
                    proposals[i][idx] = newdepth;
                }
            }


            sprintf(buffer, "%s/temp/tdisp_%s_%03d_2.jpg", file_io.getDirectory().c_str(), method.c_str(), i);
            proposals[i].saveImage(std::string(buffer), -1);
        }
    }
}//namespace dynamic_stereo

