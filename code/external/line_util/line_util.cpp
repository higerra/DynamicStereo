//
// Created by yanhang on 5/24/16.
//

#include "line_util.h"
#include "vpdetection/VPCluster.h"
#include "vpdetection/VPSample.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace LineUtil{

    void detectLineSegments(const cv::Mat& input, std::vector<KeyLine>& output, const double min_length){
        CHECK(input.data);
        cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
        vector<Vec4f> templines;
        Mat gray;
        cvtColor(input, gray, CV_RGB2GRAY);
        ls->detect(gray, templines);
        for(auto &ln: templines){
            Vector2d spt(ln[0], ln[1]);
            Vector2d ept(ln[2], ln[3]);
            double curlength = (ept-spt).norm();
            if(curlength >= min_length)
                output.push_back(KeyLine(spt, ept));
        }
    }

    bool solveVP(const vector<KeyLine>& lines,
                 Vector3d& vp){
        const double epsilon = 0.01;
        const int line_count = lines.size();
        vector<Vector3d> lines_homo(lines.size());
        for(int i=0; i<lines.size(); i++){
            const Vector2d& s = lines[i].startPoint;
            const Vector2d& e = lines[i].endPoint;
            Vector3d pt1(s[0], s[1], 1.0);
            Vector3d pt2(e[0], e[1], 1.0);
            Vector3d cur_homo = pt1.cross(pt2);
            double normal_factor = std::sqrt(cur_homo[0]*cur_homo[0] + cur_homo[1] * cur_homo[1]);
            if(normal_factor != 0){
                cur_homo[0] /= normal_factor;
                cur_homo[1] /= normal_factor;
                cur_homo[2] /= normal_factor;
            }
            lines_homo[i] = cur_homo;
        }

        if(lines_homo.size() == 2){
            vp = lines_homo[0].cross(lines_homo[1]);
            dehomoPoint(vp);
            return true;
        }

        MatrixXd A(line_count, 3);
        for(int i=0; i<lines_homo.size(); i++){
            A.block<1,3>(i,0) = lines_homo[i];
        }

        JacobiSVD<MatrixXd>svd(A, ComputeThinV);
        Matrix3d V = svd.matrixV();
        for(int i=0; i<3; i++)
            vp[i] = V(i,2);
        dehomoPoint(vp);
        return true;
    }

    void solveVP_RANSAC(const vector<KeyLine>& lines,
                        Vector3d& vp){
        CHECK_GE(lines.size(), 3);
        if(lines.size() == 2){
            solveVP(lines, vp);
            return;
        }

        int max_inliercount = -1;
        double inlier_threshold = 1.0;
        vector<bool> is_inlier;
        for(int ind1=0; ind1 < lines.size()-1; ind1++){
            for(int ind2=ind1+1; ind2 < lines.size(); ind2++){
                Vector3d curVP;
                vector<KeyLine> curpair;
                curpair.push_back(lines[ind1]);
                curpair.push_back(lines[ind2]);
                solveVP(curpair, curVP);
                int inliercount = 0;
                vector<bool> cur_inlier(lines.size(), false);
                for(int i=0; i<lines.size(); i++){
                    Vector3d line_homo = lines[i].getHomo();
                    double curdis = line_homo.dot(curVP);
                    if(curdis < inlier_threshold){
                        inliercount++;
                        cur_inlier[i] = true;
                    }
                }
                if(inliercount > max_inliercount){
                    max_inliercount = inliercount;
                    is_inlier.swap(cur_inlier);
                }
            }
        }

        vector<KeyLine> lines_inlier;
        for(int i=0; i<lines.size(); i++){
            if(is_inlier[i])
                lines_inlier.push_back(lines[i]);
        }
        //printf("inlier count: %d\n", (int)lines_inlier.size());
        solveVP(lines_inlier, vp);
    }


    void vpDetection(const std::vector<KeyLine>& lines,
                     std::vector<std::vector<KeyLine> >& line_group,
                     std::vector<Eigen::Vector3d>& vp,
                     const int min_line_num,
                     const int max_cluster_num){
        //dirty trick: identify extract nearly vertical lines in advance
        CHECK_GT(lines.size(), min_line_num);
        const double cos_margin = cos(70.0 * 3.1415926 / 180.0);
        Vector2d hline(1.0,0.0);
        vector<KeyLine> lines_non_vertical;
        vector<KeyLine> lines_vertical;
        for(int i=0; i<lines.size(); i++){
            double cosv = std::abs(hline.dot(lines[i].getLineDir()));
            if(cosv < cos_margin)
                lines_vertical.push_back(lines[i]);
            else
                lines_non_vertical.push_back(lines[i]);
        }

        printf("non vertical lines: %d, vertical lines: %d\n", (int)lines_non_vertical.size(), (int)lines_vertical.size());
        vector<vector<float> *> pts;
        for(int i=0; i<lines_non_vertical.size(); i++){
            vector<float>* p = new std::vector<float>(4);
            (*p)[0] = lines_non_vertical[i].startPoint[0];
            (*p)[1] = lines_non_vertical[i].startPoint[1];
            (*p)[2] = lines_non_vertical[i].endPoint[0];
            (*p)[3] = lines_non_vertical[i].endPoint[1];
            pts.push_back(p);
        }
        vector<unsigned int> Labels;
        vector<unsigned int> LabelCount;
        int classNum;

        vector<vector<float> *> *mModels = VPSample::run(&pts, 5000, 2, 0, 3);
        classNum = VPCluster::run(Labels, LabelCount, &pts, mModels, 3.5, 2);

        for(unsigned int i=0; i < mModels->size(); ++i)
            delete (*mModels)[i];
        delete mModels;
        for(unsigned int i=0; i<pts.size(); i++)
            delete pts[i];

        line_group.resize(classNum+1);

        for(unsigned int i=0; i<Labels.size(); i++){
            CHECK_LT(Labels[i]+1, line_group.size());
            CHECK_LT(i, lines_non_vertical.size());
            line_group[Labels[i]+1].push_back(lines_non_vertical[i]);
        }

        line_group[0].swap(lines_vertical);

        if(line_group.size() == 1)
            return;

        sort(line_group.begin()+1, line_group.end(),
             [](const std::vector<KeyLine>& gp1, const std::vector<KeyLine>& gp2){return gp1.size() > gp2.size();});

        CHECK(!line_group.empty());
        while(!line_group.empty() && (line_group.size() > max_cluster_num || line_group.back().size() < min_line_num))
            line_group.pop_back();

        vp.resize(line_group.size(), Vector3d(0.0,0.0,0.0));
        solveVP_RANSAC(line_group[0], vp[0]);
        for(int i=1; i<line_group.size();i++){
            if(line_group[i].size() > 1)
                solveVP(line_group[i], vp[i]);
        }
    }

    void mergeLines(std::vector<KeyLine>& lines){

    }

    void drawLines(cv::Mat& input, const std::vector<KeyLine>& lines,
                      const cv::Scalar c, const int thickness){
        CHECK(input.data);
        for(const auto& l: lines)
            cv::line(input, l.getStartPointCV(), l.getEndPointCV(), c, thickness);
    }

    void drawLineGroups(cv::Mat& input, const std::vector<std::vector<KeyLine> >& lines){
        vector<Scalar> colors{Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255),
                           Scalar(255,255,0), Scalar(255,0,255), Scalar(0,255,255)};
        int gid = 0;
        for(const auto& lngroup: lines){
            drawLines(input, lngroup, colors[gid]);
            gid = (gid+1) % (int)colors.size();
        }
    }
}