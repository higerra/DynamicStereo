#ifndef FILE_IO_H
#define FILE_IO_H

#include <string>
#include <iostream>
#include <fstream>
#include <glog/logging.h>
#include <assert.h>

namespace dynamic_rendering{
    class FileIO {
    public:
        void init() {
            int curid = startid;
            while (true) {
                char buffer[1024] = {};
                sprintf(buffer, "%s/images/%s%05d.jpg", directory.c_str(), imagePrefix.c_str(), curid);
                std::ifstream fin(buffer);
                if (!fin.is_open()) {
                    fin.close();
                    break;
                }
                curid++;
                fin.close();
            }
            framenum = curid;

            curid = 0;
            while (true) {
                char buffer[1024] = {};
                sprintf(buffer, "%s/Ply/Scan%05d.ply", directory.c_str(), curid);
                std::ifstream fin(buffer);
                if (!fin.is_open()) {
                    fin.close();
                    break;
                }
                curid++;
                fin.close();
            }
            pointcloudnum = curid;
        }

        FileIO(std::string directory_) : imagePrefix("image"), startid(0), directory(directory_) {
            //get the number of frame
            init();
        }

        FileIO(const std::string &directory_, const std::string &imagePrefix_, const int startid_) :
                directory(directory_), imagePrefix(imagePrefix_), startid(startid_) {
            init();
        }


        inline int getTotalNum() const { return framenum; }

        inline int getStartID() const { return startid; }

        inline int getSinglePointCloudNum() const { return pointcloudnum; }

        inline std::string getPointCloud() {
            char buffer[1024] = {};
            sprintf(buffer, "%s/Scan.ply", directory.c_str());
            return std::string(buffer);
        }

        inline std::string getSinglePointCloud(int id) const {
            CHECK_LT(id, pointcloudnum);
            char buffer[1024] = {};
            sprintf(buffer, "%s/Ply/Scan%05d.ply", directory.c_str(), id);
            return std::string(buffer);
        }

        inline std::string getTransformedPointCloud(int id) const {
            CHECK_LT(id, pointcloudnum);
            char buffer[1024] = {};
            sprintf(buffer, "%s/transformed/transformed%05d.ply", directory.c_str(), id);
//	    sprintf(buffer, "%s/Ply/Scan%03d.ply", directory.c_str(), id);
            return std::string(buffer);
        }

        inline std::string getImage(const int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
            sprintf(buffer, "%s/images/%s%05d.jpg", directory.c_str(), imagePrefix.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getDepthFile(const int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
            sprintf(buffer, "%s/depth/depth%05d.depth", directory.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getDepthImage(const int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
            sprintf(buffer, "%s/depth/depth%05d.jpg", directory.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getInpaintedDepthFile(const int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
            sprintf(buffer, "%s/depth/depth_filled%05d.depth", directory.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getInpaintedDepthImage(const int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
            sprintf(buffer, "%s/depth/depth_filled%05d.jpg", directory.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getSuperpixel(const int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
            sprintf(buffer, "%s/superpixel/frame%05d.SLIC", directory.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getDenoisedPLY() const {
            char buffer[1024] = {};
            sprintf(buffer, "%s/Scan_denoise.ply", directory.c_str());
            return std::string(buffer);
        }

        inline std::string getMainAxis() const {
            char buffer[1024] = {};
            sprintf(buffer, "%s/m_axis.txt", directory.c_str());
            return std::string(buffer);
        }

        inline std::string getPose(const int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
            sprintf(buffer, "%s/pose/pose%05d.txt", directory.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getOptimizedPose(const int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
//            sprintf(buffer, "%s/pose/pose%03d.txt", directory.c_str(), id);
            sprintf(buffer, "%s/pose/pose_optimized%05d.txt", directory.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getLines(const int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
            sprintf(buffer, "%s/lines/lines%05d.txt", directory.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getOpticalFlow_forward(const int id) const {
            CHECK_LT(id, getTotalNum() - 1);
            char buffer[1024] = {};
            sprintf(buffer, "%s/opticalflow/flow_forward%05d.jpg", directory.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getOpticalFlow_backward(const int id) const {
            CHECK_LT(id, getTotalNum());
            CHECK_GE(id, startid);
            char buffer[1024] = {};
            sprintf(buffer, "%s/opticalflow/flow_backward%05d.jpg", directory.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getBGFlow_forward(const int id)const{
            CHECK_LT(id, getTotalNum());
            CHECK_GE(id, startid);
            char buffer[1024] = {};
            sprintf(buffer, "%s/opticalflow/BGFlow_forward%05d.jpg", directory.c_str(), id + startid);
            return std::string(buffer);

        }

        inline std::string getBGFlow_backward(const int id)const{
            CHECK_LT(id, getTotalNum());
            CHECK_GE(id, startid);
            char buffer[1024] = {};
            sprintf(buffer, "%s/opticalflow/BGFlow_backward%05d.jpg", directory.c_str(), id + startid);
            return std::string(buffer);

        }

        inline std::string getConfidence3D(const int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
            sprintf(buffer, "%s/dynamic_confidence/dynamic_confidence_3D%05d.depth", directory.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getConfidence3DImage(const int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
            sprintf(buffer, "%s/dynamic_confidence/dynamic_confidence_3D%05d.jpg", directory.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getMarkResultPLY(const int id) const {
            char buffer[1024] = {};
            sprintf(buffer, "%s/dynamic_confidence/dynamic_confidence_3D%05d.ply", directory.c_str(), id);
            return std::string(buffer);
        }

        inline std::string getDynamicConfidence(int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
//            sprintf(buffer, "%s/dynamic_confidence/experimental_confidence3D%03d.confidence", directory.c_str(), id);
            sprintf(buffer, "%s/dynamic_confidence/confidence%05d.confidence", directory.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getDynamicConfidenceImage(int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
            sprintf(buffer, "%s/dynamic_confidence/confidence%05d.png", directory.c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getTimelineRGB() const {
            char buffer[1024] = {};
            sprintf(buffer, "%s/time_RGB.txt", directory.c_str());
            return std::string(buffer);
        }

        inline std::string getTimelinePointCloud() const {
            char buffer[1024] = {};
            sprintf(buffer, "%s/time_Point.txt", directory.c_str());
            return std::string(buffer);
        }

        inline std::string getSfmData() const {
            char buffer[1024] = {};
            sprintf(buffer, "%s/output_sfm/reconstruction_global/robust.json", getDirectory().c_str());
            return std::string(buffer);
        }

        inline std::string getSift(const int id) const {
            CHECK_LT(id, getTotalNum());
            char buffer[1024] = {};
            sprintf(buffer, "%s/sift/sift%05d.feature", getDirectory().c_str(), id + startid);
            return std::string(buffer);
        }

        inline std::string getQuadInitFile() const {
            char buffer[1024] = {};
            sprintf(buffer, "%s/corners_init.txt", getDirectory().c_str());
            return std::string(buffer);
        }

        inline std::string getQuadFinalFile() const {
            char buffer[1024] = {};
            sprintf(buffer, "%s/corners_final.txt", getDirectory().c_str());
            return std::string(buffer);
        }

        inline std::string getVideoTexture(const int track, const int frameid) const {
            char buffer[1024] = {};
            sprintf(buffer, "%s/video/video%03d_%05d.jpg", getDirectory().c_str(), track, frameid);
            return std::string(buffer);
        }

        inline std::string getDirectory() const {
            return directory;
        }


    private:
        const std::string imagePrefix;
        const int startid;
        const std::string directory;
        int framenum;
        int videonum;
        int pointcloudnum;
    };

}

#endif

