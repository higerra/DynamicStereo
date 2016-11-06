//
// Created by yanhang on 11/6/16.
//

#ifndef DYNAMICSTEREO_SFMMODEL_H
#define DYNAMICSTEREO_SFMMODEL_H

#include <string>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <memory>

#include <Eigen/Eigen>
#include <theia/theia.h>
#include <glog/logging.h>

namespace DynamicStereo{
    class SfMModel{
        void InitFromTheia(const std::string& path);

        void InitFromNVM(const std::string& path);

        typedef std::pair<int, theia::ViewId> IdPair;

        inline const theia::Camera& getCamera(const int vid) const{
            CHECK_LT(vid, orderedId.size());
            return reconstruction.View(orderedId[vid].second)->Camera();
        }

        inline const theia::View* getView(const int vid) const{
            CHECK_LT(vid, orderedId.size());
            return reconstruction.View(orderedId[vid].second);
        }

        inline double warpPoint(const int vid1, const Eigen::Vector2d& pt1, const double depth, const int vid2, Eigen::Vector2d& imgpt2) const{
            CHECK_LT(vid1, orderedId.size());
            CHECK_LT(vid2, orderedId.size());
            const theia::Camera& cam1 = getCamera(vid1);
            const theia::Camera& cam2 = getCamera(vid2);
            Eigen::Vector3d spt = cam1.GetPosition() + cam1.PixelToUnitDepthRay(pt1) * depth;
            double depth2 = cam2.ProjectPoint(spt.homogeneous(), &imgpt2);
            return depth2;
        }
    private:
        theia::Reconstruction reconstruction;
        std::vector<IdPair> orderedId;
        std::vector<theia::Camera> cameras;
        std::vector<theia::View> views;
    };

}


#endif //DYNAMICSTEREO_SFMMODEL_H
