//
// Created by yanhang on 11/7/16.
//

#include "stereomodel.h"

namespace dynamic_stereo{
    void SfMModel::init(const std::string& path){
        CHECK(theia::ReadReconstruction(path, &reconstruction)) << "Can not open reconstruction file";
        const std::vector<theia::ViewId>& vids = reconstruction.ViewIds();
        orderedId.resize(vids.size());
        for(auto i=0; i<vids.size(); ++i) {
            const theia::View* v = reconstruction.View(vids[i]);
            std::string nstr = v->Name().substr(5,5);
            int idx = atoi(nstr.c_str());
            orderedId[i] = IdPair(idx, vids[i]);
//                Eigen::Matrix3d intrinsic;
//                v->Camera().GetCalibrationMatrix(&intrinsic);
//				std::cout << "Intrinsic" << std::endl << intrinsic << std::endl;
        }
        std::sort(orderedId.begin(), orderedId.end(),
                  [](const std::pair<int, theia::ViewId>& v1, const std::pair<int, theia::ViewId>& v2){return v1.first < v2.first;});
    }
}