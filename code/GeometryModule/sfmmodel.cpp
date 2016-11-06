//
// Created by yanhang on 11/6/16.
//

#include "sfmmodel.h"
#include <fstream>

using namespace theia;
using namespace std;

namespace DynamicStereo{
    void SfMModel::InitFromTheia(const std::string& path){
        CHECK(theia::ReadReconstruction(path, &reconstruction)) << "Can not open reconstruction file";
        const std::vector<theia::ViewId>& vids = reconstruction.ViewIds();
        orderedId.resize(vids.size());
        for(auto i=0; i<vids.size(); ++i) {
            const theia::View* v = reconstruction.View(vids[i]);
            std::string nstr = v->Name().substr(5,5);
            int idx = atoi(nstr.c_str());
            orderedId[i] = IdPair(idx, vids[i]);
        }
        std::sort(orderedId.begin(), orderedId.end(),
                  [](const std::pair<int, theia::ViewId>& v1, const std::pair<int, theia::ViewId>& v2){return v1.first < v2.first;});
    }

    void SfMModel::InitFromNVM(const std::string& path){
        ifstream in(path.c_str());
        string token;
        in >> token;
        CHECK_EQ(token, "NVM_V3");
        int ncam = 0, npoint = 0, nproj = 0;

        in >> ncam;

    }


}