//
// Created by yanhang on 4/7/16.
//

#include "dynamic_confidence.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_stereo{
    void DynamicConfidence::init() {
        cout << "Reading reconstruction" << endl;
        CHECK(theia::ReadReconstruction(file_io.getReconstruction(), &reconstruction)) << "Can not open reconstruction file";
        CHECK_EQ(reconstruction.NumViews(), file_io.getTotalNum());
        const vector<theia::ViewId>& vids = reconstruction.ViewIds();
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

    void DynamicConfidence::run(const int anchor, Depth& confidence) {
        const int framenum = file_io.getTotalNum();
        CHECK_LT(anchor, framenum);
        const int startid = anchor - max_tWindow/2 >= 0 ? anchor-max_tWindow/2 : 0;
        const int endid = anchor + max_tWindow/2 < framenum ? anchor+max_tWindow/2:framenum-1;

        vector<FlowFrame> flow_forward((size_t)endid-startid+1);
        vector<FlowFrame> flow_backward((size_t)endid-startid+1);

        cout << "Reading optical flow..." << endl;
        for(auto i=startid; i<=endid; ++i){
            cout << '.' << flush;
            if(i < file_io.getTotalNum() - 1)
                flow_forward[i-startid].readFlowFile(file_io.getOpticalFlow_forward(i));
            if(i > 0)
                flow_backward[i-startid].readFlowFile(file_io.getOpticalFlow_backward(i));
        }
        cout << endl;
        const int width = flow_forward[0].width();
        const int height = flow_forward[0].height();

        confidence.initialize(width, height, -1);
        cout << "Computing confidence..." << endl;


    }
}//namespace dynamic_stereo