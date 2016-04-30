//
// Created by yanhang on 4/29/16.
//

#include "dynamicregularizer.h"
#include "../base/utility.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    void dynamicRegularization(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output){
	    CHECK(!input.empty());
	    const int width = input[0].cols;
	    const int height = input[0].rows;
	    const int channel = input[0].channels();
	    const int N = (int)input.size();

        vector<uchar *> inputPtr(input.size(), NULL);
	    for(auto i=0; i<input.size(); ++i)
		    inputPtr[i] = input[i].data;

	    const double huber_theta = 10;
        auto threadFun = [&](const int tid, const int num_thread){
	        vector<vector<float> > DP(N);
	        for(auto &d: DP)
		        d.resize(256,0.0);
	        vector<vector<unsigned int> > backTrack(N);
	        for(auto& b: backTrack)
		        b.resize(256, 0);

	        for(auto y=tid; y<height; y+=num_thread) {
		        for (auto x = 0; x < width; ++x) {
			        for (auto c = 0; c < channel; ++c) {
				        //reinitialize tables
				        for (auto &d: DP) {
					        for (auto &dd: d)
						        dd = 0.0;
				        }
				        for (auto &b: backTrack) {
					        for (auto &bb: b)
						        bb = 0;
				        }
				        for(auto p=0; p<256; ++p)
					        DP[0][p] = (float)math_util::huberNorm((double)inputPtr[0][channel*(y*width+x)+c] - (double)p, huber_theta);
				        for(auto v=1; v<input.size(); ++v){

				        }
			        }
		        }
	        }
        };


    }

}//namespace dynamic_stereo