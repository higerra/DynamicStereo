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

	    //prepare output
	    output.resize(input.size());
	    for(auto& o: output)
		    o = Mat(height, width, CV_8UC3, Scalar::all(0));
	    const double huber_theta = 10;
        auto threadFun = [&](const int tid, const int num_thread){
	        vector<vector<double> > DP(N);
	        for(auto &d: DP)
		        d.resize(256,0.0);
	        vector<vector<int> > backTrack(N);
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
				        //start DP
				        for(auto p=0; p<256; ++p)
					        DP[0][p] = math_util::huberNorm((double)inputPtr[0][channel*(y*width+x)+c] - (double)p, huber_theta);
				        for(auto v=1; v<input.size(); ++v){
					        for(auto p=0; p<256; ++p){
						        DP[v][p] = numeric_limits<double>::max();
						        double mdata = math_util::huberNorm((double)inputPtr[v][channel*(y*width+x)+c]-(double)p, huber_theta);
						        for(auto pf=0; pf<256; ++pf){
							        double curv = DP[v-1][pf] + mdata + math_util::huberNorm((double)pf-(double)p, huber_theta);
							        if(curv < DP[v][p]){
								        DP[v][p] = curv;
								        backTrack[v][p] = pf;
							        }
						        }
					        }
				        }
				        //back track
				        
			        }
		        }
	        }
        };


    }

}//namespace dynamic_stereo