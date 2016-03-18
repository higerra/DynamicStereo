//
// Created by yanhang on 3/4/16.
//

#include "local_matcher.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace local_matcher {
    void samplePatch(const cv::Mat &img, const Vector2d &loc, const int pR, std::vector<double> &pix) {
        const int w = img.cols;
        const int h = img.rows;
        pix.resize((size_t) 3 * (2 * pR + 1) * (2 * pR + 1));
        int index = 0;
        for (int dx = -1 * pR; dx <= pR; ++dx) {
            for (int dy = -1 * pR; dy <= pR; ++dy, ++index) {
                Vector2d curloc(loc[0] + dx, loc[1] + dy);
                if (curloc[0] < 0 || curloc[1] < 0 || curloc[0] >= w - 1 || curloc[1] >= h - 1) {
                    pix[index * 3] = -1;
                    pix[index * 3 + 1] = -1;
                    pix[index * 3 + 2] = -1;
                } else {
                    Vector3d pv = interpolation_util::bilinear<uchar, 3>(img.data, w, h, curloc);
                    pix[index * 3] = pv[0];
                    pix[index * 3 + 1] = pv[1];
                    pix[index * 3 + 2] = pv[2];
                }
            }
        }
    }

    void getSSDArray(const vector<vector<double> > &patches, const int refId, vector<double> &mCost) {
        const vector<double> &pRef = patches[refId];

        for (auto i = 0; i < patches.size(); ++i) {
            if (i == refId)
                continue;
            vector<double> p1, p2;
            for (auto j = 0; j < pRef.size(); ++j) {
                if (pRef[j] >= 0 && patches[i][j] >= 0) {
                    p1.push_back(pRef[j]);
                    p2.push_back(patches[i][j]);
                }
            }
            if (p1.size() < pRef.size() / 2)
                continue;
            double ssd = 0.0;
            for (auto j = 0; j < p1.size(); ++j)
                ssd += (p1[j] - p2[j]) * (p1[j] - p2[j]);
            mCost.push_back(ssd / (double) p1.size());
        }
    }

    void getNCCArray(const vector<vector<double> >& patches, const int refId, vector<double>& mCost){
        const vector<double> &pRef = patches[refId];
        for(auto i=0; i<patches.size(); ++i){
            if(i == refId)
                continue;
            vector<double> p1, p2;
            for (auto j = 0; j < pRef.size(); ++j) {
                if (pRef[j] >= 0 && patches[i][j] >= 0) {
                    p1.push_back(pRef[j]);
                    p2.push_back(patches[i][j]);
                }
            }
            if (p1.size() < pRef.size() / 2)
                continue;
            mCost.push_back(math_util::normalizedCrossCorrelation(p1,p2));
        }
    }

    double sumMatchingCost(const vector<vector<double> > &patches, const int refId) {
        CHECK_GE(refId, 0);
        CHECK_LT(refId, patches.size());
        const double theta = 90;
        auto phid = [theta](const double v) {
            return -1 * std::log2(1 + std::exp(-1 * v / theta));
        };
        vector<double> mCost;
        getSSDArray(patches, refId, mCost);
        //if the patch is not visible in >50% frames, assign large penalty.
        if(mCost.empty())
            return 1;
        if (mCost.size() < 2)
            return 1;
        if (mCost.size() == 2)
            return std::min(phid(mCost[0]), phid(mCost[1]));
        //sum of best half
        //sort(mCost.begin(), mCost.end());
        const size_t kth = mCost.size();
        double res = 0.0;
        for (auto i = 0; i < kth; ++i) {
            res += phid(mCost[i]);
        }
        return res / (double) kth;
    }

    double medianMatchingCost(const vector<vector<double> > &patches, const int refId) {
        CHECK_GE(refId, 0);
        CHECK_LT(refId, patches.size());
        vector<double> mCost;
        getNCCArray(patches, refId, mCost);
        //if the patch is not visible in >50% frames, assign large penalty.
        if (mCost.size() < patches.size() / 2)
            return -1;
        size_t kth = mCost.size() / 2;
        nth_element(mCost.begin(), mCost.begin() + kth, mCost.end());
        return mCost[kth];
    }
}
