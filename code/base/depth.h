#ifndef depth_h
#define depth_h
#include <Eigen/Eigen>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <limits>
namespace dynamic_stereo{
    class Depth{
    public:
        Depth(): depthwidth(0), depthheight(0), average_depth(0),median_depth(0), depth_var(0),max_depth(-1),min_depth(std::numeric_limits<double>::max()), statics_computed(false){}
        Depth(int width, int height, const double v=-1):
                average_depth(0),median_depth(0), depth_var(0),max_depth(-1),min_depth(std::numeric_limits<double>::max()), statics_computed(false){
            initialize(width, height, v);
        }
        void initialize(int width, int height, const double v = -1);

        inline int getWidth()const {return depthwidth;}
        inline int getHeight()const {return depthheight;}
        inline double getAverageDepth()const {return average_depth;}
        inline double getDepthVariance()const {return depth_var;}
        inline double getMedianDepth()const {return median_depth;}
        inline double getMaxDepth() const {return max_depth;}
        inline double getMinDepth() const {return min_depth;};
        inline std::vector<double>& getRawData(){return data;}
        inline const std::vector<double>& getRawData()const {return data;}
        inline void getRawData(std::vector<double>&array) const{
            array.insert(array.end(), data.begin(), data.end());
        }

        /*!
         * Set depth with bilinear splatting
         * @param pix Pixel location. Can be fractinal
         * @param v New depth value
         * @param over_write If set to true, original value will be discarded.
         */
        void setDepthAt(const Eigen::Vector2d& pix, double v, const bool over_write = false);

        /*!
         * Set depth with integer coordinate
         * @param x,y Integer pixel coordinate
         * @param v
         */
        inline void setDepthAtInt(const int x, const int y, const double v){
            if(!(x>=0 && x<getWidth() && y>=0 && y<getHeight()))
                return;
            data[x+y*getWidth()] = v;
        }

        inline void setDepthAtInd(const int ind, const double v){
            CHECK_GE(ind, 0);
            CHECK_LT(ind, data.size());
            data[ind] = v;
        }

        inline double operator[] (int idx) const{
            CHECK_GE(idx, 0);
            CHECK_LT(idx, data.size());
            return data[idx];
        }

        inline double& operator[] (int idx){
            CHECK_GE(idx, 0);
            CHECK_LT(idx, data.size());
            return data[idx];
        }

        inline double operator() (int x, int y) const{
            CHECK_GE(x, 0);
            CHECK_GE(y, 0);
            CHECK_LT(x, getWidth());
            CHECK_LT(y, getHeight());
            int idx = y * getWidth() + x;
            return this->operator[](idx);
        }
        inline double& operator()(int x, int y){
            CHECK_GE(x, 0);
            CHECK_GE(y, 0);
            CHECK_LT(x, getWidth());
            CHECK_LT(y, getHeight());
            int idx = y * getWidth() + x;
            return data[idx];
        }


        double getDepthAt(const Eigen::Vector2d& loc)const;

        inline double getDepthAtInt(int x, int y) const{
            return this->operator()(x,y);
        }
        inline double getDepthAtInd(int ind)const{
            return this->operator[](ind);
        }
        inline bool insideDepth(int x,int y)const{
            return (x>=0 && x<getWidth() && y>=0 && y<getHeight());
        }

        inline bool insideDepth(const Eigen::Vector2d& loc) const{
            return (loc[0] >= 0 && loc[1] >= 0 && loc[0] < getWidth() - 1 && loc[1] < getHeight() - 1);
        }

        inline double getWeightAt(const int x, const int y)const{
            CHECK_GE(x,0);
            CHECK_GE(y,0);
            CHECK_LT(x, getWidth());
            CHECK_LT(y, getHeight());
            return weight[x + y*getWidth()];
        }
        inline void setWeightAt(const int ind, const double v){
            CHECK_LT(ind, weight.size());
            CHECK_GE(ind, 0);
            weight[ind] = v;
        }

        void updateStatics() const;
        inline bool is_statics_computed()const {return statics_computed;}
        void fillhole();
        void fillholeAndSmooth(const double wSmooth = 1.0);

        //if amp == -1, the depth will be rescaled to 0~255
        void saveImage(const std::string& filename, const double amp = 1) const;
        void saveDepthFile(const std::string& filename) const;
        bool readDepthFromFile(const std::string& filename);

    private:
        int depthwidth;
        int depthheight;
        mutable double average_depth;
        mutable double median_depth;
        mutable double depth_var;
        mutable double min_depth;
        mutable double max_depth;

        mutable bool statics_computed;
        std::vector<double>data;
        std::vector<double>weight;
    };

    void fillHoleLinear(std::vector<double>&data, const int width, const int height, const std::vector<bool>& mask);
    void possionSmooth(std::vector<double>& data, const int width, const int height, const std::vector<bool>& mask, const double kDataCost = 1.0);
}

#endif






