#include "frame.h"
#include "file_io.h"
#include "quad.h"
#include <numeric>
#include <assert.h>
#include "utility.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_rendering{
	void Frame::initialize(const FileIO& file_io, int id, bool no_camera){
		if(id > file_io.getTotalNum()){
			cerr<<"Frame::initialize: invalid framenum!"<<endl;
			exit(-1);
		}
		image = imread(file_io.getImage(id));

		cv::Size dsize(image.cols / depthratio, image.rows / depthratio);
		cv::resize(image, downsampledImg, dsize);

		width = image.cols;
		height = image.rows;
		if(no_camera)
			return;
		camera.initialize(file_io, id);
		if(!camera.isIntrinsicSet())
			camera.setIntrinsic(1045.67,1045.64,639.489,350.282, 0.2165, -0.6345,0.6229);
	}

	Frame::Frame(const FileIO& file_io, int id):camera(file_io, id){
		initialize(file_io, id);
	}

	Frame::Frame(const Mat& image_): image(image_.clone()), depth(), camera(){
		height = image.rows;
		width = image.cols;
	}

	const double Frame::depthratio = 10.0;

	const Vector3d Frame::getColor(Vector2d pt, const bool downsample) const{
		if(!isValidRGB(pt))
			throw runtime_error("Frame::getColor(): out of bound!");

		const Mat* desimg;
		if(downsample){
			desimg = &downsampledImg;
			pt = pt / depthratio;
		}else
			desimg = &image;

		return interpolation_util::bilinear<uchar, 3>(desimg->data, desimg->cols, desimg->rows, pt);
//		int xl = floor(pt[0]), xh = round(pt[0]+0.5);
//		int yl = floor(pt[1]), yh = round(pt[1]+0.5);
//		double lm=pt[0]-(double)xl, rm=(double)xh-pt[0];
//		double tm=pt[1]-(double)yl, bm=(double)yh-pt[1];
//		const uchar* pixdata = desimg->data;
//		const int w = getWidth();
//		const int nc = desimg->channels();
//		Vector3d v0((double)pixdata[nc * (xl+yl*w)], (double)pixdata[nc * (xl+yl*w)+1], (double)pixdata[nc * (xl+yl*w)+2]);
//		Vector3d v1((double)pixdata[nc * (xh+yl*w)], (double)pixdata[nc * (xh+yl*w)+1], (double)pixdata[nc * (xh+yl*w)+2]);
//		Vector3d v2((double)pixdata[nc * (xh+yh*w)], (double)pixdata[nc * (xh+yh*w)+1], (double)pixdata[nc * (xh+yh*w)+2]);
//		Vector3d v3((double)pixdata[nc * (xl+yh*w)], (double)pixdata[nc * (xl+yh*w)+1], (double)pixdata[nc * (xl+yh*w)+2]);
//		return v0*rm*bm + v1*lm*bm + v2*lm*tm + v3*rm*tm;
	}

	bool Frame::isVisible(const Vector3d &pt)const{
		const double margin = 0.4 * depth.getAverageDepth();
		Vector4d localpt = getCamera().getExtrinsic().inverse() * Vector4d(pt[0], pt[1], pt[2], 1.0);
		double curdepth = localpt[2];
		Vector4d imgpt = getCamera().getIntrinsic() * localpt;
		if(imgpt[2] != 0){
			imgpt[0] /= imgpt[2];
			imgpt[1] /= imgpt[2];
		}
		else
			return false;
		Vector2d depthpix = RGBToDepth(Vector2d(imgpt[0], imgpt[1]));

		if((!isValidDepth(depthpix)) || (!isValidRGB(Vector2d(imgpt[0], imgpt[1]))) || curdepth < 0)
			return false;
		double imgdepth = depth.getDepthAt(depthpix);
		if(curdepth > imgdepth+margin)
			return false;
		return true;
	}

	void constructPyramid(const vector<Frame>& frames,
	                      vector<vector<Frame> >& pyramid,
	                      const int max_level){
		pyramid.resize(1);
		for(int i=0; i<frames.size(); i++){
			Frame curframe(frames[i].getImage());
			pyramid[0].push_back(curframe);
		}
		for(int level=1; level<max_level; ++level){
			vector<Frame> curlevel;
			for(int i=0; i<frames.size(); i++){
				Mat src = pyramid[level-1][i].getImage().clone();
				Mat dst;
				pyrDown(src, dst);
				Frame curframe(dst);
				curlevel.push_back(dst);
			}
			pyramid.push_back(curlevel);
		}
	}
}//namespace dynamic_rendering

