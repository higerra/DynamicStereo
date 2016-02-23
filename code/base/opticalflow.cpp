//
// Created by yanhang on 1/8/16.
//

#include "opticalflow.h"
#include "frame.h"
#include "file_io.h"
#include "colorwheel.h"

using namespace std;
using namespace Eigen;
using namespace cv;
namespace dynamic_rendering {

	void FlowFrame::init(const cv::Mat &img_){
		CHECK(!img_.empty()) << "Empty image!";
		CHECK_EQ(img_.channels(), 2);
		w = img_.cols;
		h = img_.rows;
		img.resize(w * h * 2);
		Mat dimg;
		img_.convertTo(dimg, CV_64F);
		const double* pDimg = (double*) dimg.data;
		for(auto i=0; i<img.size(); ++i)
			img[i] = pDimg[i];
	}

	void FlowFrame::init(const DImage &img_) {
		CHECK(!img_.IsEmpty()) << "Empty image!";
		CHECK_EQ(img_.nchannels(), 2);
		w = img_.width();
		h = img_.height();
		img.resize(w * h * 2);
		const double* pDimg = img_.data();
		for(auto i=0; i<img.size(); ++i)
			img[i] = pDimg[i];
	}

	void FlowFrame::setValue(const int x, const int y, const int c, const double v) {
		CHECK_GE(x, 0); CHECK_LT(x, width());
		CHECK_GE(y, 0); CHECK_LT(y, height());
		CHECK_GE(c, 0); CHECK_LT(c, 2);
		double *pImgdata = img.data();
		pImgdata[2 * (x + y * width()) + c] = v;
	}

	void FlowFrame::setValue(const int x, const int y, const Eigen::Vector2d &v) {
		setValue(x, y, 0, v[0]);
		setValue(x, y, 1, v[1]);
	}

	bool FlowFrame::readFlowFile(const std::string &path) {
		ifstream fin(path.c_str());
		if (!fin.is_open())
			return false;
		fin.read((char *)&w, sizeof(int));
		fin.read((char *)&h, sizeof(int));
		img.resize(w * h * 2, 0);
		fin.read((char* )img.data(), w * h * 2 * sizeof(double));
		fin.close();
		return true;
	}

	void FlowFrame::saveFlowFile(const std::string &path) const {
		CHECK_EQ(w*h*2, img.size());
		ofstream fout(path.c_str());
		CHECK(fout.is_open())<<"FlowFrame::saveFlowFile(): can not open file to write";
		fout.write((char*) &w, sizeof(int));
		fout.write((char*) &h, sizeof(int));
		fout.write((char*)img.data(), w * h * 2 * sizeof(double));
		fout.close();
	}

	void FlowEstimator::downSample(const cv::Mat &input, cv::Mat& output, const int nLevel) {
		CHECK_GT(nLevel, 0);
		vector<Mat> pyramid(nLevel);
		pyramid[0] = input.clone();
		for(size_t i=1; i<pyramid.size(); ++i)
			pyrDown(pyramid[i-1], pyramid[i]);
		output = pyramid.back().clone();
	}

	void FlowEstimatorCPU::estimate(const cv::Mat &img1, const cv::Mat &img2, FlowFrame &flow, const int nLevel) {
		CHECK_EQ(img1.size(), img2.size());
		Mat img1_small, img2_small;
		downSample(img1, img1_small, nLevel);
		downSample(img2, img2_small, nLevel);
		DImage dimg1, dimg2, dflow;
		//copy image data to DImage
		dimg1.allocate(img1_small.cols, img1_small.rows, img1_small.channels());
		dimg2.allocate(img2_small.cols, img2_small.rows, img2_small.channels());
		const uchar* pImg1 = img1_small.data;
		const uchar* pImg2 = img2_small.data;
		double* pDImg1 = dimg1.data();
		double* pDImg2 = dimg2.data();
		for(auto i=0; i<img1_small.cols * img1_small.rows * img1_small.channels(); ++i){
			pDImg1[i] = (double)pImg1[i] / 255.0;
			pDImg2[i] = (double)pImg2[i] / 255.0;
		}
		OpticalFlow::IsDisplay = false;
		OpticalFlow::ComputeOpticalFlow(dimg1, dimg2, dflow);
		flow.init(dflow);
	}

#ifdef USE_CUDA
	void FlowEstimatorGPU::estimate(const cv::Mat &img1, const cv::Mat &img2, FlowFrame &flow, const int nLevel) {
		CHECK(!img1.empty());
		CHECK_EQ(img1.channels(), 1);
		CHECK_EQ(img2.channels(), 1);
		CHECK_EQ(img1.size, img2.size);
		Mat img1_small, img2_small;
		downSample(img1, img1_small, nLevel);
		downSample(img2, img2_small, nLevel);
		cuda::GpuMat img1GPU(img1_small);
		cuda::GpuMat img2GPU(img2_small);
		cuda::GpuMat flowGPU(img1_small.size(), CV_32FC2);
		cuda::GpuMat img1GPUf, img2GPUf;
		img1GPU.convertTo(img1GPUf, CV_32F, 1.0/255.0);
		img2GPU.convertTo(img2GPUf, CV_32F, 1.0/255.0);
		brox->calc(img1GPUf, img2GPUf, flowGPU);
		Mat flowCPU(flowGPU.clone());
		flow.init(flowCPU);
	}
#endif


	namespace flow_util {
		void interpolateFlow(const FlowFrame &input, FlowFrame &output, const std::vector<bool> &mask, const bool fillHole) {
			const int w = input.width();
			const int h = input.height();
			const int kPix = w * h;
			vector<double> fx(kPix), fy(kPix);
			const double *pInputFlow = input.data().data();
			for (auto i = 0; i < kPix; ++i) {
				fx[i] = pInputFlow[2 * i];
				fy[i] = pInputFlow[2 * i + 1];
			}
			if (fillHole) {
				fillHoleLinear(fx, w, h, mask);
				fillHoleLinear(fy, w, h, mask);
			} else {
				possionSmooth(fx, w, h, mask);
				possionSmooth(fy, w, h, mask);
			}
			output.clear();
			output.allocate(w, h);
			double *pOutputFlow = output.data().data();
			for (auto i = 0; i < kPix; ++i) {
				pOutputFlow[2 * i] = fx[i];
				pOutputFlow[2 * i + 1] = fy[i];
			}
		}

		void warpImage(const cv::Mat &input, cv::Mat &output, const FlowFrame &flow) {
			if (input.cols != flow.width() || input.rows != flow.height())
				throw runtime_error("FlowFrame::warpImage(): dimensions don't match");
			output = Mat(input.rows, input.cols, CV_8UC3, Scalar(0, 0, 0));
			uchar *pOut = output.data;
			const double *pFlow = flow.data().data();
			for (auto y = 0; y < output.rows; ++y) {
				for (auto x = 0; x < output.cols; ++x) {
					const int ind = y * output.cols + x;
					Vector2d fv(pFlow[2 * ind], pFlow[2 * ind + 1]);
					Vector2d loc((double) x + fv[0], (double) y + fv[1]);
					int xl = floor(loc[0]), yl = floor(loc[1]);
					int xh = round(loc[0] + 0.5), yh = round(loc[1] + 0.5);
					//printf("%d,%d,%d,%d\n", xl, yl, xh, yh);
					if (xl <= 0 || yl <= 0 || xh >= output.cols - 1 || yh >= output.rows - 1)
						continue;
					Vector3d pix = interpolation_util::bilinear<unsigned char, 3>(input.data, input.cols, input.rows,
					                                                              loc);
					pOut[ind * 3] = (uchar) pix[0];
					pOut[ind * 3 + 1] = (uchar) pix[1];
					pOut[ind * 3 + 2] = (uchar) pix[2];
				}
			}
		}

		void visualizeFlow(const FlowFrame &flow, cv::Mat &flowimg) {
			const double *pFlow = flow.data().data();
			CHECK(pFlow) << "Empty flow";
			flowimg = Mat(flow.height(), flow.width(), CV_8UC3, Scalar(0, 0, 0));
			uchar *pImg = flowimg.data;
			for (int i = 0; i < flow.width() * flow.height(); ++i) {
				Vector2d fv(pFlow[2 * i], pFlow[2 * i + 1]);
				Vector3d RGB = ColorWheel::instance()->computeColor(fv);
				pImg[3 * i] = (uchar) RGB[0];
				pImg[3 * i + 1] = (uchar) RGB[1];
				pImg[3 * i + 2] = (uchar) RGB[2];
			}
		}

		bool trackPoint(const Eigen::Vector2d &loc, const std::vector<FlowFrame> &flow, const int src,
		                const int tgt, Eigen::Vector2d &res) {
			res = loc;
			if (src == tgt)
				return true;
			CHECK_GE(src,0);
			CHECK_GE(tgt,0);
			CHECK_LT(src, flow.size());
			CHECK_LT(src, flow.size());
			int dir = src < tgt ? 1 : -1;
			for (auto i = src; i != tgt; i += dir) {
				Vector2d fv = flow[i].getFlowAt(res);
				res += fv;
				if (!flow[i].isInsideFlowImage(res))
					return false;
			}
			return true;
		}

		Mat drawFlowDot(const Frame &frame, const Vector2d &loc, const int nlevel) {
			vector<Mat> pyramid(nlevel);
			pyramid[0] = frame.getImage().clone();
			for (int i = 1; i < nlevel; ++i)
				pyrDown(pyramid[i - 1], pyramid[i]);
			circle(pyramid.back(), cv::Point(loc[0], loc[1]), 3, Scalar(0, 255, 255), 3);
			return pyramid.back();
		}

		void verifyFlow(const std::vector<FlowFrame> &flow_forward,
		                const std::vector<FlowFrame> &flow_backward,
		                const std::vector<cv::Mat> &frames,
		                list<cv::Mat> &verifyimg,
		                const int fid, Vector2d loc) {
			if (frames.size() < flow_forward.size())
				throw runtime_error("verifyFlow(): frames.size() < flow.getFrameCount()");
			if (flow_forward.size() != flow_backward.size())
				throw runtime_error("verifyFlow(): flow_forward.size() != flow_backward.size()");
			if (flow_forward.empty())
				throw runtime_error("verifyFlow(): empty flow");
			int frameid = fid;
			const int width = flow_forward.front().width();
			const int height = flow_forward.front().height();
			Vector2d ori_loc = loc;
			//forward
			cout << "Verifying optical flow..." << endl;
			while (true) {
				cout << frameid << ' ';
				if (frameid == frames.size() - 1)
					break;
				if (!flow_forward[frameid].isInsideFlowImage(loc))
					break;
				verifyimg.push_back(drawFlowDot(frames[frameid], loc, 1));
				Vector2d dloc = flow_forward[frameid].getFlowAt(loc);
				loc += dloc;
				frameid++;
			}

			frameid = fid;
			loc = ori_loc;
			bool firstflag = true;
			//backward
			while (true) {
				if (frameid == 0)
					break;
				if (!flow_backward[frameid].isInsideFlowImage(loc))
					break;
				if (!firstflag)
					verifyimg.push_front(drawFlowDot(frames[frameid], loc, 1));

				Vector2d dloc = flow_backward[frameid].getFlowAt(loc);
				loc += dloc;
				frameid--;
				firstflag = false;
			}
			cout << endl;
		}


		void resizeFlow(const FlowFrame &input, FlowFrame &output, const double ratio, const bool rescale) {
			Vector2i dsize(input.width() / ratio, input.height() / ratio);
			resizeFlow(input, output, dsize, rescale);
		}

		void resizeFlow(const FlowFrame &input, FlowFrame &output, const Eigen::Vector2i &dsize, const bool rescale) {
			output.clear();
			output.allocate(dsize[0], dsize[1]);
			const double ratiox = (double) dsize[0] / (double) input.width();
			const double ratioy = (double) dsize[1] / (double) input.height();
			double *pOutput = output.data().data();
			for (auto x = 0; x < output.width(); ++x) {
				for (auto y = 0; y < output.height(); ++y) {
					Vector2d loc(floor(x / ratiox), floor(y / ratioy));
					Vector2d fv = input.getFlowAt(loc);
					if (rescale) {
						fv[0] *= ratiox;
						fv[1] *= ratioy;
					}
					pOutput[(y * dsize[0] + x) * 2] = fv[0];
					pOutput[(y * dsize[0] + x) * 2 + 1] = fv[1];
				}
			}
		}

		void computeMissingFlow(const FileIO &file_io, const int nlevel) {
			const int framenum = file_io.getTotalNum();
			CHECK_GT(framenum, 1);
			Mat tempMat = imread(file_io.getImage(0));
			const int w = tempMat.cols;
			const int h = tempMat.rows;
			int downsample = 1;
			for(auto i=1; i<nlevel; ++i)
				downsample *= 2;
			const int sw = w / downsample;
			const int sh = h / downsample;
			char buffer[1024] = {};

			shared_ptr<FlowEstimator> flowEstimator;
			int colorType;
#ifdef USE_CUDA
            flowEstimator = shared_ptr<FlowEstimator>(new FlowEstimatorGPU());
			colorType = IMREAD_GRAYSCALE;

#else
            flowEstimator = shared_ptr<FlowEstimator>(new FlowEstimatorCPU());
			colorType = IMREAD_ANYCOLOR;
#endif
			//forward
			for (int i = 0; i < framenum - 1; ++i) {
				ifstream fin(file_io.getOpticalFlow_forward(i).c_str());
				bool recompute = !fin.is_open();
				if(!recompute) {
					FlowFrame temp;
					temp.readFlowFile(file_io.getOpticalFlow_forward(i));
					if (temp.width() != sw || temp.height() != sh)
						recompute = true;
				}
				if (recompute) {
					printf("Computing forward opticalflow for frame %d\n", i);
					Mat img1 = imread(file_io.getImage(i), colorType);
					Mat img2 = imread(file_io.getImage(i+1), colorType);
					FlowFrame flow;
					flowEstimator->estimate(img1, img2, flow, nlevel);
					flow.saveFlowFile(file_io.getOpticalFlow_forward(i));
					Mat flowVis;
					flow_util::visualizeFlow(flow, flowVis);
					sprintf(buffer, "%s/opticalflow/flowImage_forward%03d.jpg", file_io.getDirectory().c_str(), i);
					imwrite(buffer, flowVis);
				}
				fin.close();
			}

			//backward
			for (int i = 1; i < framenum; ++i) {
				ifstream fin(file_io.getOpticalFlow_backward(i).c_str());
				bool recompute = !fin.is_open();
				if(!recompute) {
					FlowFrame temp;
					temp.readFlowFile(file_io.getOpticalFlow_backward(i));
					if (temp.width() != sw || temp.height() != sh)
						recompute = true;
				}
				if (recompute) {
					printf("Computing back opticalflow for frame %d\n", i);
					Mat img1 = imread(file_io.getImage(i), colorType);
					Mat img2 = imread(file_io.getImage(i-1), colorType);
					FlowFrame flow;
					flowEstimator->estimate(img1, img2, flow, nlevel);
					flow.saveFlowFile(file_io.getOpticalFlow_backward(i));
					Mat flowVis;
					flow_util::visualizeFlow(flow, flowVis);
					sprintf(buffer, "%s/opticalflow/flowImage_backward%03d.jpg", file_io.getDirectory().c_str(), i);
					imwrite(buffer, flowVis);
				}
				fin.close();

			}
		}
	}// namespace flow_util

}//namespace dynamic_rendering