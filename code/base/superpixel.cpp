#include "superpixel.h"
#include "frame.h"
#include "depth.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace dynamic_rendering{
	void SuperPixelHelper::MatToImagebuffer(const Mat& image, vector<unsigned int>&imagebuffer){
		if(!image.data){
			cout << "invlid image"<<endl;
			exit(-1);
		}
		const int imgheight = image.rows;
		const int imgwidth = image.cols;
		imagebuffer.clear();
		imagebuffer.resize(imgheight * imgwidth);
		for(int y=0;y<imgheight;y++){
			for(int x=0;x<imgwidth;x++){
				Vec3b curpix = image.at<Vec3b>(y,x);
				int ind = y*imgwidth + x;
				imagebuffer[ind] = (unsigned int)255*256*256*256 + (unsigned int)curpix[0]*256*256 + (unsigned int)curpix[1]*256 + (unsigned int)curpix[2];
			}
		}
	}


	void SuperPixelHelper::ImagebufferToMat(const vector <unsigned int>&imagebuffer,const int imgwidth,const int imgheight,  Mat& image){
		if(imagebuffer.size() != imgwidth * imgheight){
			cout << "Sizes don't agree!"<<endl;
			exit(-1);
		}
		image.release();
		image = Mat(imgheight,imgwidth,CV_8UC3);
		for(int y=0;y<imgheight;y++){
			for(int x=0;x<imgwidth;x++){
				Vec3b curpix;
				curpix[0] = imagebuffer[y*imgwidth+x] >> 16 & 0xff;
				curpix[1] = imagebuffer[y*imgwidth+x] >> 8 & 0xff;
				curpix[2] = imagebuffer[y*imgwidth+x] & 0xff;
				image.at<Vec3b>(y,x) = curpix;
			}
		}
	}

	void SuperPixelHelper::computeSuperpixel(const FileIO& file_io, const int id, const Frame& frame, const double ratio,const int num,  vector<int> &labels, int &num_labels){
		SLIC slic;
		char buffer[100];
		cv::Size dsize(frame.getWidth()*ratio, frame.getHeight()*ratio);
		labels.resize(dsize.width * dsize.height);

		Mat curimg = frame.getImage();

		Mat resizedimg(dsize, CV_8UC3);
		resize(curimg, resizedimg, dsize);
		cvtColor(resizedimg, resizedimg, CV_BGR2RGB);
		vector<unsigned int>imagebuffer;
		MatToImagebuffer(resizedimg, imagebuffer);

		slic.PerformSLICO_ForGivenK(&imagebuffer[0], dsize.width, dsize.height, &labels[0], num_labels, num, 0.0);
		slic.DrawContoursAroundSegments(&imagebuffer[0], &labels[0], dsize.width, dsize.height,255);
		Mat out;
		ImagebufferToMat(imagebuffer, dsize.width, dsize.height, out);
		sprintf(buffer, "%s/superpixel/image%03d.png", file_io.getDirectory().c_str(), id);
		imwrite(buffer, out);
		slic.SaveSuperpixelLabels(&labels[0], dsize.width, dsize.height, num_labels," ", file_io.getSuperpixel(id));

	}

	bool SuperPixelHelper::loadSuperpixel(const FileIO& file_io, const int id, const int pixnum, std::vector<int>&labels, int &num_labels){
		ifstream fin(file_io.getSuperpixel(id).c_str());
		if(!fin.is_open())
			return false;
		fin.read((char*)&num_labels, sizeof(int));
		for(int pixid=0; pixid<pixnum; pixid++){
			int temp;
			fin.read((char*)&temp, sizeof(int));
			labels.push_back(temp);
		}
		fin.close();
		return true;
	}

	void SuperPixelHelper::pairSuperpixel(const vector<int>&labels, const int num_labels, int width, int height, vector<vector<int> >&pairmap){
		//four connectivities
		pairmap.resize(num_labels);

		for(int y=0; y<height-1; y++){
			for(int x=0; x<width-1; x++){
				int label1 = labels[y*width+x]; //origin
				int label2 = labels[(y+1)*width+x]; //down
				int label3 = labels[y*width+x+1]; //right
				int minlabelx = std::min(label1,label3);
				int maxlabelx = std::max(label1,label3);
				int minlabely = std::min(label1,label2);
				int maxlabely = std::max(label1,label2);

				if(label1 != label3){
					bool is_found = false;
					for(int i=0; i<pairmap[minlabelx].size(); i++){
						if(pairmap[minlabelx][i] == maxlabelx)
							is_found = true;
					}
					if(!is_found){
						pairmap[minlabelx].push_back(maxlabelx);
						pairmap[maxlabelx].push_back(minlabelx);
					}
				}

				if(label1 != label2){
					bool is_found = false;
					for(int i=0; i<pairmap[minlabely].size(); i++){
						if(pairmap[minlabely][i] == maxlabely)
							is_found = true;
					}
					if(!is_found){
						pairmap[minlabely].push_back(maxlabely);
						pairmap[maxlabely].push_back(minlabely);
					}
				}
			}
		}
	}


	void SuperPixelHelper::gatherSuperpixel(const vector<Frame>&frames, const vector<vector<int> >&labels, const vector<int>&num_labels, const vector<Depth>&dynamic_confidence, vector< vector<SuperPixel> >& superpixel){
		if(frames.size() == 0)
			return;
		const int width = dynamic_confidence[0].getWidth();
		superpixel.clear();
		for(int frameid=0; frameid<frames.size(); frameid++){
			vector<SuperPixel> cursuperpixel(num_labels[frameid]);
			for(int pixid=0; pixid<labels[frameid].size(); pixid++){
				const int curind = labels[frameid][pixid];
				cursuperpixel[curind].indices.push_back(pixid);
				cursuperpixel[curind].average_confidence += dynamic_confidence[frameid].getDepthAtInd(pixid);
				int curx = pixid % width; int cury = pixid / width;
				cursuperpixel[curind].center += Vector2d(curx, cury);
			}
			for(int pixid=0; pixid<cursuperpixel.size(); pixid++){
				if(cursuperpixel[pixid].indices.size() == 0)
					continue;
				cursuperpixel[pixid].center /= (double)cursuperpixel[pixid].indices.size();
				cursuperpixel[pixid].average_confidence /= (double)cursuperpixel[pixid].indices.size();
			}
			superpixel.push_back(cursuperpixel);
		}
	}

	void SuperPixelHelper::markSuperpixel(const SuperPixel& superpixel,
	                                      Mat& image,
	                                      const double ratio){
		uchar* imgptr = image.data;
		if(!imgptr)
			return;
		const int width = image.cols;
		const int height = image.rows;

		for(int i=0; i<superpixel.indices.size(); i++){
			const int curind = superpixel.indices[i];
			int x = (curind % static_cast<int>(width / ratio)) * ratio;
			int y = (curind / static_cast<int>(width / ratio)) * ratio;
			if(x >= width || y >= height)
				continue;
			int resized_ind = y * width + x;
			imgptr[resized_ind * 3 + 0] = 255;
			imgptr[resized_ind * 3 + 1] = 255;
			imgptr[resized_ind * 3 + 2] = 255;
		}
	}


}//namespace dynamic_rendering
