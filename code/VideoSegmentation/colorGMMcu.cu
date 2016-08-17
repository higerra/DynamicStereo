#ifdef USE_CUDA
#include "colorGMM.h"

using namespace std;
using namespace cv;

namespace dynamic_stereo{

    static const int kBlock = 2056;
    static const int kThread = 512;

    static inline void HandleCuError(cudaError_t e){
	CHECK_EQ(e, cudaSuccess) << cudaGetErrorString(e);
    };

    
    __global__ void computeProb(const float* data, float* output, const float* m, const float* inverseCovs,
				const float covDetrm, const int width, const int height){
	int baseIdx = threadIdx.x + blockIdx.x * blockDim.x;
	float diff[3] = {};
	for(auto i=baseIdx; i<width * height; i+=gridDim.x * blockDim.x){
	    diff[0] = data[i*3] - m[0];
	    diff[1] = data[i*3+1] - m[1];
	    diff[2] = data[i*3+2] - m[2];
	    float mult =
		diff[0]*(diff[0]*inverseCovs[0] + diff[1]*inverseCovs[3] + diff[2]*inverseCovs[6])
		+ diff[1]*(diff[0]*inverseCovs[1] + diff[1]*inverseCovs[4] + diff[2]*inverseCovs[7])
		+ diff[2]*(diff[0]*inverseCovs[2] + diff[1]*inverseCovs[5] + diff[2]*inverseCovs[8]);
	    output[i] = 1.0 / sqrt(covDetrm) * exp(-0.5 * mult);
	}
    }

    __global__ void assignComponentKernel(const unsigned char* data, unsigned char* output, const float* means,
					  const float* inverseCovs, const float* covDetrms, const int N, const int kPix, const int kCom){
	int baseIdx = threadIdx.x + blockIdx.x * blockDim.x;
	float diff[3] = {};
	for(int v=0; v<N; ++v){
	    unsigned char* pOutput = output + kPix * v;
	    const unsigned char* pData = data + kPix * 3 * v;
	    for(int i=baseIdx; i<kPix; i+=gridDim.x * blockDim.x){
		float maxProb = -1;
		for(int ci=0; ci < kCom; ++ci){
		    const float* m = means + 3 * v;
		    const float* inv = inverseCovs + 9 * v;
		    const float* detrm = covDetrms + v;
		    diff[0] = (float)pData[i*3] - m[0];
		    diff[1] = (float)pData[i*3+1] - m[1];
		    diff[2] = (float)pData[i*3+2] - m[2];
		    float mult =
			diff[0]*(diff[0]*inv[0] + diff[1]*inv[3] + diff[2]*inv[6])
			+ diff[1]*(diff[0]*inv[1] + diff[1]*inv[4] + diff[2]*inv[7])
			+ diff[2]*(diff[0]*inv[2] + diff[1]*inv[5] + diff[2]*inv[8]);
		    float prob = 1.0 / sqrt(detrm[0]) * exp(-0.5 * mult);
		    if(prob > maxProb){
			maxProb = prob;
			pOutput[i] = (unsigned char)ci;
		    }
		}
	    }
	}
    }
    
    void ColorGMM::computeRawProbCuda(const int ci, const cv::Mat& image, cv::Mat& output) const{
	CHECK(!image.empty());
	CHECK_LT(ci, componentsCount);
	const int width = image.cols;
	const int height = image.rows;

	Mat rawimage;
	image.convertTo(rawimage, CV_32FC3);

	output.create(image.size(), CV_32FC1);
	output.setTo(cv::Scalar::all(1.0f));
	
	float* dev_data, *dev_mean, *dev_inverseCovs, *dev_output;
	
	//allocate memory	
	HandleCuError(cudaMalloc((void**)& dev_data, width * height * 3 * sizeof(float)));
	HandleCuError(cudaMalloc((void**)& dev_mean, 3 * sizeof(float)));
	HandleCuError(cudaMalloc((void**)& dev_inverseCovs, 9 * sizeof(float)));
	HandleCuError(cudaMalloc((void**)& dev_output, width * height * sizeof(float)));

	//copy data to GPU
	HandleCuError(cudaMemcpy(dev_data, (float*)rawimage.data, width * height * 3 * sizeof(float), cudaMemcpyHostToDevice));
	HandleCuError(cudaMemcpy(dev_mean, mean[ci].data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
	HandleCuError(cudaMemcpy(dev_inverseCovs, inverseCovs[ci].data(), 9 * sizeof(float), cudaMemcpyHostToDevice));

	//compute
	computeProb<<<kBlock, kThread>>>(dev_data, dev_output, dev_mean, dev_inverseCovs, covDeterms[ci], width, height);
	//copy data back to CPU
	HandleCuError(cudaMemcpy(output.data, dev_output, width * height * sizeof(float), cudaMemcpyDeviceToHost));

	HandleCuError(cudaFree(dev_data));
    	HandleCuError(cudaFree(dev_mean));
	HandleCuError(cudaFree(dev_inverseCovs));
	HandleCuError(cudaFree(dev_output));
    }

    void ColorGMM::assignComponentCuda(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& output, int kChunck) const{
	CHECK(!images.empty());
	CHECK_EQ(images[0].type(), CV_8UC3);
	const int width = images[0].cols;
	const int height = images[0].rows;

	kChunck = std::min(kChunck, (int)images.size());

	output.resize(images.size());
	for(auto& o: output){
	    o.create(images[0].size(), CV_32SC1);
	    o.setTo(Scalar::all(0));
	}
	
	//upload GMM data
	float* dev_means, *dev_inverseCovs, *dev_det;
	const size_t mem_mean = mean.size() * 3 * sizeof(float);
	const size_t mem_invCov = inverseCovs.size() * 9 * sizeof(float);
	const size_t mem_det = covDeterms.size() * sizeof(float);
	
	HandleCuError(cudaMalloc((void**) &dev_means, mem_mean));
	HandleCuError(cudaMalloc((void**) &dev_inverseCovs, mem_invCov));
	HandleCuError(cudaMalloc((void**) &dev_det, mem_det));

	HandleCuError(cudaMemcpy(dev_means, &mean[0][0], mem_mean, cudaMemcpyHostToDevice));
	HandleCuError(cudaMemcpy(dev_inverseCovs, &inverseCovs[0](0,0), mem_invCov, cudaMemcpyHostToDevice));
	HandleCuError(cudaMemcpy(dev_det, &covDeterms[0], mem_det, cudaMemcpyHostToDevice));


	for(auto v=0; v<images.size() - kChunck; v+=kChunck){
	    int curChunck = kChunck;
	    if(curChunck + v >= images.size())
		curChunck = images.size() - v;
	    const size_t mem_chunck = width * height * curChunck * sizeof(unsigned char);
	    //make a flat copy of the image data
	    std::vector<unsigned char> flatdata (curChunck * height * width * 3, 0);
	    for(auto i=0; i<curChunck; ++i){
		for(auto j=0; j<width * height * 3; ++j){
		    flatdata[i*width*height*3+j] = images[v+i].data[j];
		}
	    }

	    unsigned char* dev_data, *dev_output;
	    HandleCuError(cudaMalloc((void**) &dev_data, mem_chunck * 3));
	    HandleCuError(cudaMalloc((void**) &dev_output, mem_chunck));

	    HandleCuError(cudaMemcpy(dev_data, flatdata.data(), mem_chunck * 3, cudaMemcpyHostToDevice));

	    assignComponentKernel<<<kBlock, kThread>>>(dev_data, dev_output, dev_means, dev_inverseCovs, dev_det, curChunck,
						       width * height, componentsCount);

	    std::vector<unsigned char> flatoutput(curChunck * height * width, 0);
	    HandleCuError(cudaMemcpy(flatoutput.data(), dev_output, mem_chunck, cudaMemcpyDeviceToHost));
	    for(auto i=0; i<curChunck; ++i){
		for(auto j=0; j<width * height; ++j){
		    output[v+i].at<int>(j/width, j%width) = (int)flatoutput[i*width*height+j];
		}
	    }
	    HandleCuError(cudaFree(dev_data));
	    HandleCuError(cudaFree(dev_output));
	}

	HandleCuError(cudaFree(dev_means));
	HandleCuError(cudaFree(dev_inverseCovs));
	HandleCuError(cudaFree(dev_det));
    }
    
}

#endif
