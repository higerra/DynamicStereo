//
// Created by yanhang on 9/5/16.
//

#ifndef DYNAMICSTEREO_CUDA_STEREO_H
#define DYNAMICSTEREO_CUDA_STEREO_H
#include <cmath>
#include <cfloat>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace CudaVision{

    inline void HandleCuError(cudaError_t e){
        CHECK_EQ(e, cudaSuccess) << cudaGetErrorString(e);
    }
    //--------------------------------------
    //Math funcition
    //--------------------------------------
    template<typename TIn, typename TLoc, typename TOut>
    __device__ __host__ void bilinearInterpolation(const TIn* data, const int w,
                                                   const TLoc loc[2], TOut res[3]){
        const float epsilon = 0.00001;
        int xl = floor(loc[0] - epsilon), xh = (int) round(loc[0] + 0.5 - epsilon);
        int yl = floor(loc[1] - epsilon), yh = (int) round(loc[1] + 0.5 - epsilon);

        if (loc[0] <= epsilon)
            xl = 0;
        if (loc[1] <= epsilon)
            yl = 0;

        const int l1 = yl * w + xl;
        const int l2 = yh * w + xh;
        if (l1 == l2) {
            for (size_t i = 0; i < 3; ++i)
                res[i] = (TOut)data[l1 * 3 + i];
            return;
        }

        TOut lm = loc[0] - (TOut) xl, rm = (TOut) xh - loc[0];
        TOut tm = loc[1] - (TOut) yl, bm = (TOut) yh - loc[1];
        int ind[4] = {xl + yl * w, xh + yl * w, xh + yh * w, xl + yh * w};

        TOut v[4][3] = {};
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 3; ++j)
                v[i][j] = (TOut)data[ind[i] * 3 + j];
        }
        if (fabs(lm) <= epsilon && fabs(rm) <= epsilon){
            res[0] = (v[0][0] * bm + v[2][0] * tm) / (bm + tm);
            res[1] = (v[0][1] * bm + v[2][1] * tm) / (bm + tm);
            res[2] = (v[0][2] * bm + v[2][2] * tm) / (bm + tm);
            return;
        }

        if (fabs(bm) <= epsilon && fabs(tm) <= epsilon) {
            res[0] = (v[0][0] * rm + v[2][0] * lm) / (lm + rm);
            res[1] = (v[0][1] * rm + v[2][1] * lm) / (lm + rm);
            res[2] = (v[0][2] * rm + v[2][2] * lm) / (lm + rm);
            return;
        }

        TOut vw[4] = {rm * bm, lm * bm, lm * tm, rm * tm};
        TOut sum = vw[0] + vw[1] + vw[2] + vw[3];

        res[0] = (v[0][0] * vw[0] + v[1][0] * vw[1] + v[2][0] * vw[2] + v[3][0] * vw[3]) / sum;
        res[1] = (v[0][1] * vw[0] + v[1][1] * vw[1] + v[2][1] * vw[2] + v[3][1] * vw[3]) / sum;
        res[2] = (v[0][2] * vw[0] + v[1][2] * vw[1] + v[2][2] * vw[2] + v[3][2] * vw[3]) / sum;
    }

    template<typename T>
    __device__ __host__ T variance(const T* const a, const T mean, const int N) {
        if(N == 1)
            return (T)0;
        T res = 0;
        for(int i=0; i<N; ++i){
            res = res + (a[i] - mean) * (a[i] - mean);
        }
        res /= (T)N;
        return sqrt(res);
    }

    template<typename T>
    __device__ __host__ T variance(const T* const a, const int N) {
        T mean = 0;
        for(int i=0; i<N; ++i)
            mean += a[i];
        return variance<T>(a, mean, N);
    }

    template<typename T>
    __device__ __host__ T normalizedCrossCorrelation(const T* const a1, const T* const a2, const int N){
        T m1 = 0, m2 = 0;
        for(int i=0; i<N; ++i){
            m1 += a1[i];
            m2 += a2[i];
        }
        m1 /= (T)N;
        m2 /= (T)N;

        T var1 = variance<T>(a1, m1, N);
        T var2 = variance<T>(a2, m2, N);
        if(var1 == 0 || var2 == 0 )
            return 0;

        T ncc = 0;
        for (size_t i = 0; i < N; ++i)
            ncc += (a1[i] - m1) * (a2[i] - m2);
        ncc /= (var1 * var2 * (N-1));
        return ncc;
    }

    //----------------------------------
    //Basic algorithm
    //----------------------------------
    template<typename T>
	__device__ __host__ inline void swap(T& v1, T& v2){
	T tmp = v1;
	v1 = v2;
	v2 = tmp;
    }
    
    template<typename T>
    __device__ __host__ int partition(T* array, const int left, const int right, const int pivInd){
        T piv = array[pivInd];
        swap(array[right], array[pivInd]);
        int storeInd = left;
        for(int i=left; i<right; ++i){
            if(array[i] < piv){
                swap(array[storeInd], array[i]);
                storeInd++;
            }
        }
        swap(array[storeInd], array[right]);
        return storeInd;
    }


    template<typename T>
    __device__ __host__ T find_nth(T* array, const int N, const int nth) {
        int left = 0 , right = N - 1;
        while(left < right){
            int pivInd = left;
            pivInd = partition<T>(array, left, right, left);
            if(pivInd < nth){
                left = pivInd + 1;
            }else if(pivInd > nth){
                right = pivInd - 1;
            }else
                break;
        }
        return array[nth];
    }

    template<typename T>
	__device__ __host__ void quick_sort(T* array, const int lo, const int hi){
	if(lo < hi){
	    int pivInd = partition(array, lo, hi, (lo + hi) / 2);
	    quick_sort<T>(array, lo, pivInd-1);
	    quick_sort<T>(array, pivInd+1, hi);
	}
    }
	

}//namespace CudaVision


#endif //DYNAMICSTEREO_CUDA_STEREO_H
