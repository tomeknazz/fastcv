#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <nvtx3/nvtx3.hpp>
#DEFINE NOMINMAX

#include "utils.cuh"

/*
TODO:

Modern c++ (Choose 2):
-Templates
-Iterators
-Containers - DONE
-Functions - DONE
-Operator Overloading
-Lambda Expressions
-Type Aliases

Thrust (One of each type):
-Fancy iterators - TODO
-Vocabulary types - TODO
-Execution policies - DONE
-Execution space specifier - TODO
-Thrust alghorithms - DONE

Async, CUB, Nvidia tools (Minimum 2):
-Async elements
-Comparison between Thrust and CUB implementation
-cudaDeviceSynchronize
-Compute IO overlap
-Copy compute overlap
-Cuda streams
-Pinned memory
-cudaMemcpyAsync
Obligatory:
-Nsight analysis
-Nvidia tools extension NVTX

CUDA kenel:
-Has to use grid,block,thread indexing
-optimal block size calculation
-uses atomic operations if necessary
-shows thread synchronisation
-uses streaming multiprocessor efficiently
-uses shared and global memory


*/

// Will cause integer rounding errors
int median(thrust::device_vector<int> window){
    thrust::sort(window.begin(), window.end());
    int size = window.size();
    if (size % 2 == 1)
        return window[size / 2];

    return int((window[size / 2 - 1] + window[size / 2]) / 2);
}

//Less accurate median
//For further testing
int fast_median(thrust::device_vector<int> window){
    thrust::sort(window.begin(), window.end());
    return window[window.size() / 2];
}

//For further testing
float median(thrust::device_vector<float> window){
    thrust::sort(window.begin(), window.end());
    int size = window.size();
    if (size % 2 == 1)
        return window[size / 2];

    return (window[size / 2 - 1] + window[size / 2]) / 2.0f;
}

//Could do version that takes &window to avoid copying
//Maybe todo later

__global__ void medianBlurKernel(unsigned char* in, unsigned char* out, int width, int height, int channels, int blur_size, int blur_size_x = NULL, int blur_size_y = NULL) {
    int col = blockIdx.x +blockDim.x * threadIdx.x;
    int row = blockIdx.y +blockDim.y * threadIdx.y;




}