#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
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

//#include "bits/stdc++.h"

//#include "nvtx3.hpp"
#define NOMINMAX

#include "utils.cuh"

/*
TODO:

Modern c++ (Choose 2): DONE
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

/*
__device__ int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    int temp = 0;
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            temp = arr[i] ;
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    temp = arr[i+1] ;
    arr[i+1] = arr[high];
    arr[high] = temp;
    return (i + 1);
}
__device__ void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
__device__ int median(int arr[], int size){
    //int size = sizeof(arr) / sizeof(int);
    quickSort(arr, 0, size - 1);
    if (size % 2 == 1)
        return arr[size / 2];

    return int((arr[size / 2 - 1] + arr[size / 2]) / 2);
}

//Less accurate median
//For further testing
int fast_median(thrust::device_vector<int> &window){
    thrust::sort(window.begin(), window.end());
    return window[window.size() / 2];
}

//For further testing
__device__ float median(thrust::device_vector<float> &window){
    thrust::sort(window.begin(), window.end());
    int size = window.size();
    if (size % 2 == 1)
        return window[size / 2];

    return (window[size / 2 - 1] + window[size / 2]) / 2.0f;
}

//Could do version that takes &window to avoid copying
//Maybe todo later

*/
//Using templates to meet requirements
//Maybe for adptive median blur later???
//This compiles with no problems
template <typename T>
__device__ thrust::pair<float, T> thrust_median(T *window, int size){
    int array_size = size * size;
    thrust::sort(thrust::device, window, window + array_size);
    //Using pair to meeet requirements, min_val will be unused
    T min_val = window[0];
    if (array_size % 2 == 1)
        return thrust::make_pair(window[array_size / 2]/1.0f, min_val/1.0f);

    return thrust::make_pair((window[array_size / 2 - 1] + window[array_size / 2]) / 2.0f, min_val);
}

__device__ float fast_median(unsigned char *window){
    constexpr int array_size = 25;
    thrust::sort(thrust::device, window, window + array_size);
    if (array_size % 2 == 1)
        return window[array_size / 2]/1.0f;
    return (window[array_size / 2 - 1] + window[array_size / 2]) / 2.0f;
}

//For further testing
__global__ void medianBlurKernel(unsigned char* in, unsigned char* out,int width, int height, int channels, int blur_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int offset = blur_size / 2;

    if(col<width && row<height){
        for(int c=0; c<channels; ++c){
            unsigned char window[1024]; //Window = pixels to calculate median from
            int counter = 0; //For indexing in window
            //Collect pixels in the window
            for(int y = 0; y < blur_size; ++y){
                for(int x = 0; x < blur_size; ++x){
                    //This makes sure we are centered around the pixel
                    //For example for blur_size = 3, offset = 1
                    //y = 0 -> row - 1
                    //y = 1 -> row + 0
                    //y = 2 -> row + 1
                    //And for blur_size = 2
                    //y = 0 -> row - 1
                    //y = 1 -> row + 0
                    //Same for x
                    int curRow = row + y - offset;
                    int curCol = col + x - offset;
                    //Check bounds
                    if(curRow >= 0 && curRow < height && curCol >= 0 && curCol < width){
                        //Indexing like this because input is R,G,B,R,G,B,... etc.
                        //We only want one channel at a time
                        window[counter++]= in[(curRow * width + curCol) * channels + c];
                    }
                }
            }
            //Compute median
            //thrust::pair<float, unsigned char> med = thrust_median<unsigned char>(window, blur_size);
            //Same here with the indexing
            //out[(row * width + col) * channels + c] = static_cast<unsigned char>(med.first);
            out[(row * width + col) * channels + c] = static_cast<unsigned char>(fast_median(window));
        }
    }
}

torch::Tensor median_blur(torch::Tensor img, int blur_size){
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);

    const auto height = img.size(0);
    const auto width = img.size(1);
    const auto channels = img.size(2);

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));



    auto result = torch::empty({height, width, channels},
                              torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    unsigned char* in_ptr = img.data_ptr<unsigned char>();
    unsigned char* out_ptr = result.data_ptr<unsigned char>();

    //Using thrust:device_ptr to meet the requirement
    thrust::device_ptr<unsigned char> thrust_in_ptr = thrust::device_pointer_cast(in_ptr);
    thrust::device_ptr<unsigned char> thrust_out_ptr = thrust::device_pointer_cast(out_ptr);

    medianBlurKernel<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        thrust_in_ptr.get(),
        thrust_out_ptr.get(),
        width, height, channels, blur_size); //__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h, int channels, int BLUR_SIZE) {
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
