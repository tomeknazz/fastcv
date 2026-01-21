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
#include <cub/cub.cuh>

#include "nvtx3.hpp"
#define NOMINMAX

#include "utils.cuh"

/*
TODO:

Modern c++ (Choose 2): - DONE
-Templates
-Iterators
-Containers
-Functors
-Operator Overloading
-Lambda Expressions - DONE (getOffset)
-Type Aliases

Thrust (One of each type):
-Fancy iterators - TODO
-Vocabulary types - TODO
-Execution policies - TODO
-Execution space specifier - TODO
-Thrust alghorithms - TDOD

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
-Has to use grid,block,thread indexing - DONE
-optimal block size calculation - DONE
-uses atomic operations if necessary
-shows thread synchronisation
-uses streaming multiprocessor efficiently
-uses shared and global memory

*/


//Using templates to meet requirements
//Maybe for adptive median blur later???
//This compiles with no problems
template <typename T>
__device__ thrust::pair<int, T> thrust_median(T *window, int count){
    int array_size = count;
    thrust::sort(thrust::device, window, window + array_size);
    //Using pair to meeet requirements, min_val will be unused
    T min_val = window[0];
    if (array_size % 2 == 1)
        return thrust::make_pair(window[array_size / 2], min_val);

    return thrust::make_pair((window[array_size / 2 - 1] + window[array_size / 2]) / 2, min_val);
}
__device__ void bubbleSort(unsigned char* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                unsigned char temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

//For further testing
__global__ void medianBlurKernel(unsigned char* in, unsigned char* out,int width, int height, int channels, int blur_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
     //Using z dimension for channel
    int channel = blockIdx.z;

    const int offset = blur_size / 2;
    auto getOffset = [](int a, int b, int offset) {return a + b - offset;};
    //Shared memory allocation
    // int shared_memory_size = (dimBlock.x + blur_size) * (dimBlock.y + blur_size)* sizeof(unsigned char);
    extern __shared__ unsigned char shared_memory[];
    //Loading into shared memory
    for(int i=threadIdx.y*blockDim.x+threadIdx.x; i<(blockDim.x + blur_size)*(blockDim.y + blur_size); i+=blockDim.x*blockDim.y){
        int shared_row = i / (blockDim.x + blur_size);
        int shared_col = i % (blockDim.x + blur_size);
        int global_row = blockIdx.y * blockDim.y + shared_row - offset;
        int global_col = blockIdx.x * blockDim.x + shared_col - offset;
        global_row=max(0, min(global_row, height - 1));
        global_col=max(0, min(global_col, width - 1));
        shared_memory[i] = in[(global_row * width + global_col) * channels + channel];
    }

    __syncthreads(); //Ensure all data is loaded

    if(col<width && row<height){
            unsigned char window[121]; //Window = pixels to calculate median from
            int counter = 0; //For indexing in window
            //Collect pixels in the window
            for(int y = 0; y < blur_size; ++y){
                for(int x = 0; x < blur_size; ++x){
                    //Using shared memory here
                    int shared_memory_index = (threadIdx.y + y) * (blockDim.x + blur_size) + (threadIdx.x + x);
                    window[counter++]= shared_memory[shared_memory_index];
                    //This makes sure we are centered around the pixel
                    //For example for blur_size = 3, offset = 1
                    //y = 0 -- row - 1
                    //y = 1 -- row + 0
                    //y = 2 -- row + 1
                    //And for blur_size = 2
                    //y = 0 -- row - 1
                    //y = 1 -- row + 0
                    //Same for x
                    //int curRow = getOffset(row,y,offset);
                    //int curCol = getOffset(col,x,offset);
                    //Check bounds
                    //if(curRow >= 0 && curRow < height && curCol >= 0 && curCol < width){
                        //Indexing like this because input is R,G,B,R,G,B,... etc.
                        //We only want one channel at a time
                        //window[counter++]= in[(curRow * width + curCol) * channels + channel];
                    //}
                }
            }
            //Compute median
            //thrust::pair<int, unsigned char> med = thrust_median<unsigned char>(window, counter);
            bubbleSort(window, counter);
            out[(row * width + col) * channels + channel] = window[counter / 2];
            //Same here with the indexi
            //out[(row * width + col) * channels + channel] = static_cast<unsigned char>(med.first);
            //out[(row * width + col) * channels + c] = static_cast<unsigned char>(fast_median(window));
    }
}

torch::Tensor median_blur(torch::Tensor img, int blur_size){
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);
    //We can add thrust::pair here
    const auto height = img.size(0);
    const auto width = img.size(1);
    const auto channels = img.size(2);

    dim3 dimBlock = getOptimalBlockDim(width, height);
    //Added channels to make use of parallelism
    //We can process multiple channels at the same time
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y),channels);

    auto result = torch::empty_like(img);

    unsigned char* in_ptr = img.data_ptr<unsigned char>();
    unsigned char* out_ptr = result.data_ptr<unsigned char>();

    //Using thrust:device_ptr to meet the requirement
    thrust::device_ptr<unsigned char> thrust_in_ptr = thrust::device_pointer_cast(in_ptr);
    thrust::device_ptr<unsigned char> thrust_out_ptr = thrust::device_pointer_cast(out_ptr);

    //Memory size has to be here to compile correctly
    int shared_memory_size = (dimBlock.x + blur_size) * (dimBlock.y + blur_size)* sizeof(unsigned char);

    medianBlurKernel<<<dimGrid, dimBlock, shared_memory_size, at::cuda::getCurrentCUDAStream()>>>(
        thrust_in_ptr.get(),
        thrust_out_ptr.get(),
        width, height, channels, blur_size); //__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h, int channels, int BLUR_SIZE) {
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}