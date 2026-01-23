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
#include <time.h>

#include "nvtx3.hpp"
#define NOMINMAX

#include "utils.cuh"

/*
TODO:

Modern c++ (Choose 2): -> DONE
-Templates
-Iterators
-Containers - DONE -> thrust::device_vector
-Functors
-Operator Overloading
-Lambda Expressions - DONE -> getOffset
-Type Aliases - DONE -> using thrust_device_uchar_ptr = thrust::device_ptr<unsigned char>;

Thrust (One of each type):
-Fancy iterators - TODO
-Vocabulary types - DONE -> thrust::pair
-Execution policies - TODO
-Execution space specifier - DONE -> __device__
-Thrust alghorithms - DONE - DONE -> thrust::copy

Async, CUB, Nvidia tools (Minimum 2): -> DONE
-Async elements - DONE
-Comparison between Thrust and CUB implementation
-cudaDeviceSynchronize - DONE
-Compute IO overlap - DONE
-Copy compute overlap
-Cuda streams - DONE
-Pinned memory
-cudaMemcpyAsync - DONE
Obligatory:
-Nsight analysis - TODO
-Nvidia tools extension NVTX - TODO

CUDA kenel: -> DONE
-Has to use grid,block,thread indexing - DONE -> BlockIdx, threadIdx etc.
-optimal block size calculation - DONE -> getOptimalBlockDim
-uses atomic operations if necessary
-shows thread synchronisation - DONE -> __syncthreads()
-uses streaming multiprocessor efficiently - DONE
-uses shared and global memory - DONE -> __shared__

*/

__device__ void insertionSort(unsigned char* arr, int n) {
    for (int i = 1; i < n; ++i) {
        unsigned char key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
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
        //Ensure we don't go out of bounds
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
                }
            }
            //Compute median
            insertionSort(window, counter);
            out[(row * width + col) * channels + channel] = window[counter / 2];
    }
}
//Namespace for requirement
using thrust_device_uchar_ptr = thrust::device_ptr<unsigned char>;

torch::Tensor median_blur(torch::Tensor img, int blur_size){
    nvtxRangePushA("Median Blur Start");
    //Make sure input is correct
    assert(img.device().type() == torch::kCPU);
    assert(img.dtype() == torch::kByte);
    const auto height = img.size(0);
    const auto width = img.size(1);
    const auto channels = img.size(2);
    const int vector_size = height * width * channels;

    nvtxRangePushA("Memory allocation");
    auto in_tensor = torch::empty({height, width, channels}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kByte));
    auto result = torch::empty_like(in_tensor);
    unsigned char* img_ptr = img.data_ptr<unsigned char>();
    unsigned char* in_ptr = in_tensor.data_ptr<unsigned char>();
    unsigned char* out_ptr = result.data_ptr<unsigned char>();

    cudaMemcpyAsync(
        in_ptr,
        img_ptr,
        vector_size * sizeof(unsigned char),
        cudaMemcpyHostToDevice,
        at::cuda::getCurrentCUDAStream()
    );
    nvtxRangePop(); //Memory allocation

    nvtxRangePushA("Pre kernel setup");
    //Using thrust::pair for dimensions
    auto dimensions = thrust::make_pair(height, width);
    dim3 dimBlock = getOptimalBlockDim(width, height);
    //Added channels to make use of parallelism
    //We can process multiple channels at the same time
    dim3 dimGrid(cdiv(dimensions.second, dimBlock.x), cdiv(dimensions.first, dimBlock.y),channels);
    //Using thrust:device_ptr to meet the requirement
    thrust_device_uchar_ptr thrust_in_ptr = thrust::device_pointer_cast(in_ptr);
    thrust_device_uchar_ptr thrust_out_ptr = thrust::device_pointer_cast(out_ptr);

    //Memory size has to be here to compile correctly
    int shared_memory_size = (dimBlock.x + blur_size) * (dimBlock.y + blur_size)* sizeof(unsigned char);
    nvtxRangePop(); //Pre kernel setup
    //Kernel execution
    nvtxRangePushA("Kernel execution");
    medianBlurKernel<<<dimGrid, dimBlock, shared_memory_size, at::cuda::getCurrentCUDAStream()>>>(
        thrust_in_ptr.get(),
        thrust_out_ptr.get(),
        width, height, channels, blur_size); //__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h, int channels, int BLUR_SIZE) {
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    nvtxRangePop(); //Kernel execution
    //nvtxRangePushA("Copy back to CPU");


    //Create cpu tensor for result because otherwise we will have to copy again on python side
    /*auto result_cpu_tensor = torch::empty({height, width, channels}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kByte));
    //auto result_cpu_tensor = torch::empty({height, width, channels}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kByte).pinned_memory(true));
    unsigned char* result_cpu_ptr = result_cpu_tensor.data_ptr<unsigned char>();
    cudaMemcpyAsync(
        result_cpu_ptr,
        out_ptr,
        vector_size * sizeof(unsigned char),
        cudaMemcpyDeviceToHost,
        stream
    );
    //Synchronisation and free allocated memory
    cudaStreamSynchronize(stream);
    */
    //nvtxRangePop(); //Copy back to CPU
    nvtxRangePop(); //Median Blur End

    return result;
}