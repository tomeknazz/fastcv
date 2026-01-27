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

#define SWAP(a, b) { unsigned char temp = a; a = min(a, b); b = max(temp, b); }

__device__ unsigned char sorting_network25_simple_split(unsigned char* window){

    unsigned char p0 = window[0];
    unsigned char p1 = window[1];
    unsigned char p2 = window[2];
    unsigned char p3 = window[3];
    unsigned char p4 = window[4];
    unsigned char p5 = window[5];
    unsigned char p6 = window[6];
    unsigned char p7 = window[7];
    unsigned char p8 = window[8];
    unsigned char p9 = window[9];
    unsigned char p10 = window[10];
    unsigned char p11 = window[11];
    unsigned char p12 = window[12];
    unsigned char p13 = window[13];
    unsigned char p14 = window[14];
    unsigned char p15 = window[15];
    unsigned char p16 = window[16];
    unsigned char p17 = window[17];
    unsigned char p18 = window[18];
    unsigned char p19 = window[19];
    unsigned char p20 = window[20];
    unsigned char p21 = window[21];
    unsigned char p22 = window[22];
    unsigned char p23 = window[23];
    unsigned char p24 = window[24];

    SWAP(p1, p2); SWAP(p0, p1); SWAP(p1, p2); SWAP(p4, p5); SWAP(p3, p4);
    SWAP(p4, p5); SWAP(p0, p3); SWAP(p2, p5); SWAP(p2, p3); SWAP(p1, p4);
    SWAP(p1, p2); SWAP(p3, p4); SWAP(p7, p8); SWAP(p6, p7); SWAP(p7, p8);
    SWAP(p10, p11); SWAP(p9, p10); SWAP(p10, p11); SWAP(p6, p9); SWAP(p8, p11);
    SWAP(p8, p9); SWAP(p7, p10); SWAP(p7, p8); SWAP(p9, p10); SWAP(p0, p6);
    SWAP(p4, p10); SWAP(p4, p6); SWAP(p2, p8); SWAP(p2, p4); SWAP(p6, p8);
    SWAP(p1, p7); SWAP(p5, p11); SWAP(p5, p7); SWAP(p3, p9); SWAP(p3, p5);
    SWAP(p7, p9); SWAP(p1, p2); SWAP(p3, p4); SWAP(p5, p6); SWAP(p7, p8);
    SWAP(p9, p10); SWAP(p13, p14); SWAP(p12, p13); SWAP(p13, p14); SWAP(p16, p17);
    SWAP(p15, p16); SWAP(p16, p17); SWAP(p12, p15); SWAP(p14, p17); SWAP(p14, p15);
    SWAP(p13, p16); SWAP(p13, p14); SWAP(p15, p16); SWAP(p19, p20); SWAP(p18, p19);
    SWAP(p19, p20); SWAP(p21, p22); SWAP(p23, p24); SWAP(p21, p23); SWAP(p22, p24);
    SWAP(p22, p23); SWAP(p18, p21); SWAP(p20, p23); SWAP(p20, p21); SWAP(p19, p22);
    SWAP(p22, p24); SWAP(p19, p20); SWAP(p21, p22); SWAP(p23, p24); SWAP(p12, p18);
    SWAP(p16, p22); SWAP(p16, p18); SWAP(p14, p20); SWAP(p20, p24); SWAP(p14, p16);
    SWAP(p18, p20); SWAP(p22, p24); SWAP(p13, p19); SWAP(p17, p23); SWAP(p17, p19);
    SWAP(p15, p21); SWAP(p15, p17); SWAP(p19, p21); SWAP(p13, p14); SWAP(p15, p16);
    SWAP(p17, p18); SWAP(p19, p20); SWAP(p21, p22); SWAP(p23, p24); SWAP(p0, p12);
    SWAP(p8, p20); SWAP(p8, p12); SWAP(p4, p16); SWAP(p16, p24); SWAP(p12, p16);
    SWAP(p2, p14); SWAP(p10, p22); SWAP(p10, p14); SWAP(p6, p18); SWAP(p6, p10);
    SWAP(p10, p12); SWAP(p1, p13); SWAP(p9, p21); SWAP(p9, p13); SWAP(p5, p17);
    SWAP(p13, p17); SWAP(p3, p15); SWAP(p11, p23); SWAP(p11, p15); SWAP(p7, p19);
    SWAP(p7, p11); SWAP(p11, p13); SWAP(p11, p12);

    return p12;

}

__device__ unsigned char sorting_network9_simple_split(unsigned char* window){

    unsigned char w0=window[0];
    unsigned char w1=window[1];
    unsigned char w2=window[2];
    unsigned char w3=window[3];
    unsigned char w4=window[4];
    unsigned char w5=window[5];
    unsigned char w6=window[6];
    unsigned char w7=window[7];
    unsigned char w8=window[8];

    SWAP(w1, w2);    SWAP(w4, w5);    SWAP(w7, w8);
    SWAP(w0, w1);    SWAP(w3, w4);    SWAP(w6, w7);
    SWAP(w1, w2);    SWAP(w4, w5);    SWAP(w7, w8);
    SWAP(w0, w3);    SWAP(w5, w8);    SWAP(w4, w7);
    SWAP(w3, w6);    SWAP(w1, w4);    SWAP(w2, w5);
    SWAP(w4, w7);    SWAP(w4, w2);    SWAP(w6, w4);
    SWAP(w4, w2);
    return w4;

}

//For further testing
__global__ void medianBlurKernel_simple_split(unsigned char* in, unsigned char* out,int width, int height, int channels, int blur_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
     //Using z dimension for channel
    int channel = blockIdx.z;

    const int offset = blur_size / 2;
    //auto getOffset = [](int a, int b, int offset) {return a + b - offset;};
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
            unsigned char window[25]; //Window = pixels to calculate median from
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
            out[(row * width + col) * channels + channel] = sorting_network25_simple_split(window);
    }
}
//Namespace for requirement
using thrust_device_uchar_ptr = thrust::device_ptr<unsigned char>;

torch::Tensor median_blur_simple_split(torch::Tensor img, int blur_size){
    nvtxRangePushA("Median Blur Start");
    //Make sure input is correct
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);
    const auto height = img.size(0);
    const auto width = img.size(1);
    const auto channels = img.size(2);

    //Defining stream count and segment height for splitting image into separete streams
    const int stream_count = 16;
    int segment_height = height / stream_count;

    nvtxRangePushA("Memory allocation");
    auto result = torch::empty({height, width, channels}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kByte));
    unsigned char* img_ptr = img.data_ptr<unsigned char>();
    unsigned char* out_ptr = result.data_ptr<unsigned char>();

    cudaStream_t streams[stream_count];
    for(int i=0; i<stream_count; i++){
        cudaStreamCreate(&streams[i]);
    }
    nvtxRangePop(); //Memory allocation

    for(int i=0; i<stream_count; ++i){
        int y_start = i * segment_height;
        int current_segment_height = (i == stream_count - 1) ? (height - y_start) : segment_height;
        int offset = y_start * width * channels;
        int segment_elements = current_segment_height * width * channels; // do debugowania

        dim3 dimBlock = getOptimalBlockDim(width, current_segment_height);
        dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(current_segment_height, dimBlock.y), channels);
        int shared_memory_size = (dimBlock.x + blur_size) * (dimBlock.y + blur_size) * sizeof(unsigned char);

        medianBlurKernel_simple_split<<<dimGrid, dimBlock, shared_memory_size, streams[i]>>>(
            img_ptr + offset,
            out_ptr + offset,
            width, current_segment_height, channels, blur_size
        );
    }

    //Synchronize all streams
    for(int i=0; i<stream_count; i++){
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    nvtxRangePop(); //Median Blur End



    return result;
}