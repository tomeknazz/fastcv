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
//& to avoid copying memory
// Will cause integer rounding errors
int median(thrust::device_vector<int> &window){
    thrust::sort(window.begin(), window.end());
    int size = window.size();
    if (size % 2 == 1)
        return window[size / 2];

    return int((window[size / 2 - 1] + window[size / 2]) / 2);
}

//Less accurate median
//For further testing
int fast_median(thrust::device_vector<int> &window){
    thrust::sort(window.begin(), window.end());
    return window[window.size() / 2];
}

//For further testing
float median(thrust::device_vector<float> &window){
    thrust::sort(window.begin(), window.end());
    int size = window.size();
    if (size % 2 == 1)
        return window[size / 2];

    return (window[size / 2 - 1] + window[size / 2]) / 2.0f;
}

//Could do version that takes &window to avoid copying
//Maybe todo later


//This should in theory work
//blur_size_x and blur_size_y maybe could be used for non square blurs???
//For further testing
__global__ void medianBlurKernel(unsigned char* in, unsigned char* out, int width, int height, int channels, int blur_size, int blur_size_x = NULL, int blur_size_y = NULL) {
    int col = blockIdx.x +blockDim.x * threadIdx.x;
    int row = blockIdx.y +blockDim.y * threadIdx.y;

    if(col<width && row<height){
        for(int c=0; c<channels; ++c){
            thrust::device_vector<int> window;
            //Collect pixels in the window
            for(int blurRow = -blur_size; blurRow <= blur_size; ++blurRow){
                for(int blurCol = -blur_size; blurCol <= blur_size; ++blurCol){
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;
                    //Check bounds
                    if(curRow >= 0 && curRow < height && curCol >= 0 && curCol < width){
                        //Indexing like this because input is R,G,B,R,G,B,... etc.
                        window.push_back(in[(curRow * width + curCol) * channels + c]);
                    }
                }
            }
            //Compute median
            int med = median(window);
            //Same here with the indexing
            out[(row * width + col) * channels + c] = static_cast<unsigned char>(med);
        }


    }



}