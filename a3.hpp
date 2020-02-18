/*  YOUR_FIRST_NAME
 *  YOUR_LAST_NAME
 *  YOUR_UBIT_NAME
 */

#include <iostream>
#include <functional>
#include <algorithm>
#include <cuda.h>
#include <math.h>

#ifndef A3_HPP
#define A3_HPP
#define THREAD_COUNT 1024 /* Max supported threads in CUDA */
#define PI 3.1415926

/* Actual Gaussian Kernel logic called by CUDA RT */
__global__ void gaussian_kernel(float *x_gpu, float *y_gpu, int size, float h, float FACTOR)
{

int tid = blockIdx.x * blockDim.x + threadIdx.x; // Initialize thread ID

__shared__ float *x_shared;
x_shared[tid] = x_gpu[tid];
__syncthreads();

float f = 0.0;

for (int i = 0; i < size; i++)
    {
        f = f + ((FACTOR) * (__expf( -(__powf((x_shared[tid] - x_shared[i]), 2) / 2))));
    }

    y_gpu[tid] = f / (size * h);
}


void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y)
{

/*
The basic pseudo code for having this assignment run will be as follows:

- Allocate memory on the GPU for the input and output
- Copy the input content to the GPU mem for processing
- Initialize the dimensions
- Call the actual Gaussian kernel function over the initialized blocks of data
    which will have CUDA RT run the function in parallel
- Once completed, copy the output from GPU buffer to host memory

*/

int blk_count = (n + THREAD_COUNT - 1) / THREAD_COUNT; // Get the no of blocks

float *x_gpu, *y_gpu;

cudaMalloc(&x_gpu, sizeof(float) * n);
cudaMalloc(&y_gpu, sizeof(float) * n);

/* Copy contents of x into x_gpu i.e. from host to device */
cudaMemcpy(x_gpu, x.data(), sizeof(float) * n, cudaMemcpyHostToDevice);

dim3 gpu_thread_count(THREAD_COUNT);
dim3 gpu_block_count(blk_count);

/* Compute 1 / sqrt(2*PI) only once and save the repeated computations on the GPU */
float FACTOR = (1 / sqrt(2 * PI));
gaussian_kernel<<< gpu_block_count, gpu_thread_count >>>(x_gpu, y_gpu, n, h, FACTOR);

cudaMemcpy(y.data(), y_gpu, sizeof(float) * n, cudaMemcpyDeviceToHost);

} // gaussian_kde

#endif // A3_HPP
