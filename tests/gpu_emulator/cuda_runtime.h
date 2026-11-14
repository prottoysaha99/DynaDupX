// Minimal stand-in for the CUDA runtime header, so native/dynadupx_jni.cu compiles with g++ (see run.sh).
#pragma once
#include <stddef.h>
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__
struct uint3s { unsigned x, y, z; };
extern uint3s threadIdx, blockIdx, blockDim, gridDim;
void __syncthreads();
int atomicAdd(int*, int);
enum cudaError_t { cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorNoDevice = 100, cudaErrorUnknown = 999 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
struct cudaFuncAttributes { int binaryVersion, ptxVersion, numRegs; };
struct cudaDeviceProp { char name[256]; int major, minor, multiProcessorCount; size_t totalGlobalMem; };
cudaError_t cudaMalloc(void**, size_t);
template<class T> cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc((void**)p, n); }
cudaError_t cudaFree(void*);
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
cudaError_t cudaMemset(void*, int, size_t);
cudaError_t cudaGetLastError();
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaGetDeviceCount(int*);
cudaError_t cudaDriverGetVersion(int*);
cudaError_t cudaRuntimeGetVersion(int*);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);
cudaError_t fakeFuncAttr(cudaFuncAttributes*);
template<class T> cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, T*) { return fakeFuncAttr(a); }
const char* cudaGetErrorName(cudaError_t);
const char* cudaGetErrorString(cudaError_t);

#define __CUDA_ARCH_LIST__ 860
