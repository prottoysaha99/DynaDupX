// Fake CUDA runtime on the host: run.sh turns each kernel launch into a loop over its blocks (one thread per block).
#include "cuda_runtime.h"
#include <stdlib.h>
#include <string.h>
uint3s threadIdx = {0, 0, 0}, blockIdx = {0, 0, 0}, blockDim = {1, 1, 1}, gridDim = {1, 1, 1};
long long red[1024];
void __syncthreads() {}
int atomicAdd(int* p, int v) { int o = *p; *p += v; return o; }
static int fakeRc() { const char* e = getenv("FAKE_RC"); return e ? atoi(e) : 0; }
cudaError_t fakeFuncAttr(cudaFuncAttributes* a) { a->binaryVersion = 86; a->ptxVersion = 52; return (cudaError_t)fakeRc(); }
cudaError_t cudaMalloc(void** p, size_t n) { *p = calloc(n ? n : 1, 1); return cudaSuccess; }
cudaError_t cudaFree(void* p) { free(p); return cudaSuccess; }
cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { memcpy(d, s, n); return cudaSuccess; }
cudaError_t cudaMemset(void* p, int v, size_t n) { memset(p, v, n); return cudaSuccess; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
cudaError_t cudaGetDeviceCount(int* n) { *n = 1; return cudaSuccess; }
cudaError_t cudaDriverGetVersion(int* v) { *v = 12020; return cudaSuccess; }
cudaError_t cudaRuntimeGetVersion(int* v) { *v = 12060; return cudaSuccess; }
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
    strcpy(p->name, "CPU emulation of a CUDA GPU"); p->major = 8; p->minor = 6; p->multiProcessorCount = 84;
    p->totalGlobalMem = 48ull << 30; return cudaSuccess; }
const char* cudaGetErrorName(cudaError_t e) { return e == 222 ? "cudaErrorUnsupportedPtxVersion" : "cudaErrorOther"; }
const char* cudaGetErrorString(cudaError_t e) {
    return e == 222 ? "the provided PTX was compiled with an unsupported toolchain." : "some other error"; }
