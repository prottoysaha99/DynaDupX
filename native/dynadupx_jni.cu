// DynaDup-X CUDA back end (JNI).  Build:
//   nvcc -O3 -std=c++14 -shared -Xcompiler -fPIC \
//        -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/linux" \
//        -o libdynadupx.so native/dynadupx_jni.cu
// Run:  java -Djava.library.path=. -jar mgd.jar ... -gpu
// Use ./build_gpu.sh: it compiles machine code (SASS) for the GPU of the build machine. A library with only PTX
// fails with CUDA error 222 when the GPU driver is older than nvcc (the driver would have to compile the PTX).
//
// Two kernels, mirroring phylonet/coalescent/fast/Scorer.java exactly:
//
//  weightsKernel : W_dom(x) for candidate bipartitions x = (P | Q).
//      One block per candidate. The block marks P's species with stamp sa and Q's with sb in its private
//      slice of a global scratch array, then threads stride over the unique valid gene bipartitions y.
//      A 256-bit signature prefilter rejects most y in a few instructions; survivors get the exact test
//      (every distinct species of one side of y marked sa, of the other marked sb).
//
//  compsKernel : comps_t(Z) = (leaves of tree t in Z) - (internal nodes of t whose species lie in Z)
//      = number of maximal Z-pure subtrees.  One block per cluster; per-block per-tree accumulators in
//      global scratch; threads stride over species (leaf counts) and over internal-node clusters (pure
//      node counts), both with atomicAdd.
//
// Marks and accumulators use monotonically increasing stamps, so scratch never needs clearing between
// launches. Distinct species of a range [l, r] are the positions p with prev[p] < l (multi-copy safe).

#include <cuda_runtime.h>
#include <jni.h>
#include <stdio.h>
#include <string.h>

typedef unsigned long long u64;

#define CUDA_OK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) return (int)_e; } while (0)

struct Ctx {
    // gene trees
    int *leafAll, *prevAll, *treeOff; int nTaxa, nTrees;
    // unique valid gene bipartitions (sides as ranges, or tree = -1 with l = species)
    int *yaT, *yaL, *yaR, *ybT, *ybL, *ybR; long long *yFreq; u64 *ySig; int nY;
    // internal-node clusters (ranges) and their per-tree occurrences
    int *nT, *nL, *nR, *nSize, *nOccStart, *nOccTree, *nOccCnt; u64 *nSig; int nNode;
    // species -> (tree, copies)
    int *spStart, *spTree, *spCnt;
    u64 *xSig;                     // per-batch signatures (8 per candidate / 4 per cluster)
    // batch buffers
    int *xOff, *xMid, *xTaxa; long long *wOut; int *compsOut;
    int *markScratch;        // grid * nTaxa
    int *acc;                // grid * 2 * nTrees  (leaf counts, pure-node counts)
    int grid, block, maxBatchX, maxBatchC, maxTaxaBuf, maxSigWords;
    int stampBase;
};

__device__ __forceinline__ bool sub4(const u64* p, const u64* q) {
    return ((p[0] & ~q[0]) | (p[1] & ~q[1]) | (p[2] & ~q[2]) | (p[3] & ~q[3])) == 0ULL;
}

// every distinct species of side (t, l, r) carries mark == want
__device__ __forceinline__ bool allMarked(const int* leafAll, const int* prevAll, const int* treeOff,
                                          const int* mark, int t, int l, int r, int want) {
    if (t < 0) return mark[l] == want;                       // single-species side
    const int* L = leafAll + treeOff[t];
    const int* P = prevAll + treeOff[t];
    for (int p = l; p <= r; ++p)
        if (P[p] < l && mark[L[p]] != want) return false;
    return true;
}

__global__ void weightsKernel(const int* __restrict__ leafAll, const int* __restrict__ prevAll,
                              const int* __restrict__ treeOff, int nTaxa,
                              const int* __restrict__ yaT, const int* __restrict__ yaL, const int* __restrict__ yaR,
                              const int* __restrict__ ybT, const int* __restrict__ ybL, const int* __restrict__ ybR,
                              const long long* __restrict__ yFreq, const u64* __restrict__ ySig, int nY,
                              const int* __restrict__ xOff, const int* __restrict__ xMid,
                              const int* __restrict__ xTaxa, const u64* __restrict__ xSig,
                              int nX, int stampBase, int* __restrict__ scratch, long long* __restrict__ out) {
    extern __shared__ long long red[];
    int* mark = scratch + (size_t)blockIdx.x * nTaxa;
    int iter = 0;
    for (int x = blockIdx.x; x < nX; x += gridDim.x, ++iter) {
        const int sa = stampBase + 2 * iter + 1, sb = sa + 1;
        const u64* sA = xSig + 8 * (size_t)x;            // 256-bit signatures of the two sides, built on the CPU
        const u64* sB = sA + 4;
        const int a0 = xOff[x], am = xMid[x], a1 = xOff[x + 1];
        for (int i = a0 + threadIdx.x; i < a1; i += blockDim.x) mark[xTaxa[i]] = (i < am) ? sa : sb;
        __syncthreads();
        long long acc = 0;
        for (int y = threadIdx.x; y < nY; y += blockDim.x) {
            const u64* pa = ySig + 8 * (size_t)y;
            const u64* pb = pa + 4;
            const bool ab = sub4(pa, sA) && sub4(pb, sB);
            const bool ba = sub4(pa, sB) && sub4(pb, sA);
            if (!ab && !ba) continue;
            // The signature test is only a prefilter (a signature subset does not prove a species subset),
            // so BOTH surviving orientations must be checked exactly; at most one of them can hold.
            bool hit = ab && allMarked(leafAll, prevAll, treeOff, mark, yaT[y], yaL[y], yaR[y], sa)
                          && allMarked(leafAll, prevAll, treeOff, mark, ybT[y], ybL[y], ybR[y], sb);
            if (!hit && ba)
                hit = allMarked(leafAll, prevAll, treeOff, mark, yaT[y], yaL[y], yaR[y], sb)
                   && allMarked(leafAll, prevAll, treeOff, mark, ybT[y], ybL[y], ybR[y], sa);
            if (hit) acc += yFreq[y];
        }
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) out[x] = red[0];
        __syncthreads();
    }
}

__global__ void compsKernel(const int* __restrict__ leafAll, const int* __restrict__ prevAll,
                            const int* __restrict__ treeOff, int nTaxa, int nTrees,
                            const int* __restrict__ nT, const int* __restrict__ nL, const int* __restrict__ nR,
                            const u64* __restrict__ nSig, const int* __restrict__ nSize,
                            const int* __restrict__ nOccStart, const int* __restrict__ nOccTree,
                            const int* __restrict__ nOccCnt, int nNode,
                            const int* __restrict__ spStart, const int* __restrict__ spTree,
                            const int* __restrict__ spCnt, const u64* __restrict__ cSig,
                            const int* __restrict__ cOff, const int* __restrict__ cTaxa, int nC,
                            int stampBase, int* __restrict__ scratch, int* __restrict__ acc,
                            int* __restrict__ out) {
    int* mark = scratch + (size_t)blockIdx.x * nTaxa;
    int* Lk = acc + (size_t)blockIdx.x * 2 * nTrees;
    int* Ik = Lk + nTrees;
    int iter = 0;
    for (int c = blockIdx.x; c < nC; c += gridDim.x, ++iter) {
        const int stamp = stampBase + iter + 1;
        const u64* sC = cSig + 4 * (size_t)c;            // signature of the cluster, built on the CPU
        for (int t = threadIdx.x; t < nTrees; t += blockDim.x) { Lk[t] = 0; Ik[t] = 0; }
        __syncthreads();
        const int c0 = cOff[c], c1 = cOff[c + 1], size = c1 - c0;
        for (int i = c0 + threadIdx.x; i < c1; i += blockDim.x) {
            const int v = cTaxa[i];
            mark[v] = stamp;
            for (int e = spStart[v]; e < spStart[v + 1]; ++e) atomicAdd(&Lk[spTree[e]], spCnt[e]);   // leaves in Z
        }
        __syncthreads();
        for (int K = threadIdx.x; K < nNode; K += blockDim.x) {                                      // Z-pure internal nodes
            if (nSize[K] > size) continue;
            if (!sub4(nSig + 4 * (size_t)K, sC)) continue;
            if (!allMarked(leafAll, prevAll, treeOff, mark, nT[K], nL[K], nR[K], stamp)) continue;
            for (int e = nOccStart[K]; e < nOccStart[K + 1]; ++e) atomicAdd(&Ik[nOccTree[e]], nOccCnt[e]);
        }
        __syncthreads();
        int* row = out + (size_t)c * nTrees;
        for (int t = threadIdx.x; t < nTrees; t += blockDim.x) row[t] = Lk[t] - Ik[t];
        __syncthreads();
    }
}

// one thread writes 42: used by the self-test to prove that kernels of this library load and run on this GPU
__global__ void probeKernel(int* out) { if (blockIdx.x == 0 && threadIdx.x == 0) *out = 42; }

// ---------------------------------------------------------------- JNI plumbing
static int* upI(JNIEnv* env, jintArray a, int* nOut) {
    jsize n = env->GetArrayLength(a); if (nOut) *nOut = (int)n;
    int* d = nullptr;
    if (cudaMalloc(&d, (size_t)n * sizeof(int)) != cudaSuccess) return nullptr;
    jint* h = env->GetIntArrayElements(a, nullptr);
    cudaMemcpy(d, h, (size_t)n * sizeof(int), cudaMemcpyHostToDevice);
    env->ReleaseIntArrayElements(a, h, JNI_ABORT);
    return d;
}
static long long* upL(JNIEnv* env, jlongArray a) {
    jsize n = env->GetArrayLength(a);
    long long* d = nullptr;
    if (cudaMalloc(&d, (size_t)n * sizeof(long long)) != cudaSuccess) return nullptr;
    jlong* h = env->GetLongArrayElements(a, nullptr);
    cudaMemcpy(d, h, (size_t)n * sizeof(long long), cudaMemcpyHostToDevice);
    env->ReleaseLongArrayElements(a, h, JNI_ABORT);
    return d;
}

extern "C" {

JNIEXPORT jlong JNICALL Java_phylonet_coalescent_fast_Gpu_init(
        JNIEnv* env, jclass, jintArray leafAll, jintArray prevAll, jintArray treeOff, jint nTaxa, jint nTrees,
        jintArray yaT, jintArray yaL, jintArray yaR, jintArray ybT, jintArray ybL, jintArray ybR,
        jlongArray yFreq, jlongArray ySig, jint nY,
        jintArray nT, jintArray nL, jintArray nR, jlongArray nSig, jintArray nSize,
        jintArray nOccStart, jintArray nOccTree, jintArray nOccCnt, jint nNode,
        jintArray spStart, jintArray spTree, jintArray spCnt, jint grid, jint block) {
    Ctx* c = new Ctx();
    memset(c, 0, sizeof(Ctx));
    c->nTaxa = nTaxa; c->nTrees = nTrees; c->nY = nY; c->nNode = nNode;
    c->grid = grid; c->block = block; c->stampBase = 0;
    c->leafAll = upI(env, leafAll, nullptr); c->prevAll = upI(env, prevAll, nullptr); c->treeOff = upI(env, treeOff, nullptr);
    c->yaT = upI(env, yaT, nullptr); c->yaL = upI(env, yaL, nullptr); c->yaR = upI(env, yaR, nullptr);
    c->ybT = upI(env, ybT, nullptr); c->ybL = upI(env, ybL, nullptr); c->ybR = upI(env, ybR, nullptr);
    c->yFreq = upL(env, yFreq); c->ySig = (u64*)upL(env, ySig);
    c->nT = upI(env, nT, nullptr); c->nL = upI(env, nL, nullptr); c->nR = upI(env, nR, nullptr);
    c->nSig = (u64*)upL(env, nSig); c->nSize = upI(env, nSize, nullptr);
    c->nOccStart = upI(env, nOccStart, nullptr); c->nOccTree = upI(env, nOccTree, nullptr); c->nOccCnt = upI(env, nOccCnt, nullptr);
    c->spStart = upI(env, spStart, nullptr); c->spTree = upI(env, spTree, nullptr); c->spCnt = upI(env, spCnt, nullptr);
    if (!c->leafAll || !c->prevAll || !c->treeOff || !c->yFreq || !c->ySig) { delete c; return 0; }
    if (cudaMalloc(&c->markScratch, (size_t)grid * nTaxa * sizeof(int)) != cudaSuccess) { delete c; return 0; }
    cudaMemset(c->markScratch, 0, (size_t)grid * nTaxa * sizeof(int));
    if (cudaMalloc(&c->acc, (size_t)grid * 2 * nTrees * sizeof(int)) != cudaSuccess) { delete c; return 0; }
    c->maxBatchX = 0; c->maxBatchC = 0; c->maxTaxaBuf = 0; c->maxSigWords = 0;
    if (cudaGetLastError() != cudaSuccess) { delete c; return 0; }
    return (jlong)c;
}

static int ensureBatch(Ctx* c, int nX, int nTaxaBuf, int nC) {
    if (nTaxaBuf > c->maxTaxaBuf) {
        cudaFree(c->xTaxa); c->xTaxa = nullptr;
        CUDA_OK(cudaMalloc(&c->xTaxa, (size_t)nTaxaBuf * sizeof(int)));
        c->maxTaxaBuf = nTaxaBuf;
    }
    int sigWords = (nX > nC ? nX : nC) * 8;
    if (sigWords > c->maxSigWords) {
        cudaFree(c->xSig); c->xSig = nullptr;
        CUDA_OK(cudaMalloc(&c->xSig, (size_t)sigWords * sizeof(u64)));
        c->maxSigWords = sigWords;
    }
    if (nX > c->maxBatchX) {
        cudaFree(c->xOff); cudaFree(c->xMid); cudaFree(c->wOut);
        c->xOff = c->xMid = nullptr; c->wOut = nullptr;
        CUDA_OK(cudaMalloc(&c->xOff, (size_t)(nX + 1) * sizeof(int)));
        CUDA_OK(cudaMalloc(&c->xMid, (size_t)nX * sizeof(int)));
        CUDA_OK(cudaMalloc(&c->wOut, (size_t)nX * sizeof(long long)));
        c->maxBatchX = nX;
    }
    if (nC > c->maxBatchC) {
        cudaFree(c->compsOut); c->compsOut = nullptr;
        CUDA_OK(cudaMalloc(&c->compsOut, (size_t)nC * c->nTrees * sizeof(int)));
        c->maxBatchC = nC;
    }
    return 0;
}

JNIEXPORT jint JNICALL Java_phylonet_coalescent_fast_Gpu_weights(
        JNIEnv* env, jclass, jlong handle, jintArray xOff, jintArray xMid, jintArray xTaxa, jlongArray xSig, jint nX, jlongArray out) {
    Ctx* c = (Ctx*)handle;
    jsize nBuf = env->GetArrayLength(xTaxa);
    int rc = ensureBatch(c, nX, (int)nBuf, 0);
    if (rc) return rc;
    jint* hOff = env->GetIntArrayElements(xOff, nullptr);
    jint* hMid = env->GetIntArrayElements(xMid, nullptr);
    jint* hTx  = env->GetIntArrayElements(xTaxa, nullptr);
    cudaMemcpy(c->xOff, hOff, (size_t)(nX + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c->xMid, hMid, (size_t)nX * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c->xTaxa, hTx, (size_t)nBuf * sizeof(int), cudaMemcpyHostToDevice);
    env->ReleaseIntArrayElements(xOff, hOff, JNI_ABORT);
    env->ReleaseIntArrayElements(xMid, hMid, JNI_ABORT);
    env->ReleaseIntArrayElements(xTaxa, hTx, JNI_ABORT);
    jlong* hSig = env->GetLongArrayElements(xSig, nullptr);
    cudaMemcpy(c->xSig, hSig, (size_t)nX * 8 * sizeof(u64), cudaMemcpyHostToDevice);
    env->ReleaseLongArrayElements(xSig, hSig, JNI_ABORT);
    int grid = c->grid < nX ? c->grid : nX;
    weightsKernel<<<grid, c->block, c->block * sizeof(long long)>>>(
        c->leafAll, c->prevAll, c->treeOff, c->nTaxa, c->yaT, c->yaL, c->yaR, c->ybT, c->ybL, c->ybR,
        c->yFreq, c->ySig, c->nY, c->xOff, c->xMid, c->xTaxa, c->xSig, nX, c->stampBase, c->markScratch, c->wOut);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    c->stampBase += 2 * (nX / (grid > 0 ? grid : 1) + 2);     // keep stamps strictly increasing across launches
    jlong* hOut = env->GetLongArrayElements(out, nullptr);
    cudaMemcpy(hOut, c->wOut, (size_t)nX * sizeof(long long), cudaMemcpyDeviceToHost);
    env->ReleaseLongArrayElements(out, hOut, 0);
    return 0;
}

JNIEXPORT jint JNICALL Java_phylonet_coalescent_fast_Gpu_comps(
        JNIEnv* env, jclass, jlong handle, jintArray cOff, jintArray cTaxa, jlongArray cSig, jint nC, jintArray out) {
    Ctx* c = (Ctx*)handle;
    jsize nBuf = env->GetArrayLength(cTaxa);
    int rc = ensureBatch(c, 0, (int)nBuf, nC);
    if (rc) return rc;
    jint* hOff = env->GetIntArrayElements(cOff, nullptr);
    jint* hTx  = env->GetIntArrayElements(cTaxa, nullptr);
    int* dOff = nullptr;
    if (cudaMalloc(&dOff, (size_t)(nC + 1) * sizeof(int)) != cudaSuccess) return (int)cudaErrorMemoryAllocation;
    cudaMemcpy(dOff, hOff, (size_t)(nC + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c->xTaxa, hTx, (size_t)nBuf * sizeof(int), cudaMemcpyHostToDevice);
    env->ReleaseIntArrayElements(cOff, hOff, JNI_ABORT);
    env->ReleaseIntArrayElements(cTaxa, hTx, JNI_ABORT);
    jlong* hSig = env->GetLongArrayElements(cSig, nullptr);
    cudaMemcpy(c->xSig, hSig, (size_t)nC * 4 * sizeof(u64), cudaMemcpyHostToDevice);
    env->ReleaseLongArrayElements(cSig, hSig, JNI_ABORT);
    int grid = c->grid < nC ? c->grid : nC;
    compsKernel<<<grid, c->block>>>(
        c->leafAll, c->prevAll, c->treeOff, c->nTaxa, c->nTrees,
        c->nT, c->nL, c->nR, c->nSig, c->nSize, c->nOccStart, c->nOccTree, c->nOccCnt, c->nNode,
        c->spStart, c->spTree, c->spCnt, c->xSig, dOff, c->xTaxa, nC, c->stampBase,
        c->markScratch, c->acc, c->compsOut);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    cudaFree(dOff);
    if (e != cudaSuccess) return (int)e;
    c->stampBase += (nC / (grid > 0 ? grid : 1) + 2);
    jint* hOut = env->GetIntArrayElements(out, nullptr);
    cudaMemcpy(hOut, c->compsOut, (size_t)nC * c->nTrees * sizeof(int), cudaMemcpyDeviceToHost);
    env->ReleaseIntArrayElements(out, hOut, 0);
    return 0;
}

JNIEXPORT void JNICALL Java_phylonet_coalescent_fast_Gpu_free(JNIEnv*, jclass, jlong handle) {
    Ctx* c = (Ctx*)handle;
    if (!c) return;
    int* p[] = {c->leafAll, c->prevAll, c->treeOff, c->yaT, c->yaL, c->yaR, c->ybT, c->ybL, c->ybR,
                c->nT, c->nL, c->nR, c->nSize, c->nOccStart, c->nOccTree, c->nOccCnt,
                c->spStart, c->spTree, c->spCnt, c->xOff, c->xMid, c->xTaxa, c->compsOut, c->markScratch, c->acc};
    for (size_t i = 0; i < sizeof(p) / sizeof(p[0]); ++i) cudaFree(p[i]);
    cudaFree(c->yFreq); cudaFree(c->ySig); cudaFree(c->nSig); cudaFree(c->wOut); cudaFree(c->xSig);
    delete c;
}

// Self-test, run before any real work. Loading the kernels is where a library that carries only PTX fails when the
// driver is older than the nvcc that wrote the PTX (cudaErrorUnsupportedPtxVersion = 222). Returns 0 or the error.
JNIEXPORT jint JNICALL Java_phylonet_coalescent_fast_Gpu_probe(JNIEnv*, jclass) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) { cudaGetLastError(); return (int)e; }
    if (n <= 0) return (int)cudaErrorNoDevice;
    cudaFuncAttributes fa;
    if ((e = cudaFuncGetAttributes(&fa, weightsKernel)) != cudaSuccess) { cudaGetLastError(); return (int)e; }
    if ((e = cudaFuncGetAttributes(&fa, compsKernel)) != cudaSuccess) { cudaGetLastError(); return (int)e; }
    int* d = nullptr; int h = 0;
    if ((e = cudaMalloc(&d, sizeof(int))) != cudaSuccess) { cudaGetLastError(); return (int)e; }
    probeKernel<<<1, 1>>>(d);
    e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e == cudaSuccess) e = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);
    if (e != cudaSuccess) { cudaGetLastError(); return (int)e; }
    return h == 42 ? 0 : (int)cudaErrorUnknown;
}

// "NVIDIA ... (compute capability 8.6, ...); driver supports CUDA 12.2; library built with CUDA 12.6 for compute capability 8.6"
JNIEXPORT jstring JNICALL Java_phylonet_coalescent_fast_Gpu_info(JNIEnv* env, jclass) {
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    char dev[320];
    snprintf(dev, sizeof(dev), "no CUDA device visible");
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) == cudaSuccess)
        snprintf(dev, sizeof(dev), "%s (compute capability %d.%d, %d SMs, %.1f GB)", p.name, p.major, p.minor,
                 p.multiProcessorCount, p.totalGlobalMem / 1073741824.0);
    cudaGetLastError();
    char archs[200];
    archs[0] = 0;
#ifdef __CUDA_ARCH_LIST__
    static const int list[] = { __CUDA_ARCH_LIST__ };      // e.g. 860 for sm_86 (nvcc >= 11.5)
    for (size_t i = 0; i < sizeof(list) / sizeof(list[0]); ++i) {
        size_t used = strlen(archs);
        snprintf(archs + used, sizeof(archs) - used, "%s%d.%d", i ? " " : "", list[i] / 100, (list[i] % 100) / 10);
    }
#else
    snprintf(archs, sizeof(archs), "? (nvcc older than 11.5)");
#endif
    char buf[640];
    snprintf(buf, sizeof(buf), "%s; driver supports CUDA %d.%d; library built with CUDA %d.%d for compute capability %s",
             dev, drv / 1000, (drv % 1000) / 10, rt / 1000, (rt % 1000) / 10, archs);
    return env->NewStringUTF(buf);
}

// "cudaErrorUnsupportedPtxVersion: the provided PTX was compiled with an unsupported toolchain."
JNIEXPORT jstring JNICALL Java_phylonet_coalescent_fast_Gpu_errorText(JNIEnv* env, jclass, jint rc) {
    char buf[400];
    snprintf(buf, sizeof(buf), "%s: %s", cudaGetErrorName((cudaError_t)rc), cudaGetErrorString((cudaError_t)rc));
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL Java_phylonet_coalescent_fast_Gpu_deviceName(JNIEnv* env, jclass) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return env->NewStringUTF("unknown CUDA device");
    char buf[320];
    snprintf(buf, sizeof(buf), "%s (%d SMs, %.1f GB)", prop.name, prop.multiProcessorCount,
             prop.totalGlobalMem / 1073741824.0);
    return env->NewStringUTF(buf);
}

} // extern "C"
