package phylonet.coalescent.fast;

/**
 * JNI binding to the optional CUDA library (native/dynadupx_jni.cu -> libdynadupx.so).
 *
 * Two stages are offloaded, both "one block per item, threads stride over the work":
 *   weights() : MGD domination weight W_dom(x) of every candidate bipartition
 * Each gene bipartition side is either a post-order range (tree, l, r) or a single species (tree = -1, l = species).
 *   comps()   : per-gene-tree counts of maximal Z-pure subtrees for a batch of clusters (MGDL)
 *
 * Side/cluster signatures are built on the CPU (they already exist in ClusterTable) and passed in, so the
 * kernels need no 64-bit atomics and run on any CUDA architecture.
 *
 * The library is optional. If it cannot be loaded, AVAILABLE is false and the caller stays on the CPU.
 * Load order: -Ddynadupx.lib=/path/libdynadupx.so, else java.library.path ("dynadupx").
 */
final class Gpu {
    static final boolean AVAILABLE;
    static String loadError;
    static {
        boolean ok = false;
        try {
            String p = System.getProperty("dynadupx.lib");
            if (p != null) System.load(p); else System.loadLibrary("dynadupx");
            ok = true;
        } catch (Throwable t) { loadError = t.toString(); }
        AVAILABLE = ok;
    }

    /** Uploads the read-only data once. Returns an opaque handle, or 0 on failure. */
    static native long init(int[] leafAll, int[] prevAll, int[] treeOff, int nTaxa, int nTrees,
                            int[] yaTree, int[] yaL, int[] yaR, int[] ybTree, int[] ybL, int[] ybR,
                            long[] yFreq, long[] ySig, int nY,
                            int[] nTree, int[] nL, int[] nR, long[] nSig, int[] nSize,
                            int[] nOccStart, int[] nOccTree, int[] nOccCnt, int nNode,
                            int[] spOccStart, int[] spOccTree, int[] spOccCnt,
                            int gridBlocks, int blockDim);

    /** W_dom for candidates given as species lists: side A = xTaxa[xOff[i]..xMid[i]), side B = ..xOff[i+1]). */
    static native int weights(long h, int[] xOff, int[] xMid, int[] xTaxa, long[] xSig, int nX, long[] out);

    /** comps_t(Z) for clusters given as species lists; out has nC * nTrees ints (row per cluster). */
    static native int comps(long h, int[] cOff, int[] cTaxa, long[] cSig, int nC, int[] out);

    static native void free(long h);
    static native String deviceName();

    /** Loads both kernels and runs a one-thread kernel. 0 = the GPU works, else a cudaError_t code. */
    static native int probe();
    /** GPU name and compute capability, CUDA version of the driver, CUDA version and targets the library was built for. */
    static native String info();
    /** cudaGetErrorName + cudaGetErrorString of a cudaError_t code. */
    static native String errorText(int rc);

    /** Runs the library's self-test and prints what was found. Returns null if the GPU can be used, otherwise the
     *  reason and the fix. */
    static String selfTest() {
        int rc;
        try {
            System.err.println("[gpu] " + info());
            rc = probe();
        } catch (UnsatisfiedLinkError e) {
            return "libdynadupx.so was built from an older DynaDup-X source (it has no self-test); rebuild it with "
                    + "./build_gpu.sh";
        }
        return rc == 0 ? null : describe(rc);
    }

    /** "CUDA error 222 (cudaErrorUnsupportedPtxVersion: ...)" followed by what to do about it. */
    static String describe(int rc) {
        String text = null;
        try { text = errorText(rc); } catch (Throwable ignored) { }
        String name = text == null ? "" : text;
        StringBuilder sb = new StringBuilder("CUDA error ").append(rc);
        if (text != null) sb.append(" (").append(text.trim().replaceAll("\\.$", "")).append(")");
        if (rc == 222 || rc == 218 || rc == 221 || rc == 223 || rc == 209 || rc == 200
                || name.contains("Ptx") || name.contains("KernelImage") || name.contains("Jit")) {
            sb.append(". The library has no machine code for this GPU, and the GPU driver cannot compile the library's"
                    + " PTX (the nvcc that built it is newer than the driver). Fix: rebuild it on this machine with"
                    + " ./build_gpu.sh, which now builds machine code for the GPU it finds (or pass -arch=sm_XY, XY ="
                    + " the compute capability printed above without the dot), or build with a CUDA toolkit no newer"
                    + " than the driver's CUDA version printed above");
        } else if (rc == 35 || name.contains("InsufficientDriver")) {
            sb.append(". The GPU driver is older than the CUDA toolkit the library was built with. Fix: build with a"
                    + " toolkit of the driver's CUDA version (e.g. module load cuda/<that version>) or update the driver");
        } else if (rc == 100 || name.contains("NoDevice")) {
            sb.append(". No GPU is visible to this process (check that the job was given a GPU and CUDA_VISIBLE_DEVICES)");
        } else if (rc == 2 || name.contains("MemoryAllocation")) {
            sb.append(". The GPU ran out of memory (lower -Ddynadupx.gpu.batch or -Ddynadupx.gpu.blocks)");
        }
        return sb.toString();
    }

    private Gpu() {}
}
