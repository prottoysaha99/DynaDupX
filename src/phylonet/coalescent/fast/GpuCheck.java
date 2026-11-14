package phylonet.coalescent.fast;

/**
 * Checks that libdynadupx.so works on this machine's GPU (run by build_gpu.sh after building):
 *   java -Djava.library.path=<dir of libdynadupx.so> -cp mgd.jar phylonet.coalescent.fast.GpuCheck
 * Exit code 0 = DynaDup-X can use the GPU with -gpu; otherwise the reason and the fix are printed.
 */
public final class GpuCheck {
    public static void main(String[] args) {
        if (!Gpu.AVAILABLE) {
            System.err.println("[gpu] CUDA library not loaded: " + Gpu.loadError);
            System.exit(2);
        }
        String problem = Gpu.selfTest();
        if (problem == null) {
            System.err.println("[gpu] self-test passed: DynaDup-X can use this GPU (-gpu)");
            System.exit(0);
        }
        System.err.println("[gpu] self-test FAILED: " + problem);
        System.exit(1);
    }

    private GpuCheck() {}
}
