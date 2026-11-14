package phylonet.coalescent.fast;

import java.util.Random;

/**
 * CPU replica of weightsKernel (mark array, 256-bit prefilter, both orientations checked exactly), used to
 * validate the kernel logic without a GPU: -Ddynadupx.gpu.simulate=<sample size>.
 * Marks are reused across candidates with increasing stamps, as one CUDA block does, so stale marks from
 * earlier candidates are exercised too. Compared against a brute-force weight computed from explicit sets.
 */
final class GpuSim {
    static void run(ClusterTable ct, GeneData g, int[] cA, int[] cB, int sample, long seed) {
        final int nY = g.yA.length;
        int[] mark = new int[g.n];
        Random rnd = new Random(seed);
        boolean compOnly = System.getProperty("dynadupx.gpu.simulateComp") != null;
        IntList pool = new IntList();
        for (int i = 0; i < cA.length; i++)
            if (!compOnly || ct.compOf[cA[i]] >= 0 || ct.compOf[cB[i]] >= 0) pool.add(i);
        System.err.println("[gpusim] candidate pool: " + pool.size() + (compOnly ? " (with a complement side)" : ""));
        if (pool.size() == 0) return;
        int stamp = 1, checked = 0, bad = 0;
        for (int it = 0; it < sample; it++) {
            int c = pool.get(rnd.nextInt(pool.size()));
            int[] A = ct.taxa(cA[c]), B = ct.taxa(cB[c]);
            int sa = stamp++, sb = stamp++;
            for (int v : A) mark[v] = sa;
            for (int v : B) mark[v] = sb;
            long[] sA = new long[4], sB = new long[4]; ct.sig256(cA[c], sA, 0); ct.sig256(cB[c], sB, 0);
            long kernel = 0;
            for (int y = 0; y < nY; y++) {
                int ia = g.yA[y], ib = g.yB[y];
                long[] sgA = new long[4], sgB = new long[4]; ct.sig256(ia, sgA, 0); ct.sig256(ib, sgB, 0);
                boolean ab = sub4(sgA, 0, sA) && sub4(sgB, 0, sB);
                boolean ba = sub4(sgA, 0, sB) && sub4(sgB, 0, sA);
                if (!ab && !ba) continue;
                boolean hit = ab && allMarked(ct, g, ia, mark, sa) && allMarked(ct, g, ib, mark, sb);
                if (!hit && ba) hit = allMarked(ct, g, ia, mark, sb) && allMarked(ct, g, ib, mark, sa);
                if (hit) kernel += g.yF[y];
            }
            long brute = 0;                                   // independent reference: explicit set containment
            java.util.HashSet<Integer> SA = new java.util.HashSet<>(), SB = new java.util.HashSet<>();
            for (int v : A) SA.add(v);
            for (int v : B) SB.add(v);
            for (int y = 0; y < nY; y++) {
                java.util.HashSet<Integer> Ya = new java.util.HashSet<>(), Yb = new java.util.HashSet<>();
                for (int v : ct.taxa(g.yA[y])) Ya.add(v);
                for (int v : ct.taxa(g.yB[y])) Yb.add(v);
                if ((SA.containsAll(Ya) && SB.containsAll(Yb)) || (SB.containsAll(Ya) && SA.containsAll(Yb))) brute += g.yF[y];
            }
            checked++;
            if (kernel != brute) {
                bad++;
                if (bad <= 5) System.err.println("[gpusim] candidate " + c + ": kernel=" + kernel + " brute=" + brute
                        + " |A|=" + A.length + " |B|=" + B.length);
            }
        }
        System.err.println("[gpusim] " + checked + " candidates checked, " + bad + " mismatches");
    }

    private static boolean sub4(long[] sig, int off, long[] out) {
        return (sig[off] & ~out[0]) == 0 && (sig[off + 1] & ~out[1]) == 0
            && (sig[off + 2] & ~out[2]) == 0 && (sig[off + 3] & ~out[3]) == 0;
    }
    /** exactly what allMarked() in the kernel does: range (or single species) with prev-based distinct scan */
    private static boolean allMarked(ClusterTable ct, GeneData g, int id, int[] mark, int want) {
        if (ct.repTree[id] >= 0) {
            int t = ct.repTree[id], l = ct.repL[id], r = ct.repR[id];
            int[] L = g.leaves()[t], P = g.prevs()[t];
            for (int p = l; p <= r; p++) if (P[p] < l && mark[L[p]] != want) return false;
            return true;
        }
        return mark[ct.taxa(id)[0]] == want;
    }
    private GpuSim() {}
}
