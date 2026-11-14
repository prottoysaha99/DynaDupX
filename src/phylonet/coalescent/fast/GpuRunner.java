package phylonet.coalescent.fast;

import java.util.Arrays;

/**
 * Packs the CPU-side structures for the CUDA library and runs the two offloaded stages in batches:
 *   1. W_dom for all candidate bipartitions;
 *   2. comps_t(Z) (maximal Z-pure subtree counts) of the DP states, one layer batch at a time (MGDL).
 * Any failure (library missing, CUDA error, out of memory) leaves the CPU path untouched.
 */
final class GpuRunner {
    static final int GRID = Integer.getInteger("dynadupx.gpu.blocks", 1024);
    static final int BLOCK = Integer.getInteger("dynadupx.gpu.threads", 256);

    private final ClusterTable ct; private final GeneData g;
    private long handle;

    GpuRunner(ClusterTable ct, GeneData g, int[] yA, int[] yB, long[] yF) {
        this.ct = ct; this.g = g;
        int nY = yA.length;
        int[] yaT = new int[nY], yaL = new int[nY], yaR = new int[nY], ybT = new int[nY], ybL = new int[nY], ybR = new int[nY];
        long[] ySig = new long[8 * nY];
        for (int y = 0; y < nY; y++) {
            side(yA[y], y, yaT, yaL, yaR); side(yB[y], y, ybT, ybL, ybR);
            ct.sig256(yA[y], ySig, 8 * y);
            ct.sig256(yB[y], ySig, 8 * y + 4);
        }
        // internal-node clusters (ranges) with their per-tree occurrence lists
        IntList nt = new IntList(), nl = new IntList(), nr = new IntList(), nsz = new IntList(),
                occStart = new IntList(), occTree = new IntList(), occCnt = new IntList();
        java.util.ArrayList<long[]> sigs = new java.util.ArrayList<>();
        for (int id = 0; id < ct.n; id++) {
            if (g.nodeTree[id] == null || ct.repTree[id] < 0) continue;   // only range-represented node clusters
            occStart.add(occTree.size());
            for (int e = 0; e < g.nodeTree[id].length; e++) { occTree.add(g.nodeTree[id][e]); occCnt.add(g.nodeCnt[id][e]); }
            nt.add(ct.repTree[id]); nl.add(ct.repL[id]); nr.add(ct.repR[id]); nsz.add(ct.size[id]);
            long[] sg4 = new long[4]; ct.sig256(id, sg4, 0); sigs.add(sg4);
        }
        occStart.add(occTree.size());
        int nNode = nt.size();
        long[] nSig = new long[4 * nNode];
        for (int i = 0; i < nNode; i++) System.arraycopy(sigs.get(i), 0, nSig, 4 * i, 4);
        // species -> (tree, copies)
        IntList spStart = new IntList(), spTree = new IntList(), spCnt = new IntList();
        for (int v = 0; v < g.n; v++) {
            spStart.add(spTree.size());
            for (int e = 0; e < g.occTree[v].length; e++) { spTree.add(g.occTree[v][e]); spCnt.add(g.occCnt[v][e]); }
        }
        spStart.add(spTree.size());
        handle = Gpu.init(g.leavesAll(), g.prevsAll(), g.treeOffsets(), g.n, g.k,
                yaT, yaL, yaR, ybT, ybL, ybR, yF, ySig, nY,
                nt.toArray(), nl.toArray(), nr.toArray(), nSig, nsz.toArray(),
                occStart.toArray(), occTree.toArray(), occCnt.toArray(), nNode,
                spStart.toArray(), spTree.toArray(), spCnt.toArray(), GRID, BLOCK);
        if (handle == 0) throw new IllegalStateException("GPU initialisation failed");
    }

    /** Checks that every gene bipartition side can be packed for the GPU (used by -Ddynadupx.gpu.packcheck). */
    static void checkSides(ClusterTable ct, int[] yA, int[] yB) {
        int bad = 0;
        for (int y = 0; y < yA.length; y++)
            for (int id : new int[]{yA[y], yB[y]})
                if (ct.repTree[id] < 0 && !(ct.isExplicit(id) && ct.size[id] == 1)) {
                    if (bad++ == 0) System.err.println("[packcheck] side cluster " + id + " size=" + ct.size[id]
                            + " repTree=" + ct.repTree[id] + " explicit=" + ct.isExplicit(id));
                }
        System.err.println("[packcheck] unpackable sides: " + bad + " of " + (2 * yA.length));
    }

    /** A gene bipartition side: a post-order range, or (tree = -1, l = species) for a single species. */
    private void side(int id, int y, int[] T, int[] L, int[] R) {
        if (ct.repTree[id] >= 0) { T[y] = ct.repTree[id]; L[y] = ct.repL[id]; R[y] = ct.repR[id]; }
        else if (ct.isExplicit(id) && ct.size[id] == 1) { int v = ct.taxa(id)[0]; T[y] = -1; L[y] = v; R[y] = v; }
        else throw new IllegalStateException("gene bipartition side is neither a range nor a single species: cluster " + id);
    }

    /** W_dom for every candidate. */
    long[] weights(int[] cA, int[] cB) {
        int nX = cA.length;
        long[] out = new long[nX];
        int batch = Math.max(1, Integer.getInteger("dynadupx.gpu.batch", 4096));
        for (int lo = 0; lo < nX; lo += batch) {
            int hi = Math.min(nX, lo + batch), m = hi - lo;
            IntList taxa = new IntList(m * 8);
            int[] off = new int[m + 1], mid = new int[m];
            long[] sig = new long[8 * m];                 // signatures of both sides, straight from the cluster table
            for (int i = 0; i < m; i++) {
                off[i] = taxa.size();
                for (int v : ct.taxa(cA[lo + i])) taxa.add(v);
                mid[i] = taxa.size();
                for (int v : ct.taxa(cB[lo + i])) taxa.add(v);
                ct.sig256(cA[lo + i], sig, 8 * i);
                ct.sig256(cB[lo + i], sig, 8 * i + 4);
            }
            off[m] = taxa.size();
            long[] part = new long[m];
            int rc = Gpu.weights(handle, off, mid, taxa.toArray(), sig, m, part);
            if (rc != 0) throw new IllegalStateException("weights kernel: " + Gpu.describe(rc));
            System.arraycopy(part, 0, out, lo, m);
        }
        return out;
    }

    /** comps_t(Z) of one batch of clusters: row i = clusters[i], nTrees ints per row (one DP layer batch at a time,
     *  so host memory stays O(batch x trees) instead of O(clusters x trees)). */
    int[] compsRows(int[] clusters) {
        int m = clusters.length;
        IntList taxa = new IntList(m * 16);
        int[] off = new int[m + 1];
        long[] sig = new long[4 * m];
        for (int i = 0; i < m; i++) {
            off[i] = taxa.size();
            for (int v : ct.taxa(clusters[i])) taxa.add(v);
            ct.sig256(clusters[i], sig, 4 * i);
        }
        off[m] = taxa.size();
        int[] out = new int[m * g.k];
        int rc = Gpu.comps(handle, off, taxa.toArray(), sig, m, out);
        if (rc != 0) throw new IllegalStateException("comps kernel: " + Gpu.describe(rc));
        return out;
    }

    void close() { if (handle != 0) { Gpu.free(handle); handle = 0; } }
}
