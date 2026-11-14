package phylonet.coalescent.fast;

/**
 * Scan of ONE rooted (possibly multi-copy) gene tree, STELAR-X style.
 *
 * The tree is its post-order leaf array (species ids, repeats allowed) plus (l,m,r) tuples of
 * internal nodes. For every internal node we compute, over DISTINCT species only:
 *   key/min of side A = [l..m], side B = [m+1..r], parent = [l..r]
 * (offline Fenwick sweep; see TreeProcessor in STELAR-X-MC) and the species minimum
 * (sparse-table range-min). A node is a VALID subtree bipartition iff its sides share no species.
 *
 * Invalid (overlapping) nodes are what DynaDup's tryAddingSTB treats as certain duplications;
 * as in DynaDup we additionally emit the clusters A-B and B-A as search-space clusters.
 */
final class GeneScan {
    final int[] leaves, prev, triples;
    /** set signatures of side A, side B and the parent, stored as primitives (no per-node objects) */
    final long[] a1, a2, b1, b2, p1, p2; final int[] zA, zB, zP;
    ClusterKey kA(int u) { return new ClusterKey(a1[u], a2[u], zA[u]); }
    ClusterKey kB(int u) { return new ClusterKey(b1[u], b2[u], zB[u]); }
    ClusterKey kP(int u) { return new ClusterKey(p1[u], p2[u], zP[u]); }
    final int[] minA, minB, minP;
    final boolean[] valid;
    int invalid = 0;
    /** explicit extra clusters (A-B, B-A of invalid nodes): sorted taxon arrays */
    final java.util.ArrayList<int[]> extra = new java.util.ArrayList<>();
    /** per internal node: index in `extra` of its A-B / B-A cluster, -1 if none (only allocated for invalid nodes) */
    int[] exA, exB;

    GeneScan(int[] leaves, int[] triples, int nTaxa, long[] H1, long[] H2) {
        this.leaves = leaves; this.triples = triples;
        final int m = leaves.length, nInt = triples.length / 3, q = 3 * nInt;
        prev = new int[m];
        int[] last = new int[nTaxa];
        java.util.Arrays.fill(last, -1);
        for (int p = 0; p < m; p++) { int v = leaves[p]; prev[p] = last[v]; last[v] = p; }

        int[] ql = new int[q], qr = new int[q];
        for (int u = 0; u < nInt; u++) {
            int l = triples[3 * u], mid = triples[3 * u + 1], r = triples[3 * u + 2];
            ql[3 * u] = l; qr[3 * u] = mid; ql[3 * u + 1] = mid + 1; qr[3 * u + 1] = r; ql[3 * u + 2] = l; qr[3 * u + 2] = r;
        }
        int[] head = new int[m + 1];
        for (int k = 0; k < q; k++) head[qr[k] + 1]++;
        for (int p = 0; p < m; p++) head[p + 1] += head[p];
        int[] byR = new int[q], fill = head.clone();
        for (int k = 0; k < q; k++) byR[fill[qr[k]]++] = k;
        int[] fc = new int[m + 1]; long[] f1 = new long[m + 1], f2 = new long[m + 1];
        int[] rc = new int[q]; long[] r1 = new long[q], r2 = new long[q];
        for (int r = 0; r < m; r++) {
            int v = leaves[r]; long a = H1[v], b = H2[v];
            if (prev[r] >= 0) upd(fc, f1, f2, prev[r], -1, -a, b);
            upd(fc, f1, f2, r, 1, a, b);
            for (int t = head[r]; t < head[r + 1]; t++) {
                int k = byR[t], l = ql[k];
                rc[k] = cnt(fc, r) - cnt(fc, l - 1);
                r1[k] = s1(f1, r) - s1(f1, l - 1);
                r2[k] = x2(f2, r) ^ x2(f2, l - 1);
            }
        }
        // sparse table for range minimum of species ids
        int lg = 1; while ((1 << lg) <= m) lg++;
        int[][] sp = new int[lg][];
        sp[0] = leaves.clone();
        for (int j = 1; j < lg; j++) {
            int len = m - (1 << j) + 1; if (len <= 0) { sp[j] = new int[0]; continue; }
            sp[j] = new int[len];
            for (int i = 0; i < len; i++) sp[j][i] = Math.min(sp[j - 1][i], sp[j - 1][i + (1 << (j - 1))]);
        }
        a1 = new long[nInt]; a2 = new long[nInt]; b1 = new long[nInt]; b2 = new long[nInt]; p1 = new long[nInt]; p2 = new long[nInt];
        zA = new int[nInt]; zB = new int[nInt]; zP = new int[nInt];
        minA = new int[nInt]; minB = new int[nInt]; minP = new int[nInt]; valid = new boolean[nInt];
        int[] mark = null;
        for (int u = 0; u < nInt; u++) {
            int l = triples[3 * u], mid = triples[3 * u + 1], r = triples[3 * u + 2];
            a1[u] = r1[3 * u]; a2[u] = r2[3 * u]; zA[u] = rc[3 * u];
            b1[u] = r1[3 * u + 1]; b2[u] = r2[3 * u + 1]; zB[u] = rc[3 * u + 1];
            p1[u] = r1[3 * u + 2]; p2[u] = r2[3 * u + 2]; zP[u] = rc[3 * u + 2];
            minA[u] = rmq(sp, l, mid); minB[u] = rmq(sp, mid + 1, r); minP[u] = Math.min(minA[u], minB[u]);
            valid[u] = rc[3 * u + 2] == rc[3 * u] + rc[3 * u + 1];
            if (!valid[u]) {
                invalid++;
                if (mark == null) mark = new int[nTaxa];
                int stamp = u + 1;
                for (int p = mid + 1; p <= r; p++) mark[leaves[p]] = stamp;              // species of B
                IntList aMinusB = new IntList(), bMinusA = new IntList();
                for (int p = l; p <= mid; p++) if (prev[p] < l && mark[leaves[p]] != stamp) aMinusB.add(leaves[p]);
                int stamp2 = -(u + 1);
                for (int p = l; p <= mid; p++) mark[leaves[p]] = stamp2;                 // species of A
                for (int p = mid + 1; p <= r; p++) if (prev[p] < mid + 1 && mark[leaves[p]] != stamp2) bMinusA.add(leaves[p]);
                if (exA == null) { exA = new int[nInt]; exB = new int[nInt]; java.util.Arrays.fill(exA, -1); java.util.Arrays.fill(exB, -1); }
                if (aMinusB.size() > 0) { int[] x = aMinusB.toArray(); java.util.Arrays.sort(x); exA[u] = extra.size(); extra.add(x); }
                if (bMinusA.size() > 0) { int[] x = bMinusA.toArray(); java.util.Arrays.sort(x); exB[u] = extra.size(); extra.add(x); }
            }
        }
    }

    private static int rmq(int[][] sp, int l, int r) {
        int j = 31 - Integer.numberOfLeadingZeros(r - l + 1);
        return Math.min(sp[j][l], sp[j][r - (1 << j) + 1]);
    }
    private static void upd(int[] fc, long[] f1, long[] f2, int p, int dc, long d1, long d2) {
        for (int i = p + 1; i < fc.length; i += i & -i) { fc[i] += dc; f1[i] += d1; f2[i] ^= d2; }
    }
    private static int cnt(int[] f, int p) { int s = 0; for (int i = p + 1; i > 0; i -= i & -i) s += f[i]; return s; }
    private static long s1(long[] f, int p) { long s = 0; for (int i = p + 1; i > 0; i -= i & -i) s += f[i]; return s; }
    private static long x2(long[] f, int p) { long s = 0; for (int i = p + 1; i > 0; i -= i & -i) s ^= f[i]; return s; }
}
