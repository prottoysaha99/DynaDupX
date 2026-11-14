package phylonet.coalescent.fast;

/**
 * Gene-tree side data used by the scorer (all ids are ClusterTable ids):
 *  - valid gene STBs y = (sideA | sideB) with frequency; adjacency side -> y
 *  - side clusters bucketed by minimum species (sorted by size) for "sides inside S" enumeration
 *  - internal-node clusters (valid AND overlapping nodes) with per-tree occurrence counts,
 *    bucketed by minimum species (sorted by size), for pure-subtree counting (MGDL)
 *  - per-species per-tree copy counts; per-tree species counts
 */
final class GeneData {
    int[][] leavesRef, prevsRef; int[] leafAllFlat, prevAllFlat, treeOffFlat;
    int[][] leaves() { return leavesRef; }
    int[][] prevs() { return prevsRef; }
    int[] leavesAll() { return leafAllFlat; }
    int[] prevsAll() { return prevAllFlat; }
    int[] treeOffsets() { return treeOffFlat; }
    void attachTrees(int[][] leaves, int[][] prevs) {
        leavesRef = leaves; prevsRef = prevs;
        treeOffFlat = new int[k + 1];
        for (int t = 0; t < k; t++) treeOffFlat[t + 1] = treeOffFlat[t] + leaves[t].length;
        leafAllFlat = new int[treeOffFlat[k]]; prevAllFlat = new int[treeOffFlat[k]];
        for (int t = 0; t < k; t++) { System.arraycopy(leaves[t], 0, leafAllFlat, treeOffFlat[t], leaves[t].length); System.arraycopy(prevs[t], 0, prevAllFlat, treeOffFlat[t], prevs[t].length); }
    }
    final int n, k;
    /** packed (size, signature) copies of sideBucket / nodeBucket entries, read sequentially by the scorer */
    int[][] sideSize, nodeSize; long[][] sideSig, nodeSig;
    /** Exact species bitsets (n <= BITSET_MAX_TAXA): W words per side / node cluster, packed per bucket and by id. */
    static final int BITSET_MAX_TAXA = Integer.getInteger("dynadupx.bitsetMaxTaxa", 2048);
    int W;                          // words per bitset (0 = bitsets disabled)
    long[][] sideBits, nodeBits;    // per species bucket, aligned with sideBucket / nodeBucket
    int[] sideIdx; long[] sideBitsById;   // cluster id -> row in sideBitsById (-1 if not a side)
    long[][] adjBits;                     // side cluster K -> other-side bitsets of adjY[K], inline (sequential reads)
    void pack(ClusterTable ct) {
        if (n <= BITSET_MAX_TAXA) {
            W = (n + 63) >>> 6;
            sideBits = new long[n][]; nodeBits = new long[n][];
            if (sideBucket == null) {                      // the per-candidate weight structures are not needed
                for (int v = 0; v < n; v++) {
                    int[] b = nodeBucket[v]; nodeBits[v] = new long[b.length * W];
                    for (int i = 0; i < b.length; i++) fill(ct, b[i], nodeBits[v], i * W);
                }
                pack0(ct);
                return;
            }
            sideIdx = new int[ct.n]; java.util.Arrays.fill(sideIdx, -1);
            int nSides = 0;
            for (int v = 0; v < n; v++) for (int id : sideBucket[v]) if (sideIdx[id] < 0) sideIdx[id] = nSides++;
            sideBitsById = new long[nSides * W];
            for (int v = 0; v < n; v++) for (int id : sideBucket[v]) fill(ct, id, sideBitsById, sideIdx[id] * W);
            adjBits = new long[ct.n][];            // per side cluster K: bitsets of the OTHER side of each adjacent STB
            for (int v = 0; v < n; v++) for (int K : sideBucket[v]) {
                if (adjBits[K] != null) continue;
                int[] ys = adjY[K]; long[] ab = new long[ys.length * W];
                for (int e = 0; e < ys.length; e++) { int other = yA[ys[e]] == K ? yB[ys[e]] : yA[ys[e]]; System.arraycopy(sideBitsById, sideIdx[other] * W, ab, e * W, W); }
                adjBits[K] = ab;
            }
            for (int v = 0; v < n; v++) {
                int[] a = sideBucket[v]; sideBits[v] = new long[a.length * W];
                for (int i = 0; i < a.length; i++) System.arraycopy(sideBitsById, sideIdx[a[i]] * W, sideBits[v], i * W, W);
                int[] b = nodeBucket[v]; nodeBits[v] = new long[b.length * W];
                for (int i = 0; i < b.length; i++) fill(ct, b[i], nodeBits[v], i * W);
            }
        }
        pack0(ct);
    }
    private void fill(ClusterTable ct, int id, long[] out, int off) { for (int x : ct.taxa(id)) out[off + (x >>> 6)] |= 1L << (x & 63); }
    private void pack0(ClusterTable ct) {
        sideSize = new int[n][]; nodeSize = new int[n][]; sideSig = new long[n][]; nodeSig = new long[n][];
        for (int v = 0; v < n; v++) {
            final boolean needSig = W == 0;               // with bitsets the scans use exact bits, not signatures
            if (sideBucket != null) {
                int[] a = sideBucket[v]; sideSize[v] = new int[a.length]; sideSig[v] = needSig ? new long[4 * a.length] : null;
                for (int i = 0; i < a.length; i++) { sideSize[v][i] = ct.size[a[i]]; if (needSig) ct.sig256(a[i], sideSig[v], 4 * i); }
            }
            int[] b = nodeBucket[v]; nodeSize[v] = new int[b.length]; nodeSig[v] = needSig ? new long[4 * b.length] : null;
            for (int i = 0; i < b.length; i++) { nodeSize[v][i] = ct.size[b[i]]; if (needSig) ct.sig256(b[i], nodeSig[v], 4 * i); }
        }
    }
    // valid STBs
    final int[] yA, yB; final long[] yF;
    final int[][] adjY;                 // cluster id -> y ids having it as a side
    final int[][] sideBucket;           // species -> side cluster ids with that min species (size asc)
    // internal node clusters
    final int[][] nodeTree, nodeCnt;    // cluster id -> (tree, count) occurrence list (null if not a node cluster)
    final int[][] nodeBucket;           // species -> node cluster ids with that min species (size asc)
    // species occurrences
    final int[][] occTree, occCnt;      // species -> (tree, copies)
    final int[] speciesPerTree, leavesPerTree;
    final boolean allComplete;
    // tree topology for root-path walks (complement-side pure-subtree counting)
    final int[] intOff;                 // global offset of tree t's internal nodes
    final int[] parLeafAll, parIntAll;  // parent (local internal index) of leaf p / internal u, -1 = root
    final int[] leafOff;                // global offset of tree t's leaves
    final int[][] occPosTree, occPos;   // species -> (tree, position) for every copy
    int totalInternal;
    // ---- bit-parallel weights and totals-based loss terms ----
    /** Internal-node clusters of size >= 2, numbered q = 0..NC-1 by increasing size (ties: cluster id). A "code" names a
     *  cluster in the bit-parallel pass: q for a node cluster, NC + x for species x. */
    int nodeCount;                      // NC
    int[] nodeIdx;                      // cluster id -> q, -1 if not a node cluster of size >= 2
    int[] ndId, ndSize;                 // q -> cluster id, size
    int[] ndA, ndB;                     // q -> codes of two strictly smaller clusters whose union is the cluster
    int[] nyStart, nyA, nyB; long[] nyF;// q -> valid STBs whose parent cluster is q (CSR): side codes, frequency
    long[] ndOcc;                       // q -> number of internal nodes (all trees) with this species set
    long[] spLeafNet;                   // species x -> leaf copies of x - internal nodes whose species set is {x}
    long[] spCopies, spSingle; long totalCopies;   // leaf copies of x; internal nodes whose species set is {x}
    int[][] nodeBucketQ;                // aligned with nodeBucket: q of each entry (-1 for size-1 clusters)
    int[] prefixEnd;                    // prefixEnd[z] = number of node clusters (q) of size <= z
    int[][] tLeaves, tTriples;          // gene trees (post-order species array and (l,m,r) triples) for per-tree counts
    int maxInternal;                    // largest number of internal nodes of one gene tree
    int ndIdOf(int q) { return ndId[q]; }

    // ---- node clusters (size >= 2) indexed by their two rarest species ----------------------------------------
    // A node cluster q lies inside a set V only if its two rarest species (rarest = in the fewest node clusters) are
    // both in V. So the clusters inside V are found by visiting, for every species a of V, the partners b of a that
    // are in V, and testing only the clusters filed under (a, b) (size ascending, exact bitset test). The former
    // index (by smallest species id) tested every cluster containing one species of V.
    int[] prStart, prB, prQStart, prQ; long[] prBits; int[] prAStart;
    int[] prRStart, prRS, prRE;         // pair p -> runs [prRS, prRE) of consecutive q (ascending)
    int[] cntAZ;                        // [a * (n + 1) + z]: node clusters with rarest species a and size <= z
    void buildPairIndex(ClusterTable ct) { buildPairIndex(ct, true); }
    /** withBits: also a copy of the clusters' bitsets in pair-index order (exact tests of collectInside read it
     *  sequentially; without it they read the cluster table) */
    void buildPairIndex(ClusterTable ct, boolean withBits) {
        final int nc = nodeCount, Wd = W;
        if (Wd == 0 || nc == 0 || (long) nc * Wd > Integer.MAX_VALUE - 64) return;
        final long[] bits = ct.bits; final int BW = ct.BW;
        final int[] r1, r2;
        if (ndR1 != null) { r1 = ndR1; r2 = ndR2; }
        else {
            final long[] freq = new long[n];
            for (int q = 0; q < nc; q++) { int o = BW * ndId[q]; for (int w = 0; w < BW; w++) { long b = bits[o + w]; while (b != 0) { freq[(w << 6) + Long.numberOfTrailingZeros(b)]++; b &= b - 1; } } }
            Integer[] ord = new Integer[n]; for (int i = 0; i < n; i++) ord[i] = i;
            java.util.Arrays.sort(ord, (x, y) -> freq[x] != freq[y] ? Long.compare(freq[x], freq[y]) : Integer.compare(x, y));
            final int[] rank = new int[n]; for (int r = 0; r < n; r++) rank[ord[r]] = r;
            r1 = new int[nc]; r2 = new int[nc];
            java.util.stream.IntStream.range(0, nc).parallel().forEach(q -> {
                int o = BW * ndId[q], a = -1, b = -1;
                for (int w = 0; w < BW; w++) { long x = bits[o + w]; while (x != 0) { int v = (w << 6) + Long.numberOfTrailingZeros(x); x &= x - 1;
                    if (a < 0 || rank[v] < rank[a]) { b = a; a = v; } else if (b < 0 || rank[v] < rank[b]) b = v; } }
                r1[q] = a; r2[q] = b;
            });
        }
        prAStart = new int[n + 1];
        for (int q = 0; q < nc; q++) prAStart[r1[q] + 1]++;
        for (int a = 0; a < n; a++) prAStart[a + 1] += prAStart[a];
        prQ = new int[nc];
        { int[] fill = java.util.Arrays.copyOf(prAStart, n); for (int q = 0; q < nc; q++) prQ[fill[r1[q]]++] = q; }
        final int[] pq = prQ, as = prAStart, sz = ndSize;
        java.util.stream.IntStream.range(0, n).parallel().forEach(a -> {        // within a: by partner, then size, then q
            int lo = as[a], hi = as[a + 1];
            long[] key = new long[hi - lo];
            for (int i = lo; i < hi; i++) { int q = pq[i]; key[i - lo] = ((long) r2[q] << 44) | ((long) sz[q] << 32) | q; }
            java.util.Arrays.sort(key);
            for (int i = lo; i < hi; i++) pq[i] = (int) key[i - lo];
        });
        IntList pB = new IntList(1024), pS = new IntList(1024);
        prStart = new int[n + 1];
        for (int a = 0; a < n; a++) {
            prStart[a] = pB.n;
            for (int i = prAStart[a], hi = prAStart[a + 1]; i < hi; i++) { int b = r2[prQ[i]]; if (i == prAStart[a] || b != r2[prQ[i - 1]]) { pB.add(b); pS.add(i); } }
        }
        prStart[n] = pB.n; pS.add(nc);
        // clusters filed under rarest species a with size <= z (sparse-vs-dense estimate of a shared pass)
        cntAZ = new int[n * (n + 1)];
        for (int q = 0; q < nc; q++) cntAZ[r1[q] * (n + 1) + ndSize[q]]++;
        for (int a = 0; a < n; a++) for (int z = 1, o = a * (n + 1); z <= n; z++) cntAZ[o + z] += cntAZ[o + z - 1];
        prB = pB.toArray(); prQStart = pS.toArray();
        // runs of consecutive q per species pair (with the locality order a pair's clusters of one size are consecutive)
        {
            final int P = prB.length;
            IntList rs = new IntList(1024), re = new IntList(1024);
            prRStart = new int[P + 1];
            for (int p = 0; p < P; p++) {
                prRStart[p] = rs.n;
                for (int k = prQStart[p], ke = prQStart[p + 1]; k < ke; k++) {
                    final int q = prQ[k];
                    if (rs.n > prRStart[p] && re.a[re.n - 1] == q) re.a[re.n - 1]++;
                    else { rs.add(q); re.add(q + 1); }
                }
            }
            prRStart[P] = rs.n;
            prRS = rs.toArray(); prRE = re.toArray();
        }
        if (withBits) {
            prBits = new long[nc * Wd];
            final long[] pb = prBits;
            java.util.stream.IntStream.range(0, nc).parallel().forEach(k -> System.arraycopy(bits, BW * ndId[pq[k]], pb, k * Wd, Wd));
        }
        nodeBits = null;                                   // the pair index replaces the per-min-species copies
        ndR1 = null; ndR2 = null;
        // packed copies for the sparse passes (one cache line per visited cluster, its valid STBs contiguous)
        boolean fits = true; for (long f : nyF) if (f > Integer.MAX_VALUE) { fits = false; break; }
        if (fits && (long) 3 * nyA.length < Integer.MAX_VALUE - 64 && (long) 4 * nc < Integer.MAX_VALUE - 64) {
            // record of q: (part, part, start of its other STBs, frequency of the STB (part | part) or 0); the STBs of q end
            // where those of q + 1 start (a sentinel record follows the last cluster)
            ndRec = new int[4 * (nc + 1)];
            int[] pk = new int[3 * nyA.length]; int pos = 0;
            for (int q = 0; q < nc; q++) {
                final int a = ndA[q], b = ndB[q];
                ndRec[4 * q] = a; ndRec[4 * q + 1] = b; ndRec[4 * q + 2] = pos;
                int df = 0;
                for (int e = nyStart[q]; e < nyStart[q + 1]; e++) {
                    if (df == 0 && ((nyA[e] == a && nyB[e] == b) || (nyA[e] == b && nyB[e] == a))) { df = (int) nyF[e]; continue; }
                    pk[pos++] = nyA[e]; pk[pos++] = nyB[e]; pk[pos++] = (int) nyF[e];
                }
                ndRec[4 * q + 3] = df;
            }
            ndRec[4 * nc + 2] = pos;
            nyPack = java.util.Arrays.copyOf(pk, pos);
        }
    }
    int[] ndRec, nyPack;                // q -> (part code, part code, start of the other STBs, frequency of STB (part|part)); STB -> (side, side, frequency)
    int code(int id) { return id < n ? nodeCount + id : nodeIdx[id]; }

    /** Within each size, node clusters are numbered by (rarest species, second rarest species) rather than by cluster
     *  id, so the clusters one sparse pass visits (and their parts) lie in few contiguous runs. Any order within a size
     *  keeps the parts of a cluster before it; the weights are sums, so the results do not change. */
    static final boolean LOCALITY_ORDER = !"false".equals(System.getProperty("dynadupx.localityOrder"));
    int[] ndR1, ndR2;                   // q -> rarest / second rarest species (rank = fewest node clusters)
    private void localityOrder(ClusterTable ct, int[] zStart) {
        final int nc = nodeCount, BW = ct.BW; final long[] bits = ct.bits;
        final long[] freq = new long[n];
        for (int q = 0; q < nc; q++) { int o = BW * ndId[q]; for (int w = 0; w < BW; w++) { long b = bits[o + w]; while (b != 0) { freq[(w << 6) + Long.numberOfTrailingZeros(b)]++; b &= b - 1; } } }
        Integer[] ord = new Integer[n]; for (int i = 0; i < n; i++) ord[i] = i;
        java.util.Arrays.sort(ord, (x, y) -> freq[x] != freq[y] ? Long.compare(freq[x], freq[y]) : Integer.compare(x, y));
        final int[] rank = new int[n]; for (int r = 0; r < n; r++) rank[ord[r]] = r;
        final int[] id2 = ndId;
        final long[] key = new long[nc];
        java.util.stream.IntStream.range(0, nc).parallel().forEach(q -> {
            int o = BW * id2[q], a = -1, b = -1;
            for (int w = 0; w < BW; w++) { long x = bits[o + w]; while (x != 0) { int v = (w << 6) + Long.numberOfTrailingZeros(x); x &= x - 1;
                if (a < 0 || rank[v] < rank[a]) { b = a; a = v; } else if (b < 0 || rank[v] < rank[b]) b = v; } }
            key[q] = ((long) a << 42) | ((long) b << 31) | id2[q];
        });
        java.util.stream.IntStream.rangeClosed(2, n).parallel().forEach(z -> { if (zStart[z + 1] - zStart[z] > 1) java.util.Arrays.sort(key, zStart[z], zStart[z + 1]); });
        ndR1 = new int[nc]; ndR2 = new int[nc];
        for (int q = 0; q < nc; q++) { long k = key[q]; ndId[q] = (int) (k & 0x7fffffffL); ndR1[q] = (int) (k >>> 42); ndR2[q] = (int) ((k >>> 31) & 0x7ff); }
    }

    void setDecompositions(ClusterTable ct, int[] decA, int[] decB, int[] yPar) {
        final int N = ct.n;
        int[] cnt = new int[n + 2];
        for (int id = n; id < N; id++) if (nodeTree[id] != null && ct.size[id] >= 2) cnt[ct.size[id] + 1]++;
        for (int z = 0; z <= n; z++) cnt[z + 1] += cnt[z];
        final int nc = cnt[n + 1];
        nodeCount = nc;
        nodeIdx = new int[N]; java.util.Arrays.fill(nodeIdx, -1);
        ndId = new int[nc]; ndSize = new int[nc];
        final int[] zStart = cnt.clone();
        for (int id = n; id < N; id++) if (nodeTree[id] != null && ct.size[id] >= 2) { int q = cnt[ct.size[id]]++; ndId[q] = id; ndSize[q] = ct.size[id]; }
        if (LOCALITY_ORDER && W > 0 && ct.bits != null && nc > 0) localityOrder(ct, zStart);
        for (int q = 0; q < nc; q++) nodeIdx[ndId[q]] = q;
        ndA = new int[nc]; ndB = new int[nc]; ndOcc = new long[nc];
        for (int q = 0; q < nc; q++) {
            int id = ndId[q];
            if (id >= decA.length || decA[id] < 0) throw new IllegalStateException("node cluster " + id + " has no decomposition");
            ndA[q] = code(decA[id]); ndB[q] = code(decB[id]);
            if (ndA[q] < 0 || ndB[q] < 0) throw new IllegalStateException("decomposition of node cluster " + id + " is not a node cluster");
            long o = 0; for (int c : nodeCnt[id]) o += c; ndOcc[q] = o;
        }
        final int U = yA.length;
        nyStart = new int[nc + 1];
        for (int y = 0; y < U; y++) nyStart[nodeIdx[yPar[y]] + 1]++;
        for (int q = 0; q < nc; q++) nyStart[q + 1] += nyStart[q];
        nyA = new int[U]; nyB = new int[U]; nyF = new long[U];
        int[] fill = java.util.Arrays.copyOf(nyStart, nc);
        for (int y = 0; y < U; y++) { int e = fill[nodeIdx[yPar[y]]]++; nyA[e] = code(yA[y]); nyB[e] = code(yB[y]); nyF[e] = yF[y]; }
        spLeafNet = new long[n]; spCopies = new long[n]; spSingle = new long[n];
        for (int x = 0; x < n; x++) {
            long c = 0; for (int v : occCnt[x]) c += v;
            long si = 0; if (nodeTree[x] != null) for (int v : nodeCnt[x]) si += v;
            spCopies[x] = c; spSingle[x] = si; spLeafNet[x] = c - si; totalCopies += c;
        }
        prefixEnd = new int[n + 2];                  // prefixEnd[z] = number of node clusters of size <= z
        for (int q = 0; q < nc; q++) prefixEnd[ndSize[q]]++;
        for (int z = 1; z <= n + 1; z++) prefixEnd[z] += prefixEnd[z - 1];
        nodeBucketQ = new int[n][];
        for (int v = 0; v < n; v++) { int[] b = nodeBucket[v], q = new int[b.length]; for (int i = 0; i < b.length; i++) q[i] = nodeIdx[b[i]]; nodeBucketQ[v] = q; }
    }

    GeneData(int n, int k, int[] yA, int[] yB, long[] yF, int[][] adjY, int[][] sideBucket, int[][] nodeTree, int[][] nodeCnt,
             int[][] nodeBucket, int[][] occTree, int[][] occCnt, int[] speciesPerTree, int[] leavesPerTree) {
        this.n = n; this.k = k; this.yA = yA; this.yB = yB; this.yF = yF; this.adjY = adjY; this.sideBucket = sideBucket;
        this.nodeTree = nodeTree; this.nodeCnt = nodeCnt; this.nodeBucket = nodeBucket; this.occTree = occTree; this.occCnt = occCnt;
        this.speciesPerTree = speciesPerTree; this.leavesPerTree = leavesPerTree;
        boolean all = true;
        for (int s : speciesPerTree) if (s != n) { all = false; break; }
        allComplete = all;
        intOff = new int[k + 1]; leafOff = new int[k + 1];
        parLeafAll = null; parIntAll = null; occPosTree = null; occPos = null;
    }

    GeneData(GeneData g, int[][] triples, int[][] leaves) {   // adds topology arrays
        this.n = g.n; this.k = g.k; this.yA = g.yA; this.yB = g.yB; this.yF = g.yF; this.adjY = g.adjY; this.sideBucket = g.sideBucket;
        this.nodeTree = g.nodeTree; this.nodeCnt = g.nodeCnt; this.nodeBucket = g.nodeBucket; this.occTree = g.occTree; this.occCnt = g.occCnt;
        this.speciesPerTree = g.speciesPerTree; this.leavesPerTree = g.leavesPerTree; this.allComplete = g.allComplete;
        this.sideSize = g.sideSize; this.nodeSize = g.nodeSize; this.sideSig = g.sideSig; this.nodeSig = g.nodeSig;
        this.W = g.W; this.sideBits = g.sideBits; this.nodeBits = g.nodeBits; this.sideIdx = g.sideIdx; this.sideBitsById = g.sideBitsById; this.adjBits = g.adjBits;
        this.nodeCount = g.nodeCount; this.nodeIdx = g.nodeIdx; this.ndId = g.ndId; this.ndSize = g.ndSize; this.ndA = g.ndA; this.ndB = g.ndB;
        this.nyStart = g.nyStart; this.nyA = g.nyA; this.nyB = g.nyB; this.nyF = g.nyF; this.ndOcc = g.ndOcc; this.spLeafNet = g.spLeafNet; this.spCopies = g.spCopies; this.spSingle = g.spSingle; this.totalCopies = g.totalCopies; this.nodeBucketQ = g.nodeBucketQ; this.prefixEnd = g.prefixEnd;
        this.ndRec = g.ndRec; this.nyPack = g.nyPack; this.ndR1 = g.ndR1; this.ndR2 = g.ndR2;
        this.prRStart = g.prRStart; this.prRS = g.prRS; this.prRE = g.prRE; this.cntAZ = g.cntAZ;
        this.prStart = g.prStart; this.prB = g.prB; this.prQStart = g.prQStart; this.prQ = g.prQ; this.prBits = g.prBits; this.prAStart = g.prAStart;
        this.tLeaves = leaves; this.tTriples = triples;
        for (int[] tr : triples) maxInternal = Math.max(maxInternal, tr.length / 3);
        intOff = new int[k + 1]; leafOff = new int[k + 1];
        for (int t = 0; t < k; t++) { intOff[t + 1] = intOff[t] + triples[t].length / 3; leafOff[t + 1] = leafOff[t] + leaves[t].length; }
        totalInternal = intOff[k];
        parLeafAll = new int[leafOff[k]]; parIntAll = new int[intOff[k]];
        java.util.Arrays.fill(parLeafAll, -1); java.util.Arrays.fill(parIntAll, -1);
        java.util.stream.IntStream.range(0, k).parallel().forEach(t -> {
            int[] tr = triples[t]; int ni = tr.length / 3, lo = leafOff[t], io = intOff[t];
            for (int u = 0; u < ni; u++) {
                int l = tr[3 * u], m = tr[3 * u + 1], r = tr[3 * u + 2];
                // post-order: left child = internal node u-(r-m) if m > l, else leaf l; right child = u-1 if r > m+1, else leaf r
                if (m > l) { int c = u - (r - m); checkChild(tr, c, l, m); parIntAll[io + c] = u; } else parLeafAll[lo + l] = u;
                if (r > m + 1) { int c = u - 1; checkChild(tr, c, m + 1, r); parIntAll[io + c] = u; } else parLeafAll[lo + r] = u;
            }
        });
        IntList[] pt = new IntList[n], pp = new IntList[n];
        for (int v = 0; v < n; v++) { pt[v] = new IntList(2); pp[v] = new IntList(2); }
        for (int t = 0; t < k; t++) { int[] L = leaves[t]; for (int p = 0; p < L.length; p++) { pt[L[p]].add(t); pp[L[p]].add(p); } }
        occPosTree = new int[n][]; occPos = new int[n][];
        for (int v = 0; v < n; v++) { occPosTree[v] = pt[v].toArray(); occPos[v] = pp[v].toArray(); }
    }

    private static void checkChild(int[] tr, int c, int l, int r) {
        if (c < 0 || tr[3 * c] != l || tr[3 * c + 2] != r) throw new IllegalStateException("gene tree nodes are not in post-order");
    }


}
