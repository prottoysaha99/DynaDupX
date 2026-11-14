package phylonet.coalescent.fast;

import java.util.Arrays;

/**
 * All clusters of the search space, identified by their set signature (ClusterKey) instead of an
 * O(n)-bit STITreeCluster. A cluster is stored compactly as one of:
 *   - a range of a gene tree's post-order leaf array (tree, l, r)   -> O(1) words
 *   - the complement (w.r.t. all species) of another cluster        -> O(1) words
 *   - an explicit sorted species array (singletons, A-B/B-A extras, DP fallback clusters)
 * Clusters are bucketed by their minimum species id: in any resolution (A | C-A) of C exactly one
 * side contains min(C), and that side has min == min(C). So Psi(C) is enumerated by scanning only
 * bucket[min(C)] and testing whether key(C) - key(A) is a known cluster (O(1) hash lookup).
 *
 * Each bucket is split by cluster SIZE (bucket[min][size] = ids in insertion order). Iterating the
 * sizes in increasing order gives exactly the order of the former size-sorted bucket (a stable sort of
 * the insertion order), but inserting is O(1) instead of an insertion-sort shift that cost O(bucket)
 * per cluster (O(N^2) overall: with extended complements bucket[0] holds about half of all clusters).
 */
final class ClusterTable {
    final int nTaxa; final int[][] leaves, prev;
    int n = 0;
    long[] h1 = new long[1024], h2 = new long[1024];
    int[] size = new int[1024], minT = new int[1024], repTree = new int[1024], repL = new int[1024], repR = new int[1024], compOf = new int[1024];
    /** Species arrays of explicit clusters; only when there are no exact bitsets (then the bitset is the species set). */
    int[][] expl;
    /** 256-bit species signature (bit = h1(species) & 255) and signature of the COMPLEMENT, only without bitsets
     *  (nTaxa > GeneData.BITSET_MAX_TAXA). A subset A of C satisfies sig(A) in sig(C) and csig(C) in csig(A). */
    long[] sig, csig;
    /** 64-bit folds (OR of the four words) of sig and csig, for every cluster: the bucket prefilter. */
    long[] fold = new long[1024], cfold = new long[1024];
    /** Exact species bitsets (BW words per cluster) when nTaxa <= GeneData.BITSET_MAX_TAXA, else null. */
    long[] bits; final int BW;
    private int[] smark; private int sstamp = 0;
    final long[] H1;
    KeyIndex index = new KeyIndex(this, 1 << 16);
    /** bucket[min][size]: cluster ids with that minimum species and size, in insertion order (lazy). Each entry also
     *  carries a packed copy of the cluster's (h1, h2, sig, csig) so that resolution scans read memory sequentially. */
    SizeRow[] bucket;                    // non-complement clusters
    SizeRow[] cbucket;                   // complement clusters (kept apart: they dominate bucket[0] in the extended space)
    /** The buckets of one species, by size, stored sparsely (sorted sizes + buckets), so that the bucket index grows
     *  with the number of clusters and not with nTaxa^2 (a dense nTaxa + 1 row per species would need 40 GB for
     *  100,000 species). */
    static final class SizeRow {
        int n; int[] z = new int[2]; Bucket[] b = new Bucket[2];
        Bucket get(int size) { int i = Arrays.binarySearch(z, 0, n, size); return i >= 0 ? b[i] : null; }
        /** number of buckets with size < lim (they are b[0 .. end)) */
        int below(int lim) { int i = Arrays.binarySearch(z, 0, n, lim); return i >= 0 ? i : -i - 1; }
        Bucket put(int size, Bucket nb) {
            int i = -Arrays.binarySearch(z, 0, n, size) - 1;
            if (n == z.length) { int c = n + (n >> 1) + 2; z = Arrays.copyOf(z, c); b = Arrays.copyOf(b, c); }
            System.arraycopy(z, i, z, i + 1, n - i); System.arraycopy(b, i, b, i + 1, n - i);
            z[i] = size; b[i] = nb; n++;
            return nb;
        }
    }
    int[] compCount;                     // complement clusters per minimum species
    /** sizeMask[z]: bit v set iff bucket[v][z] (non-complement) is non-empty -> fallback skips empty (v, z) lists */
    long[][] sizeMask;
    /** Bucket entries: cluster id + a 128-bit prefilter (64-bit folds of the 256-bit signature of the cluster and of
     *  its complement). A subset A of C must satisfy fold(A) in fold(C) and cfold(C) in cfold(A). The scan reads only the
     *  prefilter array (16 bytes per entry, sequential); survivors get the exact test from the cluster table. */
    final boolean exact = true; final int stride = 2;
    final class Bucket {
        int n; int[] ids = new int[2]; long[] d = new long[4];
        boolean mixed;                   // entries of different sizes (pair buckets)
        /** With exact bitsets (non-complement buckets): the second prefilter word holds four of the cluster's rarest
         *  species (16 bits each) instead of the complement fold: a subset of C must have them in C. */
        boolean rare;
        void add(int id) { add(id, cfold[id]); }
        void add(int id, long second) {
            if (n == ids.length) { int cap = n + (n >> 1) + 2; ids = Arrays.copyOf(ids, cap); d = Arrays.copyOf(d, 2 * cap); }
            int o = 2 * n; ids[n] = id;
            d[o] = fold[id]; d[o + 1] = second;
            n++;
        }
        int size() { return n; }
        int get(int i) { return ids[i]; }
    }
    /** 256-bit signature of cluster id into out[off..off+4) (stored, or computed from the species). */
    void sig256(int id, long[] out, int off) {
        if (sig != null) { System.arraycopy(sig, 4 * id, out, off, 4); return; }
        out[off] = out[off + 1] = out[off + 2] = out[off + 3] = 0;
        for (int v : taxa(id)) { long h = H1[v]; out[off + (int) ((h >>> 6) & 3)] |= 1L << (h & 63); }
    }
    boolean isExplicit(int id) { return repTree[id] < 0 && compOf[id] < 0; }   // (before releaseSearch)
    /** Exact subset test of cluster id against species set C given as bitset inC (bits != null) or as marks. */
    boolean inside(int id, long[] inC, int[] mark, int stamp) {
        if (bits != null) { for (int j = 0, o = BW * id; j < BW; j++) if ((bits[o + j] & ~inC[j]) != 0) return false; return true; }
        return allMarked(id, mark, stamp);
    }
    /** Bucketing species of each cluster: its species of lowest RANK, where species are ranked by increasing number of
     *  clusters containing them (bucketing by the smallest species id would overload the buckets of the first
     *  species). Every resolution (A | C-A) of C has exactly one side containing bkey(C), and that side has the
     *  same bkey, so Psi(C) is still found in one bucket. */
    int[] bkey = new int[1024];
    int[] rank;                          // species -> rank; null until enableBuckets()
    /** Non-complement clusters are also filed under (rarest, second rarest species) (the rarest twice for a
     *  single species). A cluster A inside C that contains bkey(C) has rarest species bkey(C) and its second rarest in C,
     *  so the resolution scan of C reads only pbucket[bkey(C)][b] for the species b of C instead of every cluster
     *  with rarest species bkey(C). Only with exact bitsets. */
    Bucket[][] pbucket;
    static final boolean PAIR_BUCKETS = !"false".equals(System.getProperty("dynadupx.pairBuckets"));
    ClusterKey all;

    ClusterTable(int nTaxa, int[][] leaves, int[][] prev, long[] H1) {
        this.H1 = H1;
        this.nTaxa = nTaxa; this.leaves = leaves; this.prev = prev;
        BW = nTaxa <= GeneData.BITSET_MAX_TAXA ? (nTaxa + 63) >>> 6 : 0;
        if (BW > 0) bits = new long[BW * 1024];
        else { sig = new long[4 * 1024]; csig = new long[4 * 1024]; expl = new int[1024][]; }
        bucket = new SizeRow[nTaxa]; cbucket = new SizeRow[nTaxa]; compCount = new int[nTaxa];
        if (BW > 0 && PAIR_BUCKETS) pbucket = new Bucket[nTaxa][];
        sizeMask = new long[nTaxa + 1][];               // rows allocated when a size first occurs
        // per fingerprint bit, how many species set it (complement fingerprints from counts instead of a loop over
        // all species: O(cluster size) instead of O(number of species) per new cluster)
        for (int v = 0; v < nTaxa; v++) { long h = H1[v]; int b = (int) (((h >>> 6) & 3) << 6 | (h & 63)); allCnt[b]++; allSig[b >>> 6] |= 1L << b; }
    }
    private final int[] allCnt = new int[256], inCnt = new int[256];
    private final long[] allSig = new long[4];
    private static int sigPos(long h) { return (int) (((h >>> 6) & 3) << 6 | (h & 63)); }
    /** fingerprint of the species NOT in a cluster whose own fingerprint is own, after inCnt was filled with the
     *  cluster's (distinct) species; resets inCnt */
    private long[] complementFromCounts(long[] own) {
        for (int w = 0; w < 4; w++) {
            long c = allSig[w], x = own[w];
            while (x != 0) { final int bit = Long.numberOfTrailingZeros(x); x &= x - 1; final int b = (w << 6) | bit; if (inCnt[b] == allCnt[b]) c &= ~(1L << bit); inCnt[b] = 0; }
            cc4[w] = c;
        }
        return cc4;
    }

    ClusterKey key(int id) { return new ClusterKey(h1[id], h2[id], size[id]); }
    int find(ClusterKey k) { return index.get(k.h1(), k.h2(), k.size()); }
    int find(long a, long b, int z) { return index.get(a, b, z); }

    /** Species x belongs to cluster id (bitsets if available, else a scan of the cluster's species). */
    boolean bitsInOrContains(int id, int x) {
        if (bits != null) return (bits[BW * id + (x >>> 6)] >>> (x & 63) & 1L) != 0;
        for (int v : taxa(id)) if (v == x) return true;
        return false;
    }
    /** Species bitset of cluster id is a subset of the given bitset (requires bits != null). */
    boolean bitsIn(int id, long[] set) {
        for (int j = 0, o = BW * id; j < BW; j++) if ((bits[o + j] & ~set[j]) != 0) return false;
        return true;
    }
    /** Every species of cluster id carries mark == stamp (subset test against a marked set). */
    boolean allMarked(int id, int[] mark, int stamp) {
        if (expl != null && expl[id] != null) { for (int v : expl[id]) if (mark[v] != stamp) return false; return true; }
        if (bits != null && (repTree == null || repTree[id] < 0)) {
            for (int w = 0, o = BW * id; w < BW; w++) { long b = bits[o + w]; while (b != 0) { if (mark[(w << 6) + Long.numberOfTrailingZeros(b)] != stamp) return false; b &= b - 1; } }
            return true;
        }
        if (repTree[id] >= 0) {
            int t = repTree[id], l = repL[id], r = repR[id]; int[] L = leaves[t], P = prev[t];
            for (int p = l; p <= r; p++) if (P[p] < l && mark[L[p]] != stamp) return false;
            return true;
        }
        for (int v : taxa(id)) if (mark[v] != stamp) return false;
        return true;
    }
    private int add(ClusterKey k, int min) {
        int got = index.get(k.h1(), k.h2(), k.size());
        if (got >= 0) return got;
        if (n == size.length) grow();
        int id = n++;
        h1[id] = k.h1(); h2[id] = k.h2(); size[id] = k.size(); minT[id] = min;
        repTree[id] = -1; compOf[id] = -1; if (expl != null) expl[id] = null; fold[id] = 0; cfold[id] = 0;
        if (BW > 0) Arrays.fill(bits, BW * id, BW * id + BW, 0L);
        index.put(k.h1(), k.h2(), k.size(), id);
        return id;
    }
    int addRange(ClusterKey k, int min, int t, int l, int r) {
        int before = n, id = add(k, min);
        if (id < before && repTree[id] < 0) {      // same species set already stored explicitly (e.g. an A-B
            repTree[id] = t; repL[id] = l; repR[id] = r;   // cluster of an overlapping node): record its range too
        }
        if (id >= before) {
            repTree[id] = t; repL[id] = l; repR[id] = r;
            int[] L = leaves[t];
            int st = nextStamp();
            final long[] sg = sc4; sg[0] = sg[1] = sg[2] = sg[3] = 0;
            for (int p = l; p <= r; p++) { final int v = L[p]; if (smark[v] == st) continue; smark[v] = st; long h = H1[v]; sg[(int) ((h >>> 6) & 3)] |= 1L << (h & 63); inCnt[sigPos(h)]++; }
            if (BW > 0) for (int p = l; p <= r; p++) bits[BW * id + (L[p] >>> 6)] |= 1L << (L[p] & 63);
            setSigs(id, sg, complementFromCounts(sg));
            toBucket(id);
        }
        return id;
    }
    int addExplicit(ClusterKey k, int min, int[] taxa) {
        int before = n, id = add(k, min);
        if (id >= before) {
            if (expl != null) expl[id] = taxa;
            int st = nextStamp();
            final long[] sg = sc4; sg[0] = sg[1] = sg[2] = sg[3] = 0;
            for (int v : taxa) { if (smark[v] == st) continue; smark[v] = st; long h = H1[v]; sg[(int) ((h >>> 6) & 3)] |= 1L << (h & 63); inCnt[sigPos(h)]++; }
            if (BW > 0) for (int v : taxa) bits[BW * id + (v >>> 6)] |= 1L << (v & 63);
            setSigs(id, sg, complementFromCounts(sg));
            toBucket(id);
        }
        return id;
    }
    int addExplicit(int[] sortedTaxa) { return addExplicit(ClusterKey.of(sortedTaxa), sortedTaxa[0], sortedTaxa); }
    int addComplement(ClusterKey k, int min, int of) {
        int before = n, id = add(k, min);
        if (id >= before) {
            compOf[id] = of;
            int st = nextStamp();
            long[] ofSig = new long[4]; sig256(of, ofSig, 0);    // complement of the complement = of
            for (int v : taxa(of)) { if (smark[v] == st) continue; smark[v] = st; inCnt[sigPos(H1[v])]++; }
            long[] own = complementFromCounts(ofSig).clone();   // species NOT in `of` = species of the complement
            if (BW > 0) for (int v = 0; v < nTaxa; v++) if (smark[v] != st) bits[BW * id + (v >>> 6)] |= 1L << (v & 63);
            setSigs(id, own, ofSig);
            toBucket(id);
        }
        return id;
    }

    /** Species of a cluster (distinct). */
    int[] taxa(int id) {
        if (expl != null) { if (expl[id] != null) return expl[id]; }
        else if (repTree == null || repTree[id] < 0) {      // explicit or complement (or ranges dropped): decode the bitset
            int[] out = new int[size[id]]; int k = 0;
            for (int w = 0, o = BW * id; w < BW; w++) { long b = bits[o + w]; while (b != 0) { out[k++] = (w << 6) + Long.numberOfTrailingZeros(b); b &= b - 1; } }
            return out;
        }
        if (repTree[id] >= 0) {
            int t = repTree[id], l = repL[id], r = repR[id];
            int[] L = leaves[t], P = prev[t];
            int[] out = new int[size[id]]; int k = 0;
            for (int p = l; p <= r; p++) if (P[p] < l) out[k++] = L[p];
            return out;
        }
        int[] inner = taxa(compOf[id]);
        boolean[] in = new boolean[nTaxa];
        for (int v : inner) in[v] = true;
        int[] out = new int[size[id]]; int k = 0;
        for (int v = 0; v < nTaxa; v++) if (!in[v]) out[k++] = v;
        return out;
    }

    /** Smallest species NOT in cluster id (for complements); O(|cluster|). */
    int minOfComplement(int id, int[] scratch, int stamp) {
        if (minT[id] > 0) return 0;
        for (int v : taxa(id)) scratch[v] = stamp;
        int v = 0; while (scratch[v] == stamp) v++;
        return v;
    }

    /** Fixes the species ranks and buckets all clusters so far (in id order); later clusters are bucketed when added. */
    void enableBuckets() {
        long[] freq = new long[nTaxa];
        for (int id = 0; id < n; id++) { if (compOf[id] >= 0) continue; for (int v : taxa(id)) freq[v]++; }
        Integer[] ord = new Integer[nTaxa];
        for (int i = 0; i < nTaxa; i++) ord[i] = i;
        if (!"false".equals(System.getProperty("dynadupx.rankKeys")))
            Arrays.sort(ord, (a, b) -> freq[a] != freq[b] ? Long.compare(freq[a], freq[b]) : Integer.compare(a, b));
        rank = new int[nTaxa];
        for (int r = 0; r < nTaxa; r++) rank[ord[r]] = r;
        for (int id = 0; id < n; id++) toBucket(id);
    }
    private int rankMin(int id) {
        int best = -1;
        if (bits != null) {
            for (int w = 0, o = BW * id; w < BW; w++) { long b = bits[o + w]; while (b != 0) { int v = (w << 6) + Long.numberOfTrailingZeros(b); b &= b - 1; if (best < 0 || rank[v] < rank[best]) best = v; } }
        } else for (int v : taxa(id)) if (best < 0 || rank[v] < rank[best]) best = v;
        return best;
    }

    private void toBucket(int id) {
        if (rank == null) return;                     // bucketed by enableBuckets()
        bkey[id] = rankMin(id);
        int min = bkey[id], z = size[id];
        boolean comp = compOf[id] >= 0;
        SizeRow[] bk = comp ? cbucket : bucket;
        if (comp) compCount[min]++;
        else { long[] sm = sizeMask[z]; if (sm == null) sm = sizeMask[z] = new long[(nTaxa + 63) >>> 6]; sm[min >>> 6] |= 1L << (min & 63); }
        SizeRow row = bk[min];
        if (row == null) row = bk[min] = new SizeRow();
        Bucket b = row.get(z);
        final boolean rare = !comp && bits != null && RARE_FILTER;
        if (b == null) { b = row.put(z, new Bucket()); b.rare = rare; }
        if (!rare) { b.add(id); return; }
        // the six rarest species r[0] = min, r[1], ... (padded with min)
        final int[] r = rare6; Arrays.fill(r, min); int m = 1;
        for (int w = 0, o = BW * id; w < BW; w++) { long x = bits[o + w]; while (x != 0) { int v = (w << 6) + Long.numberOfTrailingZeros(x); x &= x - 1;
            if (v == min) continue;
            int rv = rank[v];
            if (m < 6) { int i = m++; while (i > 1 && rank[r[i - 1]] > rv) { r[i] = r[i - 1]; i--; } r[i] = v; }
            else if (rv < rank[r[5]]) { int i = 5; while (i > 1 && rank[r[i - 1]] > rv) { r[i] = r[i - 1]; i--; } r[i] = v; } } }
        b.add(id, pack4(r[1], r[2], r[3], r[4]));
        if (pbucket != null) {
            Bucket[] pr = pbucket[min];
            if (pr == null) pr = pbucket[min] = new Bucket[nTaxa];
            Bucket pb = pr[r[1]];                     // second rarest species (min again for a single species)
            if (pb == null) { pb = pr[r[1]] = new Bucket(); pb.mixed = true; pb.rare = true; }
            pb.add(id, pack4(r[2], r[3], r[4], r[5]));
        }
    }

    private final int[] rare6 = new int[6];
    static final boolean RARE_FILTER = !"false".equals(System.getProperty("dynadupx.rareFilter"));
    private static long pack4(int a, int b, int c, int d) { return (long) a | ((long) b << 16) | ((long) c << 32) | ((long) d << 48); }
    /** All four species packed in word pk are in the species bitset inC. */
    static boolean rareIn(long pk, long[] inC) {
        final int x1 = (int) pk & 0xffff, x2 = (int) (pk >>> 16) & 0xffff, x3 = (int) (pk >>> 32) & 0xffff, x4 = (int) (pk >>> 48);
        return ((inC[x1 >>> 6] >>> x1) & (inC[x2 >>> 6] >>> x2) & (inC[x3 >>> 6] >>> x3) & (inC[x4 >>> 6] >>> x4) & 1L) != 0;
    }
    private int nextStamp() { if (smark == null) smark = new int[nTaxa]; return ++sstamp; }
    private final long[] sc4 = new long[4], cc4 = new long[4];
    /** Signature of the species NOT carrying mark st (shared scratch array). */
    private void setSigs(int id, long[] own, long[] comp) {
        fold[id] = own[0] | own[1] | own[2] | own[3]; cfold[id] = comp[0] | comp[1] | comp[2] | comp[3];
        if (sig != null) { System.arraycopy(own, 0, sig, 4 * id, 4); System.arraycopy(comp, 0, csig, 4 * id, 4); }
    }

    /** After the state space: drops the search-only structures (hash index, buckets, hash keys, prefilter folds) and
     *  trims the arrays to size. With exact bitsets the gene-tree ranges are dropped too (the bitset is the species set),
     *  unless keepRanges (the GPU path reads them). No cluster can be added or looked up afterwards. */
    void releaseSearch(boolean keepRanges) {
        index = null; bucket = null; cbucket = null; pbucket = null; sizeMask = null; compIdx = null; compCount = null; bkey = null;
        h1 = null; h2 = null; fold = null; cfold = null;
        int c = n;
        size = Arrays.copyOf(size, c); minT = Arrays.copyOf(minT, c); compOf = Arrays.copyOf(compOf, c);
        if (bits != null && !keepRanges) { repTree = null; repL = null; repR = null; }
        else { repTree = Arrays.copyOf(repTree, c); repL = Arrays.copyOf(repL, c); repR = Arrays.copyOf(repR, c); }
        if (expl != null) expl = Arrays.copyOf(expl, c);
        if (sig != null) { sig = Arrays.copyOf(sig, 4 * c); csig = Arrays.copyOf(csig, 4 * c); }
        if (BW > 0) bits = Arrays.copyOf(bits, BW * c);
    }

    /** Ids with bucketing species min and the given size (null if none). */
    Bucket bucket(int min, int z) { SizeRow row = bucket[min]; return row == null ? null : row.get(z); }
    /** All non-complement buckets of species min, by increasing size (null if none). */
    SizeRow row(int min) { return bucket[min]; }
    /** All complement buckets of species min, by increasing size (null if none). */
    SizeRow crow(int min) { return cbucket[min]; }
    /** Complement clusters with minimum species min and size z (null if none). */
    Bucket cbucket(int min, int z) { SizeRow row = cbucket[min]; return row == null ? null : row.get(z); }

    /**
     * Exact inverted index for the complement clusters of one minimum species m: for every species x, the ids
     * (ascending) of complement clusters comp(Y) with min m whose base cluster Y contains x. A complement comp(Y) is
     * inside C iff Y contains every species of all - C, so a resolution scan of C only needs the list of ONE
     * species x not in C (the shortest one). Built once, when bucket m holds many complements.
     */
    int[][] compIndex(int m) {
        if (compIdx == null) compIdx = new int[nTaxa][][];
        if (compIdx[m] != null) return compIdx[m];
        IntList[] lists = new IntList[nTaxa];
        for (int x = 0; x < nTaxa; x++) lists[x] = new IntList(4);
        SizeRow row = cbucket[m];
        IntList all = new IntList(compCount[m]);
        if (row != null) for (int k = 0; k < row.n; k++) { Bucket b = row.b[k]; for (int i = 0; i < b.n; i++) all.add(b.ids[i]); }
        int[] allIds = all.toArray(); Arrays.sort(allIds);
        for (int id : allIds) for (int x : taxa(compOf[id])) lists[x].add(id);
        int[][] out = new int[nTaxa][];
        for (int x = 0; x < nTaxa; x++) out[x] = lists[x].toArray();
        return compIdx[m] = out;
    }
    private int[][][] compIdx;

    private void grow() {
        int c = size.length + (size.length >> 1) + 16;              // 1.5x: less slack at the memory peak
        h1 = Arrays.copyOf(h1, c); h2 = Arrays.copyOf(h2, c); size = Arrays.copyOf(size, c); minT = Arrays.copyOf(minT, c);
        repTree = Arrays.copyOf(repTree, c); repL = Arrays.copyOf(repL, c); repR = Arrays.copyOf(repR, c);
        compOf = Arrays.copyOf(compOf, c); fold = Arrays.copyOf(fold, c); cfold = Arrays.copyOf(cfold, c);
        if (expl != null) expl = Arrays.copyOf(expl, c);
        if (sig != null) { sig = Arrays.copyOf(sig, 4 * c); csig = Arrays.copyOf(csig, 4 * c); }
        bkey = Arrays.copyOf(bkey, c);
        if (BW > 0) bits = Arrays.copyOf(bits, BW * c);
    }
}
