package phylonet.coalescent.fast;

import java.util.stream.IntStream;

/**
 * Scores of the candidate bipartitions x = (P|Q) of every DP state V = P u Q, fused with the DP.
 *
 * MGD weight  W(x) = sum f(y) over valid gene STBs y = (Ya|Yb) dominated by x.
 *   A dominated y has one side inside the SMALLER side S of x and the other inside V - S.
 *   So we enumerate known side clusters K subset S (species buckets of S, size <= |S|,
 *   signature prefilter, exact mark test), follow adjacency K -> y, and test the other side
 *   against the marks of V - S. Cost scales with |S| instead of |UGB|.
 *
 * MGDL (DynaDup -dl wd, identical formula):
 *   c(x) = -( DLC(V) - 3 W - |T_P u T_Q| - SSM + wd (SSM + |T_P u T_Q| + |T_P n T_Q|) )
 * with, for gene tree t and cluster Z, comps_t(Z) = (#leaves of t in Z) - (#internal nodes of t
 * whose species set is inside Z) = number of maximal Z-pure subtrees:
 *   DLC(V)      = sum_t comps_t(V)                          (= calculateDLstdClusterCost)
 *   XL(t, Z)    = max(comps_t(Z) - 1, 0)                    (= getClusterCoalNum_rooted)
 *   T_Z         = trees containing a species of Z
 *   SSM         = sum_{t in T_P - T_Q} XL(t,P) + sum_{t in T_Q - T_P} XL(t,Q)
 *
 * Memory-lean evaluation. The terms are rewritten so that NO per-cluster, per-tree list is kept:
 *   |T_P u T_Q| = |T_V|  and  |T_P n T_Q| = |T_P| + |T_Q| - |T_V|                       (three counts)
 *   SSM = sum_{t absent from Q} XL(t,V) + sum_{t absent from P} XL(t,V)
 * because a tree without Q-species has comps_t(V) = comps_t(P), and a tree without any V-species has XL = 0.
 * So a cluster only stores |T_Z| and the (usually tiny) set of trees in which it is ABSENT; comps_t(V) is
 * computed once per state into a per-thread scratch row and discarded after the state's candidates are scored.
 * The DP runs layer by layer (states of equal size are independent), so no per-candidate score arrays exist.
 */
final class Scorer {
    static final boolean NO_COMPLEMENT_PATH = System.getenv("DDX_NO_COMPLEMENT_PATH") != null;   // debug switch

    final ClusterTable ct; final GeneData g; final StateSpace ss;
    final boolean dl; final double wd;
    double lambda = Double.NaN;
    double mu = 0;                        // weight of DynaDup's 'bd' missing-taxa loss term B(x) = SSM + |T_V| + |T_P n T_Q|
    long[] gpuW;                          // GPU-computed W_dom per candidate (null = compute on CPU)
    boolean verifyGpu;
    GpuRunner gpuComps;                   // if set: comps rows of each DP layer are computed on the GPU in batches
    private final int words;
    private final java.util.ArrayList<Scratch> allScratch = new java.util.ArrayList<>();
    static final boolean PROF = System.getenv("DDX_PROFILE") != null;
    /** 0 = per-candidate weights, 1 = bit-parallel weights for every state, 2 = automatic choice per state */
    static final int BPW_MODE = Integer.getInteger("dynadupx.bpw", 2);
    static final boolean CHECK_W = System.getenv("DDX_CHECK_W") != null;      // debug: compare both weight methods
    /** MGDL loss terms of small/medium states from totals + per-tree counts on the few trees that need them */
    static final boolean SPARSE = !"0".equals(System.getProperty("dynadupx.sparse", "1"));
    static final boolean CHECK_SPARSE = System.getenv("DDX_CHECK_SPARSE") != null;   // debug: compare with the full comps row
    final java.util.concurrent.atomic.LongAdder tCompN = new java.util.concurrent.atomic.LongAdder(), tCompC = new java.util.concurrent.atomic.LongAdder(),
            tW = new java.util.concurrent.atomic.LongAdder(), nCompN = new java.util.concurrent.atomic.LongAdder(), nCompC = new java.util.concurrent.atomic.LongAdder(),
            nW = new java.util.concurrent.atomic.LongAdder();

    // per-cluster presence data (MGDL only): |T_Z| and the trees where Z is absent
    private int[] presCount;              // -1 = not yet computed
    private Object[] absent;              // int[] sorted absent trees, or long[] bitset of PRESENT trees
    private static final int[] NONE = new int[0];
    // singleton clusters: cluster id == species id
    private double[] singleVal; private long[] singleXL;
    /** I(V) = internal nodes (all trees) whose species set is inside V, per state (sparse MGDL terms):
     *  I(V) = I(A) + I(B) + (node occurrences of clusters split by (A|B)), for the first candidate (A|B) of V. */
    private int[] stateI;
    private long iOf(int Z) { return ct.size[Z] == 1 ? g.spSingle[Z] : stateI[ss.state(Z)]; }

    Scorer(ClusterTable ct, GeneData g, StateSpace ss, boolean dl, double wd) {
        this.ct = ct; this.g = g; this.ss = ss; this.dl = dl; this.wd = wd;
        words = (g.k + 63) >>> 6;
    }

    static final class Scratch {
        final int[] mark, mark2, Lk, Ik, comps; final IntList touched = new IntList(64); int stamp = 1; final long[] sig = new long[4];
        long[] nodeBits;                       // complement comps walk: one bit per internal node
        int[] taxaV; final IntList posBuf = new IntList(64);   // species of the current state; leaf positions (treeXLFromLeaves)
        long totalComps; int present;          // of the cluster whose comps row is in `comps`
        Scratch(int n, int k) { mark = new int[n]; mark2 = new int[n]; Lk = new int[k]; Ik = new int[k]; comps = new int[k]; }
        private long[] bS, bO, bV;
        long[] bitsS(int w) { if (bS == null) bS = new long[w]; return bS; }
        long[] bitsO(int w) { if (bO == null) bO = new long[w]; return bO; }
        long[] bitsV(int w) { if (bV == null) bV = new long[w]; return bV; }
        long cScan, cSub, cAdj;
        // bit-parallel weights
        final IntList inside = new IntList(64); int insideFor = -1;   // q of the node clusters inside cluster insideFor
        long[] io, wbuf = new long[64]; int[] sorted = new int[64], sizeCnt;   // io: 3 * (node clusters + species) words, shared by both weight modes
        // per-tree XL of the current state (sparse MGDL path), memoised per state
        int[] treeStamp, treeXL, pure; int treeCur = 0;
        long insideOcc;                                     // I(V) of the current state
        int[] estStamp; int estCur = 0;
        long straddle;
        long[] wc = new long[8];   // PROF: prefix states, list states, q visited, inside q, split q, y tested, hits, chunks
        // sparse shared passes: cluster -> row in lio (-1 = not visited in this pass), two-level visit mask, lane union
        int[] qMap, wRank, kvq, ksp; int kNv; long[] qMask, qSum, uBits, lio; final IntList visited = new IntList(1024), splitList = new IntList(1024); int[] ra, rb; long tGat, tPro, hits, yHit, f1, nBig; int[] pairStamp; int passStamp; long[][] vbuf; final long[] tph = new long[4];
        int dirtyEnd; long nDense;           // rows [0, dirtyEnd) were written by a full pass (sparse passes need them zero)
    }

    /** DDX_BENCH_COMPS: time both comps paths on a sample of states per size decile (tuning aid). */
    void benchComps() {
        int S = ss.states.size(); java.util.Random r = new java.util.Random(1);
        Scratch sc = new Scratch(g.n, g.k);
        int bins = 10; long[] tn = new long[bins], tc = new long[bins]; int[] cnt = new int[bins];
        for (int it = 0; it < 4000; it++) {
            int V = ss.states.a[r.nextInt(S)]; int b = Math.min(bins - 1, ct.size[V] * bins / g.n);
            if (cnt[b] >= 200) continue;
            long t0 = System.nanoTime(); computeCompsNormal(V, sc); long x = sc.totalComps; clearComps(sc); long t1 = System.nanoTime();
            computeCompsComplement(V, sc); long y = sc.totalComps; clearComps(sc); long t2 = System.nanoTime();
            if (x != y) throw new IllegalStateException("comps paths disagree for cluster " + V);
            tn[b] += t1 - t0; tc[b] += t2 - t1; cnt[b]++;
        }
        for (int b = 0; b < bins; b++) if (cnt[b] > 0)
            System.err.printf("[bench] size %3d-%3d%%: %d states, normal %.2f ms, complement %.2f ms%n", 100 * b / bins, 100 * (b + 1) / bins, cnt[b], tn[b] / 1e6 / cnt[b], tc[b] / 1e6 / cnt[b]);
    }
    private void computeCompsNormal(int id, Scratch sc) { computeCompsImpl(id, sc, false); }

    // ------------------------------------------------------------------------------------------------ DP
    /** Result of the DP: value and chosen candidate of every state. */
    static final class Result { double[] V; int[] best; }

    /**
     * Bottom-up DP fused with scoring. States are processed in layers of equal size (a state's candidate sides
     * are strictly smaller, hence finished); inside a layer states are independent and run in parallel.
     * For each state the candidates are scanned in their enumeration order with a strict '>' test, exactly as
     * the former two-pass implementation (scores array, then DP), so ties are broken identically.
     */
    Result solve(int[] cA, int[] cB) {
        final int S = ss.states.size(), n = g.n;
        final int[] stCl = ss.states.a;
        if (dl) { presCount = new int[ct.n]; java.util.Arrays.fill(presCount, -1); absent = new Object[ct.n]; stateI = new int[S]; java.util.Arrays.fill(stateI, -1);
            presBudget = Long.getLong("dynadupx.presBudget", (long) (0.25 * Runtime.getRuntime().maxMemory())); }
        singleVal = new double[n]; singleXL = new long[n];
        {
            Scratch sc = new Scratch(n, g.k);
            for (int v = 0; v < n; v++) {
                if (ct.size[v] != 1 || ct.taxa(v)[0] != v) throw new IllegalStateException("singleton cluster id " + v + " is not species " + v);
                if (!dl) continue;
                computeComps(v, sc);
                long xl = 0;
                for (int j = 0; j < sc.touched.n; j++) { int c = sc.comps[sc.touched.a[j]]; if (c > 1) xl += c - 1; }
                singleXL[v] = xl;
                singleVal[v] = Double.isNaN(lambda) ? -xl : -lambda * xl;
                storePresence(v, sc);
                clearComps(sc);
            }
        }
        // states ordered by size (stable counting sort = the former stable sort by size)
        int[] cnt = new int[n + 2];
        for (int i = 0; i < S; i++) cnt[ct.size[stCl[i]] + 1]++;
        for (int z = 0; z <= n; z++) cnt[z + 1] += cnt[z];
        int[] layerStart = cnt.clone();
        int[] order = new int[S];
        for (int i = 0; i < S; i++) order[cnt[ct.size[stCl[i]]]++] = i;
        // within a layer, states with the same rarest species next to each other, so the states sharing a pass
        // overlap and the pass visits fewer distinct gene clusters (states of a layer are independent: same results)
        if (LAYER_ORDER && ct.bits != null && ct.rank != null && n < (1 << 11)) {
            final int[] rk = ct.rank; final int BW = ct.BW; final long[] bits = ct.bits;
            IntStream.rangeClosed(2, n).parallel().forEach(z -> {
                final int lo = layerStart[z], hi = layerStart[z + 1];
                if (hi - lo < 2) return;
                long cands = 0; for (int i = lo; i < hi; i++) { final int si = order[i]; cands += ss.candEnd.a[si] - ss.candStart.a[si]; }
                if (cands > LAYER_ORDER_MAXC * (hi - lo)) return;      // many candidates per state (-x0, -xc): few states per pass, keep the order
                final long[] key = new long[hi - lo];
                for (int i = lo; i < hi; i++) {
                    final int si = order[i], o = BW * stCl[si]; int a = Integer.MAX_VALUE, b = Integer.MAX_VALUE, c = Integer.MAX_VALUE;
                    for (int w = 0; w < BW; w++) { long x = bits[o + w]; while (x != 0) { final int r = rk[(w << 6) + Long.numberOfTrailingZeros(x)]; x &= x - 1;
                        if (r < a) { c = b; b = a; a = r; } else if (r < b) { c = b; b = r; } else if (r < c) c = r; } }
                    if (b == Integer.MAX_VALUE) b = 0;
                    if (c == Integer.MAX_VALUE) c = 0;
                    key[i - lo] = ((long) a << 53) | ((long) b << 42) | ((long) c << 31) | si;
                }
                java.util.Arrays.sort(key);
                for (int i = lo; i < hi; i++) order[i] = (int) (key[i - lo] & ((1L << 31) - 1));
            });
        }
        final double[] V = new double[S]; final int[] best = new int[S];
        // the presence data of a cluster (absent-tree set) is read only while scoring the states that have it
        // as a candidate side; it is freed after the last such layer (the events report recomputes it when needed)
        int[] freeStart = null, freeList = null;
        if (dl && FREE_PRESENCE) {
            int[] last = new int[ct.n];
            for (int si = 0; si < S; si++) {
                int z = ct.size[stCl[si]];
                for (int c = ss.candStart.a[si], e = ss.candEnd.a[si]; c < e; c++) {
                    if (last[cA[c]] < z) last[cA[c]] = z;
                    if (last[cB[c]] < z) last[cB[c]] = z;
                }
            }
            freeStart = new int[n + 3]; freeList = new int[S + n];
            for (int v = 0; v < n; v++) freeStart[Math.max(2, last[v]) + 1]++;
            for (int si = 0; si < S; si++) { int C = stCl[si]; freeStart[Math.max(ct.size[C], last[C]) + 1]++; }
            for (int z = 0; z <= n + 1; z++) freeStart[z + 1] += freeStart[z];
            int[] fill = java.util.Arrays.copyOf(freeStart, n + 2);
            for (int v = 0; v < n; v++) freeList[fill[Math.max(2, last[v])]++] = v;
            for (int si = 0; si < S; si++) { int C = stCl[si]; freeList[fill[Math.max(ct.size[C], last[C])]++] = C; }
        }
        // shared passes with 128 candidates (two words per lane set) when the per-thread buffer (48 bytes per gene cluster) is
        // small compared with the heap; otherwise 64
        final int thr = Math.max(1, java.util.concurrent.ForkJoinTask.getPool() != null ? java.util.concurrent.ForkJoinTask.getPool().getParallelism() : Runtime.getRuntime().availableProcessors());
        wide = WIDE_MODE == 1 || (WIDE_MODE == 0 && 48.0 * (g.nodeCount + n) * (thr + 1) <= Runtime.getRuntime().maxMemory() / 16.0);
        // sparse passes with rows indexed by code (48 bytes per gene cluster and thread) when that fits into the free heap;
        // otherwise compact rows whose large buffers are reserved from a heap budget
        if (SPARSE_PASS && g.prQ != null && g.ndRec != null && PREFIX_MODE != 2 && BPW_MODE != 0 && g.nodeIdx != null && (gpuW == null || verifyGpu)) {
            long maxLayer = 0, maxState = 0;
            for (int z = 2; z <= n; z++) { long c = 0; for (int i = layerStart[z]; i < layerStart[z + 1]; i++) { int si = order[i]; long k = ss.candEnd.a[si] - ss.candStart.a[si]; c += k; maxState = Math.max(maxState, k); } maxLayer = Math.max(maxLayer, c); }
            maxLayer = Math.min(maxLayer, LAYER_CHUNK + maxState);                                // layers run in chunks
            final long NC = g.nodeCount;
            final long perThreadFixed = (NC >>> 3) + (NC >>> 4) + 8L * g.k + 64L * n;          // visit mask, word ranks, small rows
            final long free = freeHeap();
            // the MGDL presence sets are kept within a budget (a set over it is recomputed when needed)
            if (dl) presBudget = Long.getLong("dynadupx.presBudget", (long) Math.min(0.25 * Runtime.getRuntime().maxMemory(), 0.35 * free));
            final long dlRes = dl ? (long) thr * ((g.totalInternal >>> 3) + 16L * g.k + 3L * 8 * words) + presBudget : 0;  // comps walk, presence data
            final long budget = (long) (0.7 * Math.max(0, free - 41L * maxLayer - dlRes - (long) thr * perThreadFixed - (64L << 20)));
            final long directNeed = (long) thr * 48L * (NC + n);
            directRows = DIRECT_MODE == 1 || (DIRECT_MODE == 0 && directNeed <= budget);
            if (!directRows) {
                final long b = Math.max(budget, rowBytes(n, 4096) * thr);
                rowBudget = new RowBudget(Long.getLong("dynadupx.rowBudget", b / 2));                       // (test switches)
                keepNv = Long.getLong("dynadupx.keepRows", Math.max(1024, (b / 2 / thr - 48L * (n + 1)) / 64));
                final long largest = rowBytes(n, NC);
                if (DIRECT_MODE == 0 || PROF) System.err.printf("[memory] weight passes use compact rows with a %.0f MB budget: direct rows for %d threads would need %.0f MB, "
                    + "free heap %.0f MB (a larger -Xmx or fewer -threads makes the scoring faster; results are the same)%s%n", b / 1048576.0, thr, directNeed / 1048576.0, free / 1048576.0,
                    largest > b / 2 ? String.format("; the largest pass may need up to %.0f MB and then runs alone", largest / 1048576.0) : "");
                if (PROF) System.err.printf("[profile] compact rows: per thread up to %d rows kept, shared pool %.0f MB%n", keepNv, rowBudget.cap / 1048576.0);
            }
        } else directRows = DIRECT_MODE == 1;
        ThreadLocal<Scratch> tl = ThreadLocal.withInitial(() -> { Scratch x = new Scratch(g.n, g.k); synchronized (allScratch) { allScratch.add(x); } return x; });
        int batch = Math.max(1, Integer.getInteger("dynadupx.gpu.compsBatch", 256));
        long tStart = System.nanoTime(), tLast = tStart;
        for (int z = 2; z <= n; z++) {
            if (freeList != null) for (int j = freeStart[z - 1]; j < freeStart[z]; j++) { final int x = freeList[j]; unkeep(absent[x]); absent[x] = null; }   // last read in layer z - 1
            final int lo0 = layerStart[z], hi0 = layerStart[z + 1];
            if (PROF && System.nanoTime() - tLast > 30_000_000_000L) {
                tLast = System.nanoTime();
                System.err.printf("[profile] DP: layer %d of %d, %d of %d states done, %.0fs%n", z, n, lo0, S, (tLast - tStart) / 1e9);
            }
            if (lo0 >= hi0) continue;
            if (dl && gpuComps != null) {
                for (int b = lo0; b < hi0; b += batch) {
                    final int e = Math.min(hi0, b + batch), b0 = b;
                    int[] cl = new int[e - b];
                    for (int i = b; i < e; i++) cl[i - b] = stCl[order[i]];
                    final int[] rows = gpuComps.compsRows(cl);
                    if (verifyGpu) verifyRows(cl, rows);
                    IntStream.range(b, e).parallel().forEach(i -> processState(order[i], cA, cB, V, best, tl.get(), rows, (i - b0) * g.k, null, -1, 0));
                }
            } else if (BPW_MODE == 0 || g.nodeIdx == null || (gpuW != null && !verifyGpu)) {
                IntStream.range(lo0, hi0).parallel().forEach(i -> processState(order[i], cA, cB, V, best, tl.get(), null, 0, null, -1, 0));
            } else for (int clo = lo0; clo < hi0; ) {
                // a layer is processed in chunks of at most LAYER_CHUNK candidates (bounded per-layer arrays; the states
                // of a layer are independent, so the results do not change)
                int chi = clo; long cc = 0;
                while (chi < hi0) { final int si = order[chi]; final long k = ss.candEnd.a[si] - ss.candStart.a[si]; if (chi > clo && cc + k > LAYER_CHUNK) break; cc += k; chi++; }
                final int lo = clo, hi = chi; clo = chi;
                // states whose weights are cheaper in a shared pass over all node clusters of size <= z: their candidates
                // are packed 64 per pass across states (all states of the layer have the same size z)
                final int m = hi - lo;
                final int[] preOff = new int[m];
                final boolean sparseAll = SPARSE_PASS && g.prQ != null && g.ndRec != null && PREFIX_MODE != 2;   // every state in shared passes
                final long[] est = sparseAll && directRows ? new long[m] : null;   // per state: clusters filed under its species (upper bound)
                if (sparseAll) {
                    java.util.Arrays.fill(preOff, 1);
                    if (est != null) {
                        final int zz0 = z;
                        IntStream.range(lo, hi).parallel().forEach(i -> {
                            final int V0 = stCl[order[i]], BW = ct.BW, o = BW * V0; final long[] bits = ct.bits; final int[] cz = g.cntAZ; long e = 0;
                            for (int w = 0; w < BW; w++) { long x = bits[o + w]; while (x != 0) { e += cz[((w << 6) + Long.numberOfTrailingZeros(x)) * (n + 1) + zz0]; x &= x - 1; } }
                            est[i - lo] = e;
                        });
                    }
                }
                else IntStream.range(lo, hi).parallel().forEach(i -> { int si = order[i]; preOff[i - lo] = usePrefix(stCl[si], ss.candEnd.a[si] - ss.candStart.a[si], tl.get()) ? 1 : 0; });
                int tot = 0;
                for (int j = 0; j < m; j++) { if (preOff[j] == 0) { preOff[j] = -1; continue; } int si = order[lo + j]; preOff[j] = tot; tot += ss.candEnd.a[si] - ss.candStart.a[si]; }
                final long[] layerW = new long[2 * tot];                // [weight..., straddle...]
                if (tot > 0) {
                    final int[] flatC = new int[tot], flatV = new int[tot];
                    final boolean[] first = new boolean[tot];
                    final long[] estL = est != null ? new long[tot] : null;
                    for (int j = 0; j < m; j++) if (preOff[j] >= 0) { int si = order[lo + j], c0 = ss.candStart.a[si], c1 = ss.candEnd.a[si];
                        first[preOff[j]] = true;
                        if (estL != null) estL[preOff[j]] = est[j];
                        for (int c = c0; c < c1; c++) { flatC[preOff[j] + c - c0] = c; flatV[preOff[j] + c - c0] = stCl[si]; } }
                    final int zz = z, T = tot;
                    long t0 = PROF ? System.nanoTime() : 0;
                    if (sparseAll && directRows) {
                        // per pass: sparse if the clusters filed under the species of its states are few compared with all
                        // node clusters of size <= z, else one sequential pass over all of them (large states)
                        final long qEndZ = g.prefixEnd[z];
                        IntStream.range(0, (tot + 127) / 128).parallel().forEach(ch -> {
                            final int j0 = ch * 128, j1 = Math.min(T, ch * 128 + 128);
                            long e = estL[j0];
                            for (int j = j0 + 1; j < j1; j++) if (first[j]) e += estL[j];
                            // (the first lane's state may have started in an earlier pass: its estimate is at its first lane)
                            if (!first[j0]) { int j = j0; while (j > 0 && !first[j]) j--; e += estL[j]; }
                            final Scratch sc = tl.get();
                            if (zz <= SPARSE_MAXZ * n && e <= SPARSE_RATIO * qEndZ) {
                                if (sc.dirtyEnd > 0) { java.util.Arrays.fill(sc.io, 0, 6 * sc.dirtyEnd, 0L); sc.dirtyEnd = 0; }
                                sparseWeightsDirect(zz, flatC, flatV, first, j0, j1, cA, cB, layerW, T, sc);
                            } else {
                                batchedWeights2(zz, flatC, flatV, first, j0, j1, cA, cB, layerW, T, sc);
                                sc.dirtyEnd = Math.max(sc.dirtyEnd, (int) qEndZ);
                                if (PROF) sc.nDense++;
                            }
                        });
                    }
                    else if (sparseAll) IntStream.range(0, (tot + 127) / 128).parallel().forEach(ch -> sparseWeights(zz, flatC, flatV, first, ch * 128, Math.min(T, ch * 128 + 128), cA, cB, layerW, T, tl.get()));
                    else if (wide) IntStream.range(0, (tot + 127) / 128).parallel().forEach(ch -> batchedWeights2(zz, flatC, flatV, first, ch * 128, Math.min(T, ch * 128 + 128), cA, cB, layerW, T, tl.get()));
                    else IntStream.range(0, (tot + 63) / 64).parallel().forEach(ch -> batchedWeights(zz, flatC, flatV, first, ch * 64, Math.min(T, ch * 64 + 64), cA, cB, layerW, T, tl.get()));
                    if (PROF) { tW.add(System.nanoTime() - t0); nW.add(tot); }
                }
                final int totF = tot;
                IntStream.range(lo, hi).parallel().forEach(i -> processState(order[i], cA, cB, V, best, tl.get(), null, 0, preOff[i - lo] >= 0 ? layerW : null, preOff[i - lo], totF));
            }
        }
        if (PROF) {
            long[] cs = new long[3];
            synchronized (allScratch) { for (Scratch x : allScratch) { cs[0] += x.cScan; cs[1] += x.cSub; cs[2] += x.cAdj; } }
            long[] w8 = new long[8];
            synchronized (allScratch) { for (Scratch x : allScratch) for (int i = 0; i < 8; i++) w8[i] += x.wc[i]; }
            System.err.printf("[profile] bit-parallel weights: prefix-mode states %d, list-mode states %d, prefix q visited %d, inside q %d, split q %d, y tested %d, 64-candidate chunks %d%n",
                w8[0], w8[1], w8[2], w8[3], w8[4], w8[5], w8[7]);
            long tg = 0, tp = 0, hh = 0, yh = 0, ff = 0; synchronized (allScratch) { for (Scratch x : allScratch) { tg += x.tGat; tp += x.tPro; hh += x.hits; yh += x.yHit; ff += x.f1; } }
            long[] ph = new long[4]; synchronized (allScratch) { for (Scratch x : allScratch) for (int i = 0; i < 4; i++) ph[i] += x.tph[i]; }
            System.err.printf("[profile] sparse passes: STBs with a hit %d (frequency 1: %d), lane hits %d; phases A %.1fs B %.1fs C %.1fs D %.1fs%n", yh, ff, hh, ph[0] / 1e9, ph[1] / 1e9, ph[2] / 1e9, ph[3] / 1e9);
            long nd = 0; synchronized (allScratch) { for (Scratch x : allScratch) nd += x.nDense; }
            System.err.printf("[profile] full passes (large states): %d%n", nd);
            if (dl) System.err.printf("[profile] MGDL presence sets: peak %.0f MB stored (budget %.0f MB), %d recomputed on demand instead of stored, %d recomputations%n",
                presPeak / 1048576.0, presBudget / 1048576.0, nVirt.sum(), nVirtUse.sum());
            if (rowBudget != null) { long nb = 0; synchronized (allScratch) { for (Scratch x : allScratch) nb += x.nBig; }
                System.err.printf("[profile] compact rows: %d passes with budgeted buffers, peak %.0f MB of %.0f MB, %d waits%n", nb, rowBudget.peak / 1048576.0, rowBudget.cap / 1048576.0, rowBudget.waits); }
            if (SPARSE_PASS && g.ndRec != null) System.err.printf("[profile] sparse passes: %d, gathered (with repeats) %d, partner entries %d, visited %d, inside a lane %d; gather %.1fs, visit %.1fs (thread-seconds)%n", w8[0], w8[1], w8[6], w8[2], w8[3], tg / 1e9, tp / 1e9);
            System.err.printf("[profile] scorer: comps small %d in %.1fs, comps large %d in %.1fs, weights %d in %.1fs (thread-seconds); "
                + "weight: side entries scanned %d, sides inside S %d, adjacency checks %d%n",
                nCompN.sum(), tCompN.sum() / 1e9, nCompC.sum(), tCompC.sum() / 1e9, nW.sum(), tW.sum() / 1e9, cs[0], cs[1], cs[2]);
        }
        if (STATS) printStats();
        // the per-thread pass buffers are not needed after the DP; free them for the report that follows (EVENTS)
        synchronized (allScratch) { for (Scratch x : allScratch) {
            x.io = null; x.lio = null; x.kvq = null; x.ksp = null; x.ra = null; x.rb = null; x.kNv = 0; x.qMap = null; x.qMask = null; x.qSum = null;
            x.wRank = null; x.nodeBits = null; x.dirtyEnd = 0; } }
        Result r = new Result(); r.V = V; r.best = best;
        return r;
    }

    private void processState(int si, int[] cA, int[] cB, double[] Vv, int[] best, Scratch sc, int[] rows, int rowOff, long[] preW, int preOff, int preTot) {
        int V = ss.states.a[si];
        int stampV = sc.stamp++;
        final int[] taxaV = ct.taxa(V); sc.taxaV = taxaV;
        for (int v : taxaV) sc.mark[v] = stampV;
        final int c0 = ss.candStart.a[si], c1 = ss.candEnd.a[si];
        final boolean bpw = (gpuW == null || verifyGpu) && BPW_MODE != 0 && g.nodeIdx != null;
        final boolean totals = dl && rows == null && SPARSE && bpw;
        if (dl && rows != null) loadRow(rows, rowOff, sc);
        long straddle = 0;                                      // node occurrences split by the first candidate
        if (preW != null) {                                     // weights from the layer's shared pass
            if (sc.wbuf.length < c1 - c0) sc.wbuf = new long[Math.max(c1 - c0, 2 * sc.wbuf.length)];
            System.arraycopy(preW, preOff, sc.wbuf, 0, c1 - c0);
            straddle = preW[preTot + preOff];
        } else if (bpw) {
            long t0 = PROF ? System.nanoTime() : 0;
            if (sc.insideFor != V) collectInside(V, sc);
            weightsBPW(V, c0, c1, cA, cB, sc);
            straddle = sc.straddle;
            if (PROF) { tW.add(System.nanoTime() - t0); nW.add(c1 - c0); }
        }
        if (totals) sc.insideOcc = iOf(cA[c0]) + iOf(cB[c0]) + straddle;      // = I(V), size-1 node clusters included
        // MGDL loss terms: from totals plus per-tree counts on the trees where a side is absent ("sparse"), or from the
        // full per-tree comps row when that is cheaper (or when the row came from the GPU)
        boolean row = dl && rows != null;
        long dlc = 0, tvS = 0; Object absV = null;
        if (dl && !row) {
            long t0 = PROF ? System.nanoTime() : 0;
            boolean sparse = false;
            if (totals) {
                long copiesV = 0, singleV = 0, occLenV = 0;
                if (ct.bits != null) {
                    final int BW = ct.BW; final long[] bits = ct.bits;
                    for (int w = 0, o = BW * V; w < BW; w++) { long b = bits[o + w]; while (b != 0) { int x = (w << 6) + Long.numberOfTrailingZeros(b); b &= b - 1;
                        copiesV += g.spCopies[x]; singleV += g.spSingle[x]; occLenV += g.occTree[x].length; } }
                } else for (int x : ct.taxa(V)) { copiesV += g.spCopies[x]; singleV += g.spSingle[x]; occLenV += g.occTree[x].length; }
                final long insideV = sc.insideOcc;                                   // internal nodes inside V
                stateI[si] = (int) insideV;
                dlc = copiesV - insideV;
                if (g.allComplete) { tvS = g.k; sparse = true; }
                else {
                    absV = intersectAbsent(absOf(cA[c0], sc, 0), absOf(cB[c0], sc, 1));
                    tvS = g.k - absentCount(absV);
                    long costNormal = occLenV + insideV + g.k;
                    long costComp = (g.totalCopies - copiesV) + (g.totalInternal - insideV) + g.k;
                    long costSparse = sparseCost(c0, c1, cA, cB, absV, sc, Math.min(costNormal, costComp));
                    if (costSparse <= Math.min(costNormal, costComp) || SPARSE_ALWAYS) { sparse = true; newTreeMemo(sc, absV); }
                    else { computeCompsImpl(V, sc, costComp < costNormal && g.parIntAll != null); row = true; }
                }
                if (sparse && CHECK_SPARSE) {
                    computeComps(V, sc);
                    if (sc.totalComps != dlc || sc.present != tvS)
                        throw new IllegalStateException("sparse totals " + dlc + "/" + tvS + " != " + sc.totalComps + "/" + sc.present + " for cluster " + V);
                }
            }
            if (!totals) { computeComps(V, sc); row = true; }
            if (PROF) { long d = System.nanoTime() - t0; if (row) { tCompC.add(d); nCompC.increment(); } else { tCompN.add(d); nCompN.increment(); } }
        }
        double bv = Double.NEGATIVE_INFINITY; int arg = -1;
        for (int c = c0; c < c1; c++) {
            int A = cA[c], B = cB[c];
            long w;
            if (bpw) {
                w = sc.wbuf[c - c0];
                if (CHECK_W || verifyGpu) {
                    long w0 = weight(A, B, stampV, sc);
                    if (w0 != w) throw new IllegalStateException("bit-parallel weight " + w + " != " + w0 + " for candidate " + c + " of cluster " + V);
                    if (gpuW != null && gpuW[c] != w) throw new IllegalStateException("GPU/CPU weight mismatch at candidate " + c + ": gpu=" + gpuW[c] + " cpu=" + w);
                }
            }
            else if (!PROF) w = candidateWeight(c, A, B, stampV, sc);
            else { long t0 = System.nanoTime(); w = candidateWeight(c, A, B, stampV, sc); tW.add(System.nanoTime() - t0); nW.increment(); }
            double score;
            if (!dl) score = w;
            else {
                long tv, both, ssm, tot;
                if (!row) {
                    tot = dlc;
                    if (g.allComplete) { tv = g.k; both = g.k; ssm = 0; }
                    else {
                        tv = tvS; both = (long) presCount[A] + presCount[B] - tv;
                        ssm = absentXLTrees(absOf(A, sc, 0), stampV, sc) + absentXLTrees(absOf(B, sc, 1), stampV, sc);
                        if (CHECK_SPARSE) { long o = absentXL(A, sc) + absentXL(B, sc); if (o != ssm) throw new IllegalStateException("sparse ssm " + ssm + " != " + o + " for candidate " + c); }
                    }
                } else {
                    tot = sc.totalComps;
                    if (g.allComplete) { tv = g.k; both = g.k; ssm = 0; }
                    else { tv = sc.present; both = (long) presCount[A] + presCount[B] - tv; ssm = absentXL(A, sc) + absentXL(B, sc); }
                }
                long xl = tot - tv - ssm;
                score = Double.isNaN(lambda) ? -(tot - 3.0 * w - tv - ssm + wd * (ssm + tv + both)) : w - lambda * xl - mu * (ssm + tv + both);
            }
            double v = score + val(A, Vv) + val(B, Vv);
            if (v > bv) { bv = v; arg = c; }
        }
        if (arg < 0) throw new IllegalStateException("Was not able to build a fully resolved tree for a cluster of size " + ct.size[V]);
        Vv[si] = bv; best[si] = arg;
        if (STATS) stats(V, si, cA, cB, sc);
        if (dl && totals && stateI[si] < 0) stateI[si] = (int) sc.insideOcc;
        if (dl) {
            if (!row) {
                if (presCount[V] < 0) { absent[V] = g.allComplete ? NONE : keep(V, absV); presCount[V] = (int) tvS; }
                if (CHECK_SPARSE) clearComps(sc);
            } else { storePresence(V, sc); clearComps(sc); }
        }
        sc.insideFor = -1;
    }
    static final boolean SPARSE_ALWAYS = System.getenv("DDX_SPARSE_ALWAYS") != null;   // debug
    static final boolean FREE_PRESENCE = !"false".equals(System.getProperty("dynadupx.freePresence"));

    /** Work of the per-tree counts: total size of the distinct trees where some side is absent but V is present
     *  (stops early once it exceeds `limit`). */
    private long sparseCost(int c0, int c1, int[] cA, int[] cB, Object absV, Scratch sc, long limit) {
        if (sc.treeStamp == null) { sc.treeStamp = new int[g.k]; sc.treeXL = new int[g.k]; sc.pure = new int[Math.max(1, g.maxInternal)]; }
        if (sc.estStamp == null) sc.estStamp = new int[g.k];
        final int cur = ++sc.estCur; final int[] es = sc.estStamp;
        if (absV instanceof int[] l) for (int t : l) es[t] = cur;
        else {
            final long[] pr = (long[]) absV; final int k = g.k;
            for (int j = 0; j < pr.length; j++) {
                long abs = ~pr[j];
                if (j == pr.length - 1 && (k & 63) != 0) abs &= (1L << (k & 63)) - 1;
                while (abs != 0) { es[(j << 6) + Long.numberOfTrailingZeros(abs)] = cur; abs &= abs - 1; }
            }
        }
        long cost = 0;
        for (int c = c0; c < c1; c++) {
            for (int side = 0; side < 2; side++) {
                Object a = absOf(side == 0 ? cA[c] : cB[c], sc, 0);
                if (a instanceof int[] list) {
                    for (int t : list) if (es[t] != cur) { es[t] = cur; cost += g.tLeaves[t].length + g.tTriples[t].length / 3; }
                } else {
                    long[] bits = (long[]) a;
                    for (int j = 0; j < bits.length; j++) {
                        long abs = ~bits[j];
                        if (j == bits.length - 1 && (g.k & 63) != 0) abs &= (1L << (g.k & 63)) - 1;
                        while (abs != 0) { int t = (j << 6) + Long.numberOfTrailingZeros(abs); abs &= abs - 1;
                            if (es[t] != cur) { es[t] = cur; cost += g.tLeaves[t].length + g.tTriples[t].length / 3; } }
                    }
                }
                if (cost > limit) return cost;
            }
        }
        return cost;
    }

    // ------------------------------------------------------------------------------------------------ sparse MGDL terms
    private long leafNet(int V) {
        long s = 0;
        if (ct.bits != null) {
            final int BW = ct.BW; final long[] bits = ct.bits;
            for (int w = 0, o = BW * V; w < BW; w++) { long b = bits[o + w]; while (b != 0) { s += g.spLeafNet[(w << 6) + Long.numberOfTrailingZeros(b)]; b &= b - 1; } }
        } else for (int x : ct.taxa(V)) s += g.spLeafNet[x];
        return s;
    }

    /** List mode (bucket scan of the node clusters inside V, per state) vs prefix mode (a pass over all node clusters of
     *  size <= |V| shared by 64 candidates of the layer), by estimated memory reads. */
    private boolean usePrefix(int V, int cands, Scratch sc) {
        final int z = ct.size[V];
        if (PREFIX_MODE != 0) return PREFIX_MODE == 1;
        if (g.prQ != null) {
            // units: one node cluster visited by a shared pass for 128 candidate lanes (64 without the wide passes)
            final int qEnd = g.prefixEnd[z];
            final double prefix = qEnd * (double) Math.max(1, cands) / (wide ? 128.0 : 64.0);
            final double per = PAIR_TEST + PAIR_W * ((cands + 63) >>> 6);   // test + list-mode weight pass per cluster
            double scan = 0;
            final int BW = ct.BW; final long[] bits = ct.bits; final int o = BW * V;
            final int[] prStart = g.prStart, prB = g.prB, prQStart = g.prQStart, prQ = g.prQ;
            for (int w = 0; w < BW; w++) {
                long x = bits[o + w];
                while (x != 0) {
                    final int a = (w << 6) + Long.numberOfTrailingZeros(x); x &= x - 1;
                    final int p0 = prStart[a], p1 = prStart[a + 1];
                    scan += PAIR_PARTNER * (p1 - p0);
                    for (int p = p0; p < p1; p++) {
                        final int b = prB[p];
                        if ((bits[o + (b >>> 6)] & (1L << b)) == 0) continue;
                        int lo = prQStart[p], hi = prQStart[p + 1];
                        if (prQ[lo] >= qEnd) continue;
                        if (prQ[hi - 1] >= qEnd) { int l = lo, h = hi; while (l < h) { int mid = (l + h) >>> 1; if (prQ[mid] < qEnd) l = mid + 1; else h = mid; } hi = l; }
                        scan += (hi - lo) * per;
                    }
                    if (scan > prefix) return true;
                }
            }
            return false;
        }
        long prefix = (long) ((PREFIX_COST > 0 ? PREFIX_COST : wide ? 0.8 : 1.2) * g.prefixEnd[z] * Math.max(1, cands) / 64.0);
        long scan = 0; final long per = g.W > 0 ? g.W + 3 : 8;
        final int[] taxa = ct.taxa(V);
        for (int v : taxa) {
            int[] zs = g.nodeSize[v];
            int lo = 0, hi = zs.length;                      // first index with size > z
            while (lo < hi) { int mid = (lo + hi) >>> 1; if (zs[mid] <= z) lo = mid + 1; else hi = mid; }
            scan += lo * per;
            if (scan > prefix) return true;
        }
        return false;
    }
    /** relative cost of a shared pass per gene cluster (tuned on 501-species inputs; <= 0: 0.8 with 128-candidate passes, else 1.2) */
    static final double PREFIX_COST = Double.parseDouble(System.getProperty("dynadupx.prefixCost", "0"));
    static final int PREFIX_MODE = Integer.getInteger("dynadupx.prefix", 0);   // 0 = auto, 1 = always, 2 = never
    /** list-mode cost with the pair index, in shared-pass units: per tested cluster, per inside cluster and 64
     *  candidates, per partner entry visited */
    static final double PAIR_TEST = Double.parseDouble(System.getProperty("dynadupx.pairTest", "3"));
    static final double PAIR_W = Double.parseDouble(System.getProperty("dynadupx.pairW", "1"));
    static final double PAIR_PARTNER = Double.parseDouble(System.getProperty("dynadupx.pairPartner", "0.25"));
    static final boolean PAIR_INDEX = !"false".equals(System.getProperty("dynadupx.pairIndex"));

    /** DLC(V) = sum_t comps_t(V) = (leaf copies in V) - (internal nodes whose species set is inside V), from totals. */
    private long totalCompsInside(int V, Scratch sc) {
        long s = 0;
        if (ct.bits != null) {
            final int BW = ct.BW; final long[] bits = ct.bits;
            for (int w = 0, o = BW * V; w < BW; w++) { long b = bits[o + w]; while (b != 0) { s += g.spLeafNet[(w << 6) + Long.numberOfTrailingZeros(b)]; b &= b - 1; } }
        } else for (int x : ct.taxa(V)) s += g.spLeafNet[x];
        final long[] occ = g.ndOcc; final int[] ins = sc.inside.a;
        for (int j = 0, m = sc.inside.n; j < m; j++) s -= occ[ins[j]];
        return s;
    }

    private int absentCount(Object a) {
        if (a instanceof int[] l) return l.length;
        long[] pr = (long[]) a; int present = 0; for (long w : pr) present += Long.bitCount(w);
        return g.k - present;
    }

    /** Trees absent from both sides = trees absent from V (same representation rule as storePresence). */
    private Object intersectAbsent(Object a, Object b) {
        final int k = g.k;
        if (a == null || b == null) throw new IllegalStateException("presence requested before it was computed");
        if (a instanceof int[] la && b instanceof int[] lb) {
            int[] out = new int[Math.min(la.length, lb.length)]; int i = 0, j = 0, m = 0;
            while (i < la.length && j < lb.length) { if (la[i] < lb[j]) i++; else if (la[i] > lb[j]) j++; else { out[m++] = la[i]; i++; j++; } }
            return m == 0 ? NONE : m == out.length ? out : java.util.Arrays.copyOf(out, m);
        }
        if (a instanceof long[] && b instanceof int[]) { Object t = a; a = b; b = t; }
        if (a instanceof int[] la) {                       // list & present-bitset
            long[] pb = (long[]) b; int m = 0; int[] out = new int[la.length];
            for (int t : la) if ((pb[t >>> 6] >>> (t & 63) & 1L) == 0) out[m++] = t;
            return m == 0 ? NONE : m == out.length ? out : java.util.Arrays.copyOf(out, m);
        }
        long[] pa = (long[]) a, pb = (long[]) b, pr = new long[pa.length];
        int present = 0;
        for (int j = 0; j < pa.length; j++) { pr[j] = pa[j] | pb[j]; present += Long.bitCount(pr[j]); }
        int nAbs = k - present;
        if (nAbs == 0) return NONE;
        if (nAbs <= Math.max(1, k >>> 5)) {
            int[] list = new int[nAbs]; int i = 0;
            for (int t = 0; t < k; t++) if ((pr[t >>> 6] >>> (t & 63) & 1L) == 0) list[i++] = t;
            return list;
        }
        return pr;
    }

    /** Starts a per-state memo of XL_t(V); trees where V is absent get XL = 0 without a scan. */
    private void newTreeMemo(Scratch sc, Object absV) {
        if (sc.treeStamp == null) { sc.treeStamp = new int[g.k]; sc.treeXL = new int[g.k]; sc.pure = new int[Math.max(1, g.maxInternal)]; }
        int cur = ++sc.treeCur;
        if (absV instanceof int[] l) for (int t : l) { sc.treeStamp[t] = cur; sc.treeXL[t] = 0; }
        else {                                              // absent trees word by word
            final long[] pr = (long[]) absV; final int k = g.k;
            for (int j = 0; j < pr.length; j++) {
                long abs = ~pr[j];
                if (j == pr.length - 1 && (k & 63) != 0) abs &= (1L << (k & 63)) - 1;
                while (abs != 0) { final int t = (j << 6) + Long.numberOfTrailingZeros(abs); abs &= abs - 1; sc.treeStamp[t] = cur; sc.treeXL[t] = 0; }
            }
        }
    }

    /** sum over the trees in an absent set of XL_t(V) (V marked with stampV in sc.mark). */
    private long absentXLTrees(Object a, int stampV, Scratch sc) {
        long s = 0;
        if (a instanceof int[] list) { for (int t : list) s += treeXL(t, stampV, sc); }
        else {
            long[] bits = (long[]) a;
            for (int j = 0; j < bits.length; j++) {
                long abs = ~bits[j];
                if (j == bits.length - 1 && (g.k & 63) != 0) abs &= (1L << (g.k & 63)) - 1;
                while (abs != 0) { int t = (j << 6) + Long.numberOfTrailingZeros(abs); abs &= abs - 1; s += treeXL(t, stampV, sc); }
            }
        }
        return s;
    }

    /** XL_t(V) = max(comps_t(V) - 1, 0), comps_t(V) = V-leaves of t - V-pure internal nodes of t (one post-order pass). */
    private int treeXL(int t, int stampV, Scratch sc) {
        if (sc.treeStamp[t] == sc.treeCur) return sc.treeXL[t];
        if (g.parIntAll != null && 16L * sc.taxaV.length < g.tLeaves[t].length) {   // few species: start from their leaves
            int xl = treeXLFromLeaves(t, sc);
            sc.treeStamp[t] = sc.treeCur; sc.treeXL[t] = xl;
            return xl;
        }
        final int[] L = g.tLeaves[t], tr = g.tTriples[t], mark = sc.mark, pure = sc.pure;
        int lc = 0;
        for (int v : L) if (mark[v] == stampV) lc++;
        int xl = 0;
        if (lc > 1) {
            int ic = 0; final int ni = tr.length / 3;
            for (int u = 0, o = 0; u < ni; u++, o += 3) {
                int l = tr[o], m = tr[o + 1], r = tr[o + 2];
                boolean pl = m == l ? mark[L[l]] == stampV : pure[u - (r - m)] != 0;
                boolean pr = r == m + 1 ? mark[L[r]] == stampV : pure[u - 1] != 0;
                boolean p = pl && pr; pure[u] = p ? 1 : 0; if (p) ic++;
            }
            int comps = lc - ic; xl = comps > 1 ? comps - 1 : 0;
        }
        sc.treeStamp[t] = sc.treeCur; sc.treeXL[t] = xl;
        return xl;
    }

    /**
     * XL_t(V) from the leaves of V's species in tree t (for small V in large trees): positions p_0 < ... of
     * those leaves (per species by binary search in its occurrence list, which is sorted by tree). Each internal node is
     * examined from its leftmost leaf only: walking up from leaf p while p is the leftmost leaf of the node u = [l..r],
     * u is V-pure iff the sorted positions contain l..r, i.e. iff pos[i + r - l] == r. The first impure node ends the walk.
     */
    private int treeXLFromLeaves(int t, Scratch sc) {
        final IntList pos = sc.posBuf; pos.n = 0;
        for (int x : sc.taxaV) {
            final int[] tt = g.occPosTree[x], pp = g.occPos[x];
            int lo = 0, hi = tt.length;                       // first entry of tree t
            while (lo < hi) { int mid = (lo + hi) >>> 1; if (tt[mid] < t) lo = mid + 1; else hi = mid; }
            for (int i = lo; i < tt.length && tt[i] == t; i++) pos.add(pp[i]);
        }
        final int lc = pos.n;
        if (lc <= 1) return 0;
        final int[] P = pos.a;
        java.util.Arrays.sort(P, 0, lc);
        final int[] tr = g.tTriples[t], parLeaf = g.parLeafAll, parInt = g.parIntAll;
        final int lo = g.leafOff[t], io = g.intOff[t];
        int ic = 0;
        for (int i = 0; i < lc; i++) {
            final int p = P[i];
            int u = parLeaf[lo + p];
            while (u >= 0) {
                final int l = tr[3 * u], r = tr[3 * u + 2];
                if (l != p) break;                             // examined from its leftmost leaf
                final int j = i + (r - l);
                if (j >= lc || P[j] != r) break;               // not V-pure, nor are its ancestors
                ic++;
                u = parInt[io + u];
            }
        }
        int comps = lc - ic;
        return comps > 1 ? comps - 1 : 0;
    }

    static final boolean STATS = System.getenv("DDX_STATS") != null;
    final java.util.concurrent.atomic.LongAdder[] st = new java.util.concurrent.atomic.LongAdder[12];
    { for (int i = 0; i < st.length; i++) st[i] = new java.util.concurrent.atomic.LongAdder(); }
    static final String[] ST_NAMES = {"states", "sum|V|", "Lk updates", "Ik updates", "inside nodes", "split nodes", "y scanned", "|D(V)|", "sum absent sides", "cands", "cands with singleton side", "T_V sum"};
    private void stats(int V, int si, int[] cA, int[] cB, Scratch sc) {
        st[0].increment(); st[1].add(ct.size[V]);
        long lk = 0; for (int x : ct.taxa(V)) lk += g.occTree[x].length; st[2].add(lk);
        if (sc.insideFor == V) { long ik = 0; for (int j = 0; j < sc.inside.n; j++) { int q = sc.inside.a[j]; ik += g.nodeTree[g.ndIdOf(q)].length; } st[3].add(ik); st[4].add(sc.inside.n); }
        if (!dl || g.allComplete) return;
        java.util.HashSet<Integer> D = new java.util.HashSet<>(); long sa = 0; int c0 = ss.candStart.a[si], c1 = ss.candEnd.a[si];
        for (int c = c0; c < c1; c++) {
            if (ct.size[cA[c]] == 1 || ct.size[cB[c]] == 1) st[10].increment();
            for (int Z : new int[]{cA[c], cB[c]}) {
                Object a = absOf(Z, sc, 0);
                if (a instanceof int[] list) { for (int t : list) if (sc.comps[t] > 0) { D.add(t); sa++; } }
                else { long[] bits = (long[]) a; for (int t = 0; t < g.k; t++) if ((bits[t >>> 6] >>> (t & 63) & 1) == 0 && sc.comps[t] > 0) { D.add(t); sa++; } }
            }
        }
        st[7].add(D.size()); st[8].add(sa); st[9].add(c1 - c0); st[11].add(sc.present);
    }
    void printStats() {
        StringBuilder b = new StringBuilder("[stats]");
        long S = Math.max(1, st[0].sum());
        for (int i = 0; i < st.length; i++) b.append(String.format(" %s=%.1f", ST_NAMES[i], i == 0 ? (double) st[0].sum() : st[i].sum() / (double) S));
        System.err.println(b + "  (per state)");
    }

    // ------------------------------------------------------------------------------------------------ bit-parallel weights
    /** Per-state choice between the bit-parallel method (one pass over the node clusters inside V for all candidates)
     *  and the per-candidate method (sides inside the smaller side S). */
    private boolean useBpw(int V, int nCand) {
        if (BPW_MODE == 0 || g.nodeIdx == null) return false;
        if (BPW_MODE == 1) return true;
        return 8 * ct.size[V] <= 5 * g.n;       // the node clusters inside V are enumerated anyway (normal comps path)
    }

    /** q of all internal-node clusters (size >= 2) inside V -> sc.inside. */
    private void collectInside(int V, Scratch sc) {
        IntList ins = sc.inside; ins.n = 0;
        final int size = ct.size[V], W = g.W;
        if (W > 0 && g.prQ != null) {                          // pair index: clusters whose two rarest species are in V
            final long[] bV = sc.bitsV(W);
            System.arraycopy(ct.bits, W * V, bV, 0, W);
            final int qEnd = g.prefixEnd[size];
            final int[] prStart = g.prStart, prB = g.prB, prQStart = g.prQStart, prQ = g.prQ;
            final long[] pb = g.prBits;
            for (int w = 0; w < W; w++) {
                long bits = bV[w];
                while (bits != 0) {
                    final int a = (w << 6) + Long.numberOfTrailingZeros(bits); bits &= bits - 1;
                    for (int p = prStart[a], pe = prStart[a + 1]; p < pe; p++) {
                        final int b = prB[p];
                        if ((bV[b >>> 6] & (1L << b)) == 0) continue;
                        node:
                        for (int k = prQStart[p], ke = prQStart[p + 1]; k < ke; k++) {
                            final int q = prQ[k];
                            if (q >= qEnd) break;                  // runs are in q order = size order
                            if (pb != null) { for (int j = 0, o = k * W; j < W; j++) if ((pb[o + j] & ~bV[j]) != 0) continue node; }
                            else { final long[] cb = ct.bits; for (int j = 0, o = W * g.ndId[q]; j < W; j++) if ((cb[o + j] & ~bV[j]) != 0) continue node; }
                            ins.add(q);
                        }
                    }
                }
            }
        } else if (W > 0) {
            final long[] bV = sc.bitsV(W);
            System.arraycopy(ct.bits, W * V, bV, 0, W);
            for (int w = 0; w < W; w++) {
                long bits = bV[w];
                while (bits != 0) {
                    int v = (w << 6) + Long.numberOfTrailingZeros(bits); bits &= bits - 1;
                    final int[] qs = g.nodeBucketQ[v], zs = g.nodeSize[v]; final long[] nb = g.nodeBits[v];
                    node:
                    for (int i = 0, p = 0; i < qs.length; i++, p += W) {
                        if (zs[i] > size) break;
                        final int q = qs[i];
                        if (q < 0) continue;
                        for (int j = 0; j < W; j++) if ((nb[p + j] & ~bV[j]) != 0) continue node;
                        ins.add(q);
                    }
                }
            }
        } else {
            int[] taxa = ct.taxa(V);
            int st = sc.stamp++;
            long[] sig = new long[4];
            for (int v : taxa) { sc.mark2[v] = st; long h = ct.H1[v]; sig[(int) ((h >>> 6) & 3)] |= 1L << (h & 63); }
            final long n0 = ~sig[0], n1 = ~sig[1], n2 = ~sig[2], n3 = ~sig[3];
            for (int v : taxa) {
                final int[] ids = g.nodeBucket[v], qs = g.nodeBucketQ[v], zs = g.nodeSize[v]; final long[] sg = g.nodeSig[v];
                for (int i = 0; i < ids.length; i++) {
                    if (zs[i] > size) break;
                    if (qs[i] < 0) continue;
                    int p = 4 * i;
                    if (((sg[p] & n0) | (sg[p + 1] & n1) | (sg[p + 2] & n2) | (sg[p + 3] & n3)) != 0) continue;
                    if (!ct.allMarked(ids[i], sc.mark2, st)) continue;
                    ins.add(qs[i]);
                }
            }
        }
        sc.insideFor = V;
    }

    /**
     * W_dom of candidates c0..c1-1 of state V into sc.wbuf, 64 candidates per machine word.
     * For a cluster K inside V let In_i(K) = [K inside S_i] and Out_i(K) = [K inside V - S_i] (S_i = a side of candidate
     * i). Singletons: In = [x in S_i]. For an internal-node cluster P = d1 u d2: In(P) = In(d1) & In(d2) and
     * Out(P) = Out(d1) & Out(d2). A valid STB y = (Ya|Yb) is dominated by candidate i iff y lies inside V and
     * (In_i(Ya) & Out_i(Yb)) | (Out_i(Ya) & In_i(Yb)). Node clusters are numbered by increasing size, so processing the
     * inside clusters by size makes d1, d2 ready. io[2c] = In, io[2c+1] = Out of the cluster with code c.
     */
    private void weightsBPW(int V, int c0, int c1, int[] cA, int[] cB, Scratch sc) {
        final int n = g.n, cands = c1 - c0, NC = g.nodeCount;
        if (sc.io == null) sc.io = new long[(wide ? 6 : 3) * (NC + n)];
        if (sc.sizeCnt == null) sc.sizeCnt = new int[n + 2];
        if (sc.wbuf.length < cands) sc.wbuf = new long[Math.max(cands, 2 * sc.wbuf.length)];
        final long[] Wb = sc.wbuf; java.util.Arrays.fill(Wb, 0, cands, 0L);
        if (PROF) sc.wc[1]++;
        // node clusters inside V by increasing size (counting sort)
        final IntList ins = sc.inside; final int m = ins.n;
        if (sc.sorted.length < m) sc.sorted = new int[Math.max(m, 2 * sc.sorted.length)];
        final int[] sorted = sc.sorted, cnt = sc.sizeCnt, ndSize = g.ndSize;
        final int zV = ct.size[V];
        java.util.Arrays.fill(cnt, 0, zV + 2, 0);
        for (int j = 0; j < m; j++) cnt[ndSize[ins.a[j]] + 1]++;
        for (int z = 0; z <= zV; z++) cnt[z + 1] += cnt[z];
        for (int j = 0; j < m; j++) sorted[cnt[ndSize[ins.a[j]]]++] = ins.a[j];
        final long[] io = sc.io;
        final int[] ndA = g.ndA, ndB = g.ndB, nyStart = g.nyStart, nyA = g.nyA, nyB = g.nyB;
        final long[] nyF = g.nyF;
        final int[] taxaV = ct.bits == null ? ct.taxa(V) : null;
        final int sp = 2 * NC;                                   // io offset of species 0
        for (int base = 0; base < cands; base += 64) {
            final int cnt64 = Math.min(64, cands - base);
            final long full = cnt64 == 64 ? -1L : (1L << cnt64) - 1;
            // In of species: candidates whose side S (the smaller side: fewer species to mark) contains the species
            if (taxaV != null) { for (int x : taxaV) io[sp + 2 * x] = 0; } else forBits(V, x -> io[sp + 2 * x] = 0);
            for (int j = 0; j < cnt64; j++) {
                int A = cA[c0 + base + j], B = cB[c0 + base + j];
                int S = ct.size[A] <= ct.size[B] ? A : B;
                long bit = 1L << j;
                if (ct.bits != null) {
                    final int BW = ct.BW; final long[] bits = ct.bits;
                    for (int w = 0, o = BW * S; w < BW; w++) {
                        long b = bits[o + w];
                        while (b != 0) { io[sp + 2 * ((w << 6) + Long.numberOfTrailingZeros(b))] |= bit; b &= b - 1; }
                    }
                } else for (int x : ct.taxa(S)) io[sp + 2 * x] |= bit;
            }
            if (taxaV != null) { for (int x : taxaV) io[sp + 2 * x + 1] = ~io[sp + 2 * x] & full; }
            else forBits(V, x -> io[sp + 2 * x + 1] = ~io[sp + 2 * x] & full);
            if (PROF) { sc.wc[7]++; sc.wc[3] += m; }
            long strad = 0; final long[] occ = g.ndOcc;
            for (int j = 0; j < m; j++) {
                final int P = sorted[j];
                final int a = 2 * ndA[P], b = 2 * ndB[P];
                final long inP = io[a] & io[b], outP = io[a + 1] & io[b + 1];
                io[2 * P] = inP; io[2 * P + 1] = outP;
                // only candidates that split P can dominate an STB whose parent cluster is P
                if ((inP | outP) == full) continue;
                if (dl && base == 0 && ((inP | outP) & 1L) == 0) strad += occ[P];     // I(V) recurrence (MGDL only)
                if (PROF) { sc.wc[4]++; sc.wc[5] += nyStart[P + 1] - nyStart[P]; }
                for (int e = nyStart[P], eEnd = nyStart[P + 1]; e < eEnd; e++) {
                    final int ca = 2 * nyA[e], cb = 2 * nyB[e];
                    long h = (io[ca] & io[cb + 1]) | (io[ca + 1] & io[cb]);
                    if (h == 0) continue;
                    final long f = nyF[e];
                    do { Wb[base + Long.numberOfTrailingZeros(h)] += f; h &= h - 1; } while (h != 0);
                }
            }
            if (base == 0) sc.straddle = strad;
        }
        if (cands == 0) sc.straddle = 0;
        // leave the species entries of the batched layout (offset 3 NC) zero: they may overlap this layout's (offset 2 NC)
        if (taxaV != null) { for (int x : taxaV) { io[sp + 2 * x] = 0; io[sp + 2 * x + 1] = 0; } }
        else forBits(V, x -> { io[sp + 2 * x] = 0; io[sp + 2 * x + 1] = 0; });
    }

    /**
     * Weights of 64 candidates (flat indices j0..j1-1, possibly of different states, all of size z) in one pass over the
     * node clusters of size <= z. Per candidate lane: In(K) = [K inside S], Out(K) = [K inside V - S], InV(K) = [K inside V].
     */
    private void batchedWeights(int z, int[] flatC, int[] flatV, boolean[] first, int j0, int j1, int[] cA, int[] cB, long[] out, int tot, Scratch sc) {
        final int n = g.n, NC = g.nodeCount, lanes = j1 - j0;
        long F = 0; if (dl) for (int j = 0; j < lanes; j++) if (first[j0 + j]) F |= 1L << j;   // straddle lanes (MGDL only)
        final long[] occ = g.ndOcc;
        if (sc.io == null) sc.io = new long[3 * (NC + n)];
        final long[] io = sc.io;                                // species entries are all zero on entry and on exit
        final int sp = 3 * NC;
        for (int j = 0; j < lanes; j++) {
            final int c = flatC[j0 + j], Vc = flatV[j0 + j], A = cA[c], B = cB[c];
            final int S = ct.size[A] <= ct.size[B] ? A : B;
            final long bit = 1L << j;
            if (ct.bits != null) {
                final int BW = ct.BW; final long[] bits = ct.bits;
                for (int w = 0, o = BW * Vc; w < BW; w++) { long b = bits[o + w]; while (b != 0) { io[sp + 3 * ((w << 6) + Long.numberOfTrailingZeros(b)) + 2] |= bit; b &= b - 1; } }
                for (int w = 0, o = BW * S; w < BW; w++) { long b = bits[o + w]; while (b != 0) { io[sp + 3 * ((w << 6) + Long.numberOfTrailingZeros(b))] |= bit; b &= b - 1; } }
            } else {
                for (int x : ct.taxa(Vc)) io[sp + 3 * x + 2] |= bit;
                for (int x : ct.taxa(S)) io[sp + 3 * x] |= bit;
            }
        }
        for (int x = 0, o = sp; x < n; x++, o += 3) io[o + 1] = io[o + 2] & ~io[o];
        final int qEnd = g.prefixEnd[z];
        final int[] ndA = g.ndA, ndB = g.ndB, nyStart = g.nyStart, nyA = g.nyA, nyB = g.nyB;
        final long[] nyF = g.nyF;
        long tested = 0, split = 0;
        for (int q = 0, o = 0; q < qEnd; q++, o += 3) {
            final int a = 3 * ndA[q], b = 3 * ndB[q];
            final long inv = io[a + 2] & io[b + 2];
            if (inv == 0) { io[o] = 0; io[o + 1] = 0; io[o + 2] = 0; continue; }
            final long inP = io[a] & io[b], outP = io[a + 1] & io[b + 1];
            io[o] = inP; io[o + 1] = outP; io[o + 2] = inv;
            final long spl = inv & ~(inP | outP);
            if (spl == 0) continue;
            long sf = spl & F;
            while (sf != 0) { out[tot + j0 + Long.numberOfTrailingZeros(sf)] += occ[q]; sf &= sf - 1; }
            if (PROF) { split++; tested += nyStart[q + 1] - nyStart[q]; }
            for (int e = nyStart[q], eEnd = nyStart[q + 1]; e < eEnd; e++) {
                final int ca = 3 * nyA[e], cb = 3 * nyB[e];
                long h = (io[ca] & io[cb + 1]) | (io[ca + 1] & io[cb]);
                if (h == 0) continue;
                final long f = nyF[e];
                do { out[j0 + Long.numberOfTrailingZeros(h)] += f; h &= h - 1; } while (h != 0);
            }
        }
        for (int x = 0, o = sp; x < n; x++, o += 3) { io[o] = 0; io[o + 1] = 0; io[o + 2] = 0; }
        if (PROF) { sc.wc[0]++; sc.wc[2] += qEnd; sc.wc[4] += split; sc.wc[5] += tested; sc.wc[7]++; }
    }

    /** 0 = automatic, 1 = always 128 candidates per shared pass, 2 = always 64 */
    static final int WIDE_MODE = Integer.getInteger("dynadupx.wide", 0);
    private boolean wide;

    /** batchedWeights with 128 lanes: per code 6 words (In0, In1, Out0, Out1, InV0, InV1). One pass serves twice as many
     *  candidates while the scattered reads (the parts and sides of each gene cluster) still touch one cache line each. */
    private void batchedWeights2(int z, int[] flatC, int[] flatV, boolean[] first, int j0, int j1, int[] cA, int[] cB, long[] out, int tot, Scratch sc) {
        final int n = g.n, NC = g.nodeCount, lanes = j1 - j0;
        long F0 = 0, F1 = 0;
        if (dl) for (int j = 0; j < lanes; j++) if (first[j0 + j]) { if (j < 64) F0 |= 1L << j; else F1 |= 1L << (j - 64); }
        final long[] occ = g.ndOcc;
        if (sc.io == null || sc.io.length < 6 * (NC + n)) sc.io = new long[6 * (NC + n)];
        final long[] io = sc.io;
        final int sp = 6 * NC;
        for (int j = 0; j < lanes; j++) {
            final int c = flatC[j0 + j], Vc = flatV[j0 + j], A = cA[c], B = cB[c];
            final int S = ct.size[A] <= ct.size[B] ? A : B;
            final int wd = j >>> 6; final long bit = 1L << (j & 63);
            if (ct.bits != null) {
                final int BW = ct.BW; final long[] bits = ct.bits;
                for (int w = 0, o = BW * Vc; w < BW; w++) { long b = bits[o + w]; while (b != 0) { io[sp + 6 * ((w << 6) + Long.numberOfTrailingZeros(b)) + 4 + wd] |= bit; b &= b - 1; } }
                for (int w = 0, o = BW * S; w < BW; w++) { long b = bits[o + w]; while (b != 0) { io[sp + 6 * ((w << 6) + Long.numberOfTrailingZeros(b)) + wd] |= bit; b &= b - 1; } }
            } else {
                for (int x : ct.taxa(Vc)) io[sp + 6 * x + 4 + wd] |= bit;
                for (int x : ct.taxa(S)) io[sp + 6 * x + wd] |= bit;
            }
        }
        for (int x = 0, o = sp; x < n; x++, o += 6) { io[o + 2] = io[o + 4] & ~io[o]; io[o + 3] = io[o + 5] & ~io[o + 1]; }
        final int qEnd = g.prefixEnd[z];
        final int[] ndA = g.ndA, ndB = g.ndB, nyStart = g.nyStart, nyA = g.nyA, nyB = g.nyB;
        final long[] nyF = g.nyF;
        long tested = 0, split = 0;
        for (int q = 0, o = 0; q < qEnd; q++, o += 6) {
            final int a = 6 * ndA[q], b = 6 * ndB[q];
            final long v0 = io[a + 4] & io[b + 4], v1 = io[a + 5] & io[b + 5];
            if ((v0 | v1) == 0) { io[o] = 0; io[o + 1] = 0; io[o + 2] = 0; io[o + 3] = 0; io[o + 4] = 0; io[o + 5] = 0; continue; }
            final long i0 = io[a] & io[b], i1 = io[a + 1] & io[b + 1], u0 = io[a + 2] & io[b + 2], u1 = io[a + 3] & io[b + 3];
            io[o] = i0; io[o + 1] = i1; io[o + 2] = u0; io[o + 3] = u1; io[o + 4] = v0; io[o + 5] = v1;
            final long s0 = v0 & ~(i0 | u0), s1 = v1 & ~(i1 | u1);
            if ((s0 | s1) == 0) continue;
            long sf = s0 & F0;
            while (sf != 0) { out[tot + j0 + Long.numberOfTrailingZeros(sf)] += occ[q]; sf &= sf - 1; }
            sf = s1 & F1;
            while (sf != 0) { out[tot + j0 + 64 + Long.numberOfTrailingZeros(sf)] += occ[q]; sf &= sf - 1; }
            if (PROF) { split++; tested += nyStart[q + 1] - nyStart[q]; }
            for (int e = nyStart[q], eEnd = nyStart[q + 1]; e < eEnd; e++) {
                final int ca = 6 * nyA[e], cb = 6 * nyB[e];
                long h0 = (io[ca] & io[cb + 2]) | (io[ca + 2] & io[cb]), h1 = (io[ca + 1] & io[cb + 3]) | (io[ca + 3] & io[cb + 1]);
                if ((h0 | h1) == 0) continue;
                final long f = nyF[e];
                while (h0 != 0) { out[j0 + Long.numberOfTrailingZeros(h0)] += f; h0 &= h0 - 1; }
                while (h1 != 0) { out[j0 + 64 + Long.numberOfTrailingZeros(h1)] += f; h1 &= h1 - 1; }
            }
        }
        for (int x = 0, o = sp; x < n; x++, o += 6) { io[o] = 0; io[o + 1] = 0; io[o + 2] = 0; io[o + 3] = 0; io[o + 4] = 0; io[o + 5] = 0; }
        if (PROF) { sc.wc[0]++; sc.wc[2] += qEnd; sc.wc[4] += split; sc.wc[5] += tested; sc.wc[7] += 2; }
    }

    // ---- sparse shared passes --------------------------------------------------------------------------------------
    static final boolean SPARSE_PASS = !"false".equals(System.getProperty("dynadupx.sparsePass"));
    static final int DIRECT_MODE = Integer.getInteger("dynadupx.directRows", 0);   // 0 = automatic, 1 = always, 2 = never
    /** a pass is sparse if the clusters filed under its states' species (upper bound) are at most this share of all node
     *  clusters of size <= z (0 = always a full pass, large = always sparse) */
    static final double SPARSE_RATIO = Double.parseDouble(System.getProperty("dynadupx.sparseRatio", "1e9"));
    /** passes of states larger than this share of all species are full passes */
    static final double SPARSE_MAXZ = Double.parseDouble(System.getProperty("dynadupx.sparseMaxZ", "0.5"));
    /** candidates per chunk of a DP layer (the per-layer arrays take about 41 bytes per candidate) */
    static final long LAYER_CHUNK = Long.getLong("dynadupx.layerChunk", 1L << 22);
    static final boolean LAYER_ORDER = !"false".equals(System.getProperty("dynadupx.layerOrder"));
    static final int LAYER_ORDER_MAXC = Integer.getInteger("dynadupx.layerOrderMaxC", 8);
    private boolean directRows;
    /**
     * Weights of up to 128 candidates (possibly of different states, all of size z) in one pass that visits only the node
     * clusters whose two rarest species both lie in the state of some lane (pair index), in q order (= size order, so the
     * parts of a cluster come first). A cluster is inside a lane's V only if both of its parts are, so a cluster whose
     * part was not visited, or is inside no lane's V, is skipped. Visited clusters that are inside some lane's V get
     * consecutive rows of a compact buffer (rows 0..n-1 are the species). Same sums as batchedWeights2.
     */
    private void sparseWeights(int z, int[] flatC, int[] flatV, boolean[] first, int j0, int j1, int[] cA, int[] cB, long[] out, int tot, Scratch sc) {
        final int n = g.n, NC = g.nodeCount, lanes = j1 - j0, BW = ct.BW;
        final long[] bits = ct.bits;
        long F0 = 0, F1 = 0;
        if (dl) for (int j = 0; j < lanes; j++) if (first[j0 + j]) { if (j < 64) F0 |= 1L << j; else F1 |= 1L << (j - 64); }
        // no cluster -> row map (it would take 4 bytes per gene cluster and thread): the row of a visited cluster is its
        // rank in the visit mask (rank of its mask word + bits below it in the word)
        if (sc.qMask == null) {
            sc.qMask = new long[(NC >>> 6) + 1]; sc.qSum = new long[(NC >>> 12) + 1]; sc.uBits = new long[BW]; sc.wRank = new int[(NC >>> 6) + 1];
        }
        nextPass(sc);
        if (sc.lio == null) { sc.lio = new long[6 * (n + 1)]; sc.kNv = 0; }
        final long[] kio = sc.lio;                              // kept buffer: species rows 0..n-1 are written here
        final long[] mask = sc.qMask, sum = sc.qSum, U = sc.uBits; final int[] wRank = sc.wRank;
        final int qEnd = g.prefixEnd[z];
        int prevV = -1;
        final long tg0 = PROF ? System.nanoTime() : 0;
        for (int j = 0; j < lanes; j++) {
            final int c = flatC[j0 + j], Vc = flatV[j0 + j], A = cA[c], B = cB[c];
            final int S = ct.size[A] <= ct.size[B] ? A : B;
            final int wd = j >>> 6; final long bit = 1L << (j & 63);
            final int oV = BW * Vc;
            for (int w = 0; w < BW; w++) { long b = bits[oV + w]; U[w] |= b; while (b != 0) { kio[6 * ((w << 6) + Long.numberOfTrailingZeros(b)) + 4 + wd] |= bit; b &= b - 1; } }
            for (int w = 0, o = BW * S; w < BW; w++) { long b = bits[o + w]; while (b != 0) { kio[6 * ((w << 6) + Long.numberOfTrailingZeros(b)) + wd] |= bit; b &= b - 1; } }
            if (Vc == prevV) continue;
            prevV = Vc;                                         // gather the clusters filed under two species of V
            for (int w = 0; w < BW; w++) {
                long x = bits[oV + w];
                while (x != 0) {
                    final int a = (w << 6) + Long.numberOfTrailingZeros(x); x &= x - 1;
                    if (PROF) sc.wc[6] += g.prStart[a + 1] - g.prStart[a];
                    gatherPairs(a, oV, qEnd, mask, sum, sc);
                }
            }
        }
        for (int w = 0; w < BW; w++) { long b = U[w]; while (b != 0) { final int o = 6 * ((w << 6) + Long.numberOfTrailingZeros(b)); b &= b - 1;
            kio[o + 2] = kio[o + 4] & ~kio[o]; kio[o + 3] = kio[o + 5] & ~kio[o + 1]; } }
        final int[] rec = g.ndRec, ny = g.nyPack;
        final long[] occ = g.ndOcc;
        final long tg1 = PROF ? System.nanoTime() : 0;
        // (A) number of visited clusters, then their rows: kept per-thread buffers if they are small enough, otherwise
        // buffers of this pass only, reserved from the heap budget (at most a budgeted amount is in use at any time)
        final int sEnd = Math.min(sum.length, (qEnd + 4095) >>> 12);
        int nv = 0;
        for (int s = 0; s < sEnd; s++) { long sw = sum[s]; while (sw != 0) { nv += Long.bitCount(mask[(s << 6) + Long.numberOfTrailingZeros(sw)]); sw &= sw - 1; } }
        final int Z = n + nv;
        long held = 0;
        long[] io; int[] vq, ra, rb, sp;
        if (nv <= sc.kNv) { io = kio; vq = sc.kvq; ra = sc.ra; rb = sc.rb; sp = sc.ksp; }
        else if (nv <= keepNv) {
            final int cap = (int) Math.min(Math.min(keepNv, (long) NC), Math.max((long) nv, 2L * sc.kNv));
            io = sc.lio = java.util.Arrays.copyOf(kio, 6 * (n + cap + 1));
            vq = sc.kvq = new int[cap]; ra = sc.ra = new int[cap]; rb = sc.rb = new int[cap]; sp = sc.ksp = new int[cap]; sc.kNv = cap;
        } else {
            held = rowBytes(n, nv);
            rowBudget.acquire(held);
            io = java.util.Arrays.copyOf(kio, 6 * (Z + 1));
            vq = new int[nv]; ra = new int[nv]; rb = new int[nv]; sp = new int[nv];
            for (int w = 0; w < BW; w++) { long b = U[w]; while (b != 0) { final int o = 6 * ((w << 6) + Long.numberOfTrailingZeros(b)); b &= b - 1;
                kio[o] = 0; kio[o + 1] = 0; kio[o + 2] = 0; kio[o + 3] = 0; kio[o + 4] = 0; kio[o + 5] = 0; } }
            if (PROF) sc.nBig++;
        }
        try {
            int cnt = 0;
            for (int s = 0; s < sEnd; s++) {
                long sw = sum[s];
                if (sw == 0) continue;
                sum[s] = 0;
                while (sw != 0) {
                    final int w = (s << 6) + Long.numberOfTrailingZeros(sw); sw &= sw - 1;
                    long m = mask[w]; wRank[w] = cnt;
                    while (m != 0) { vq[cnt++] = (w << 6) + Long.numberOfTrailingZeros(m); m &= m - 1; }
                }
            }
            final long tA = PROF ? System.nanoTime() : 0;
            for (int j = 6 * Z; j < 6 * Z + 6; j++) io[j] = 0;
            // (B) rows of the two parts (a part that was not visited reads the zero row Z)
            for (int i = 0; i < nv; i++) {
                final int rq = 4 * vq[i], ca = rec[rq], cb = rec[rq + 1];
                ra[i] = ca >= NC ? ca - NC : rowOf(ca, mask, wRank, n, Z);
                rb[i] = cb >= NC ? cb - NC : rowOf(cb, mask, wRank, n, Z);
            }
            final long tB = PROF ? System.nanoTime() : 0;
            // (C) In / Out / InV of every visited cluster (a cluster with a part inside no lane's V gets a zero row)
            int ns = 0;
            long inl = 0;
            for (int i = 0; i < nv; i++) {
                final int a = 6 * ra[i], b = 6 * rb[i], o = 6 * (n + i);
                final long v0 = io[a + 4] & io[b + 4], v1 = io[a + 5] & io[b + 5];
                final long i0 = io[a] & io[b], i1 = io[a + 1] & io[b + 1], u0 = io[a + 2] & io[b + 2], u1 = io[a + 3] & io[b + 3];
                io[o] = i0; io[o + 1] = i1; io[o + 2] = u0; io[o + 3] = u1; io[o + 4] = v0; io[o + 5] = v1;
                if (PROF && (v0 | v1) != 0) inl++;
                if (((v0 & ~(i0 | u0)) | (v1 & ~(i1 | u1))) != 0) sp[ns++] = i;
            }
            final long tC = PROF ? System.nanoTime() : 0;
            // (D) valid STBs of the clusters split by some lane
            long tested = 0, hits = 0, ysum = 0, f1 = 0;
            for (int t = 0; t < ns; t++) {
                final int i = sp[t], q = vq[i], o = 6 * (n + i);
                final long s0 = io[o + 4] & ~(io[o] | io[o + 2]), s1 = io[o + 5] & ~(io[o + 1] | io[o + 3]);
                long sf = s0 & F0;
                while (sf != 0) { out[tot + j0 + Long.numberOfTrailingZeros(sf)] += occ[q]; sf &= sf - 1; }
                sf = s1 & F1;
                while (sf != 0) { out[tot + j0 + 64 + Long.numberOfTrailingZeros(sf)] += occ[q]; sf &= sf - 1; }
                final int rq = 4 * q, e0 = rec[rq + 2], e1 = rec[rq + 6], df = rec[rq + 3];
                if (df != 0) {
                    final int pa = 6 * ra[i], pb = 6 * rb[i];
                    long h0 = (io[pa] & io[pb + 2]) | (io[pa + 2] & io[pb]), h1 = (io[pa + 1] & io[pb + 3]) | (io[pa + 3] & io[pb + 1]);
                    while (h0 != 0) { out[j0 + Long.numberOfTrailingZeros(h0)] += df; h0 &= h0 - 1; }
                    while (h1 != 0) { out[j0 + 64 + Long.numberOfTrailingZeros(h1)] += df; h1 &= h1 - 1; }
                }
                if (PROF) tested += (e1 - e0) / 3;
                for (int e = e0; e < e1; e += 3) {
                    final int ya = ny[e], yb = ny[e + 1];
                    final int x = ya >= NC ? ya - NC : rowOf(ya, mask, wRank, n, Z), y = yb >= NC ? yb - NC : rowOf(yb, mask, wRank, n, Z);
                    final int pa = 6 * x, pb = 6 * y;
                    long h0 = (io[pa] & io[pb + 2]) | (io[pa + 2] & io[pb]), h1 = (io[pa + 1] & io[pb + 3]) | (io[pa + 3] & io[pb + 1]);
                    if ((h0 | h1) == 0) continue;
                    final long f = ny[e + 2];
                    if (PROF) { hits += Long.bitCount(h0) + Long.bitCount(h1); ysum++; if (f == 1) f1++; }
                    while (h0 != 0) { out[j0 + Long.numberOfTrailingZeros(h0)] += f; h0 &= h0 - 1; }
                    while (h1 != 0) { out[j0 + 64 + Long.numberOfTrailingZeros(h1)] += f; h1 &= h1 - 1; }
                }
            }
            for (int i = 0; i < nv; i++) mask[vq[i] >>> 6] = 0;
            if (io == sc.lio) for (int w = 0; w < BW; w++) { long b = U[w]; while (b != 0) { final int o = 6 * ((w << 6) + Long.numberOfTrailingZeros(b)); b &= b - 1;
                io[o] = 0; io[o + 1] = 0; io[o + 2] = 0; io[o + 3] = 0; io[o + 4] = 0; io[o + 5] = 0; } }
            for (int w = 0; w < BW; w++) U[w] = 0;
            final long seen = nv, split = ns; final int L = n + (int) inl;
            if (PROF) { sc.wc[0]++; sc.wc[2] += seen; sc.wc[3] += L - n; sc.wc[4] += split; sc.wc[5] += tested; sc.wc[7] += 2; sc.tGat += tg1 - tg0; sc.tPro += System.nanoTime() - tg1; sc.hits += hits; sc.yHit += ysum; sc.f1 += f1; sc.tph[0] += tA - tg1; sc.tph[1] += tB - tA; sc.tph[2] += tC - tB; sc.tph[3] += System.nanoTime() - tC; }
        } finally {
            if (held > 0) rowBudget.release(held);
        }
    }

    /** row of node cluster q in a compact pass: n + its rank among the visited clusters, or the zero row Z */
    private static int rowOf(int q, long[] mask, int[] wRank, int n, int Z) {
        final long m = mask[q >>> 6], bit = 1L << q;
        return (m & bit) == 0 ? Z : n + wRank[q >>> 6] + Long.bitCount(m & (bit - 1));
    }
    /** bytes of the compact buffers of a pass with nv visited clusters: 6 words per row, 4 ints per visited cluster */
    static long rowBytes(int n, long nv) { return 48L * (n + nv + 1) + 16L * nv; }

    /** Heap budget for the compact row buffers of the large shared passes (a byte semaphore). A pass that needs more
     *  than the whole budget runs alone. */
    static final class RowBudget {
        final long cap; private long used; long peak, waits;
        RowBudget(long cap) { this.cap = cap; }
        synchronized void acquire(long b) {
            boolean waited = false;
            while (used > 0 && used + b > cap) { waited = true; try { wait(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new IllegalStateException(e); } }
            if (waited) waits++;
            used += b; if (used > peak) peak = used;
        }
        synchronized void release(long b) { used -= b; notifyAll(); }
    }
    private RowBudget rowBudget;
    /** free heap (maximum heap minus live data after a collection) */
    static long freeHeap() {
        Runtime rt = Runtime.getRuntime();
        System.gc();
        return rt.maxMemory() - (rt.totalMemory() - rt.freeMemory());
    }
    private long keepNv;                     // a thread keeps compact buffers for up to this many visited clusters

    /** new pass: pair stamps (a pair's runs are marked once per pass, however many of its states contain the pair) */
    private void nextPass(Scratch sc) {
        if (sc.pairStamp == null || sc.pairStamp.length < g.prB.length) { sc.pairStamp = new int[g.prB.length]; sc.passStamp = 0; }
        if (++sc.passStamp == Integer.MAX_VALUE) { java.util.Arrays.fill(sc.pairStamp, 0); sc.passStamp = 1; }
    }
    /** Marks (two-level mask) the node clusters of size <= z (q < qEnd) filed under (a, b) for the species b of the state
     *  whose bitset starts at bits[oV]: consecutive runs of q are set a word at a time. */
    private void gatherPairs(int a, int oV, int qEnd, long[] mask, long[] sum, Scratch sc) {
        final long[] bits = ct.bits;
        final int[] prStart = g.prStart, prB = g.prB, rst = g.prRStart, rs = g.prRS, re = g.prRE;
        final int[] pst = sc.pairStamp; final int cur = sc.passStamp;
        for (int p = prStart[a], pe = prStart[a + 1]; p < pe; p++) {
            final int b = prB[p];
            if ((bits[oV + (b >>> 6)] & (1L << b)) == 0) continue;
            if (pst[p] == cur) continue;                           // pair already gathered in this pass
            pst[p] = cur;
            for (int r = rst[p], rEnd = rst[p + 1]; r < rEnd; r++) {
                final int s0 = rs[r];
                if (s0 >= qEnd) break;
                final int e0 = Math.min(re[r], qEnd);
                if (PROF) sc.wc[1] += e0 - s0;
                final int w0 = s0 >>> 6, w1 = (e0 - 1) >>> 6;
                if (w0 == w1) mask[w0] |= (-1L << s0) & (-1L >>> (63 - ((e0 - 1) & 63)));
                else {
                    mask[w0] |= -1L << s0;
                    for (int w = w0 + 1; w < w1; w++) mask[w] = -1L;
                    mask[w1] |= -1L >>> (63 - ((e0 - 1) & 63));
                }
                for (int w = w0; w <= w1; w++) sum[w >>> 6] |= 1L << w;
            }
        }
    }

    /**
     * sparseWeights with rows indexed by code (as batchedWeights2: 6 words per node cluster and species, 48 bytes per
     * gene cluster and thread): no row map. Rows are zero outside a pass; a pass writes the rows of the visited clusters
     * that are inside some lane's V and clears them at the end, so a part that was not visited reads as zero.
     */
    private void sparseWeightsDirect(int z, int[] flatC, int[] flatV, boolean[] first, int j0, int j1, int[] cA, int[] cB, long[] out, int tot, Scratch sc) {
        final int n = g.n, NC = g.nodeCount, lanes = j1 - j0, BW = ct.BW;
        final long[] bits = ct.bits;
        long F0 = 0, F1 = 0;
        if (dl) for (int j = 0; j < lanes; j++) if (first[j0 + j]) { if (j < 64) F0 |= 1L << j; else F1 |= 1L << (j - 64); }
        if (sc.io == null || sc.io.length < 6 * (NC + n)) sc.io = new long[6 * (NC + n)];
        if (sc.qMask == null) { sc.qMask = new long[(NC >>> 6) + 1]; sc.qSum = new long[(NC >>> 12) + 1]; sc.uBits = new long[BW]; }
        nextPass(sc);
        final long[] io = sc.io, mask = sc.qMask, sum = sc.qSum, U = sc.uBits;
        final int sp = 6 * NC, qEnd = g.prefixEnd[z];
        int prevV = -1;
        final long tg0 = PROF ? System.nanoTime() : 0;
        for (int j = 0; j < lanes; j++) {
            final int c = flatC[j0 + j], Vc = flatV[j0 + j], A = cA[c], B = cB[c];
            final int S = ct.size[A] <= ct.size[B] ? A : B;
            final int wd = j >>> 6; final long bit = 1L << (j & 63);
            final int oV = BW * Vc;
            for (int w = 0; w < BW; w++) { long b = bits[oV + w]; U[w] |= b; while (b != 0) { io[sp + 6 * ((w << 6) + Long.numberOfTrailingZeros(b)) + 4 + wd] |= bit; b &= b - 1; } }
            for (int w = 0, o = BW * S; w < BW; w++) { long b = bits[o + w]; while (b != 0) { io[sp + 6 * ((w << 6) + Long.numberOfTrailingZeros(b)) + wd] |= bit; b &= b - 1; } }
            if (Vc == prevV) continue;
            prevV = Vc;
            for (int w = 0; w < BW; w++) { long x = bits[oV + w]; while (x != 0) { final int a = (w << 6) + Long.numberOfTrailingZeros(x); x &= x - 1;
                if (PROF) sc.wc[6] += g.prStart[a + 1] - g.prStart[a];
                gatherPairs(a, oV, qEnd, mask, sum, sc); } }
        }
        for (int w = 0; w < BW; w++) { long b = U[w]; while (b != 0) { final int o = sp + 6 * ((w << 6) + Long.numberOfTrailingZeros(b)); b &= b - 1;
            io[o + 2] = io[o + 4] & ~io[o]; io[o + 3] = io[o + 5] & ~io[o + 1]; } }
        final long tg1 = PROF ? System.nanoTime() : 0;
        final int[] rec = g.ndRec, ny = g.nyPack;
        final long[] occ = g.ndOcc;
        final IntList vis = sc.visited; vis.n = 0;
        long seen = 0, split = 0, tested = 0;
        final int sEnd = Math.min(sum.length, (qEnd + 4095) >>> 12);
        for (int s = 0; s < sEnd; s++) {
            long sw = sum[s];
            if (sw == 0) continue;
            sum[s] = 0;
            while (sw != 0) {
                final int w = (s << 6) + Long.numberOfTrailingZeros(sw); sw &= sw - 1;
                long m = mask[w]; mask[w] = 0;
                while (m != 0) {
                    final int q = (w << 6) + Long.numberOfTrailingZeros(m); m &= m - 1;
                    if (PROF) seen++;
                    final int rq = 4 * q, a = 6 * rec[rq], b = 6 * rec[rq + 1];
                    final long v0 = io[a + 4] & io[b + 4], v1 = io[a + 5] & io[b + 5];
                    if ((v0 | v1) == 0) continue;
                    final int o = 6 * q; vis.add(q);
                    final long ia0 = io[a], ia1 = io[a + 1], ua0 = io[a + 2], ua1 = io[a + 3], ib0 = io[b], ib1 = io[b + 1], ub0 = io[b + 2], ub1 = io[b + 3];
                    final long i0 = ia0 & ib0, i1 = ia1 & ib1, u0 = ua0 & ub0, u1 = ua1 & ub1;
                    io[o] = i0; io[o + 1] = i1; io[o + 2] = u0; io[o + 3] = u1; io[o + 4] = v0; io[o + 5] = v1;
                    final long s0 = v0 & ~(i0 | u0), s1 = v1 & ~(i1 | u1);
                    if ((s0 | s1) == 0) continue;
                    long sf = s0 & F0;
                    while (sf != 0) { out[tot + j0 + Long.numberOfTrailingZeros(sf)] += occ[q]; sf &= sf - 1; }
                    sf = s1 & F1;
                    while (sf != 0) { out[tot + j0 + 64 + Long.numberOfTrailingZeros(sf)] += occ[q]; sf &= sf - 1; }
                    final int df = rec[rq + 3];
                    if (df != 0) {                                   // the STB (part | part): parts already read
                        long h0 = (ia0 & ub0) | (ua0 & ib0), h1 = (ia1 & ub1) | (ua1 & ib1);
                        while (h0 != 0) { out[j0 + Long.numberOfTrailingZeros(h0)] += df; h0 &= h0 - 1; }
                        while (h1 != 0) { out[j0 + 64 + Long.numberOfTrailingZeros(h1)] += df; h1 &= h1 - 1; }
                    }
                    final int e0 = rec[rq + 2], e1 = rec[rq + 6];
                    if (PROF) { split++; tested += (e1 - e0) / 3; }
                    for (int e = e0; e < e1; e += 3) {
                        final int pa = 6 * ny[e], pb = 6 * ny[e + 1];
                        long h0 = (io[pa] & io[pb + 2]) | (io[pa + 2] & io[pb]), h1 = (io[pa + 1] & io[pb + 3]) | (io[pa + 3] & io[pb + 1]);
                        if ((h0 | h1) == 0) continue;
                        final long f = ny[e + 2];
                        while (h0 != 0) { out[j0 + Long.numberOfTrailingZeros(h0)] += f; h0 &= h0 - 1; }
                        while (h1 != 0) { out[j0 + 64 + Long.numberOfTrailingZeros(h1)] += f; h1 &= h1 - 1; }
                    }
                }
            }
        }
        final int[] vq = vis.a;
        for (int i = 0, m = vis.n; i < m; i++) { final int o = 6 * vq[i]; io[o] = 0; io[o + 1] = 0; io[o + 2] = 0; io[o + 3] = 0; io[o + 4] = 0; io[o + 5] = 0; }
        for (int w = 0; w < BW; w++) { long b = U[w]; U[w] = 0; while (b != 0) { final int o = sp + 6 * ((w << 6) + Long.numberOfTrailingZeros(b)); b &= b - 1;
            io[o] = 0; io[o + 1] = 0; io[o + 2] = 0; io[o + 3] = 0; io[o + 4] = 0; io[o + 5] = 0; } }
        if (PROF) { sc.wc[0]++; sc.wc[2] += seen; sc.wc[3] += vis.n; sc.wc[4] += split; sc.wc[5] += tested; sc.wc[7] += 2; sc.tGat += tg1 - tg0; sc.tPro += System.nanoTime() - tg1; }
    }

    private interface IntOp { void run(int x); }
    private void forBits(int V, IntOp op) {
        final int BW = ct.BW; final long[] bits = ct.bits;
        for (int w = 0, o = BW * V; w < BW; w++) {
            long b = bits[o + w];
            while (b != 0) { op.run((w << 6) + Long.numberOfTrailingZeros(b)); b &= b - 1; }
        }
    }


    private double val(int cluster, double[] Vv) {
        int s = ss.state(cluster);
        return s >= 0 ? Vv[s] : singleVal[cluster];
    }

    private long candidateWeight(int c, int A, int B, int stampV, Scratch sc) {
        if (gpuW != null && !verifyGpu) return gpuW[c];
        long w = weight(A, B, stampV, sc);
        if (gpuW != null && gpuW[c] != w)
            throw new IllegalStateException("GPU/CPU weight mismatch at candidate " + c + ": gpu=" + gpuW[c] + " cpu=" + w);
        return w;
    }

    /** {|T_P u T_Q|, |T_P n T_Q|, SSM} for candidate (A|B) of the state whose comps row is in sc. */
    private long[] terms(int A, int B, Scratch sc) {
        long[] out = new long[3];
        if (g.allComplete) { out[0] = g.k; out[1] = g.k; return out; }
        long tv = sc.present;
        out[0] = tv;
        out[1] = (long) presCount[A] + presCount[B] - tv;
        out[2] = absentXL(A, sc) + absentXL(B, sc);
        return out;
    }

    /** sum over trees where cluster Z is absent of XL(t, V) = max(comps_t(V) - 1, 0). */
    private long absentXL(int Z, Scratch sc) {
        Object a = absent[Z];
        if (a == null) a = presenceOf(Z);
        else if (a == VIRT) a = virtualPresence(Z, sc, 2);
        long s = 0; int[] comps = sc.comps;
        if (a instanceof int[] list) {
            for (int t : list) { int c = comps[t]; if (c > 1) s += c - 1; }
        } else {
            long[] bits = (long[]) a;
            for (int j = 0; j < bits.length; j++) {
                long abs = ~bits[j];
                if (j == bits.length - 1 && (g.k & 63) != 0) abs &= (1L << (g.k & 63)) - 1;
                while (abs != 0) { int t = (j << 6) + Long.numberOfTrailingZeros(abs); abs &= abs - 1; int c = comps[t]; if (c > 1) s += c - 1; }
            }
        }
        return s;
    }

    /** Absent-tree set of Z (same representation as storePresence) for a cluster whose stored set was already freed or
     *  never computed (only used by the events report, after the DP). */
    private Object presenceOf(int Z) {
        if (presCount[Z] < 0) throw new IllegalStateException("presence of cluster " + Z + " requested before it was computed");
        if (evPres == null) evPres = new Scratch(g.n, g.k);
        Scratch t = evPres;
        computeComps(Z, t);
        int k = g.k, nAbs = k - t.present;
        Object rep;
        if (nAbs == 0) rep = NONE;
        else if (nAbs <= Math.max(1, k >>> 5)) { int[] list = new int[nAbs]; int i = 0; for (int x = 0; x < k; x++) if (t.comps[x] == 0) list[i++] = x; rep = list; }
        else {
            long[] bits = new long[words];
            for (int j = 0; j < t.touched.n; j++) { int x = t.touched.a[j]; bits[x >>> 6] |= 1L << (x & 63); }
            rep = bits;
        }
        clearComps(t);
        return rep;
    }
    /** Scratch space of the events report (after the DP; single-threaded), reused across calls. */
    private Scratch evSc, evPres;

    /** Stores |T_Z| and the absent-tree set of Z (from the comps row in sc). */
    private void storePresence(int Z, Scratch sc) {
        if (presCount[Z] >= 0) return;
        int k = g.k, present = sc.present, nAbs = k - present;
        Object rep;
        if (nAbs == 0) rep = NONE;
        else if (nAbs <= Math.max(1, k >>> 5)) {             // short list of absent trees (<= k/8 bytes)
            int[] list = new int[nAbs]; int i = 0;
            for (int t = 0; t < k; t++) if (sc.comps[t] == 0) list[i++] = t;
            rep = list;
        } else {                                               // bitset of present trees (k/8 bytes)
            long[] bits = new long[words];
            for (int j = 0; j < sc.touched.n; j++) { int t = sc.touched.a[j]; bits[t >>> 6] |= 1L << (t & 63); }
            rep = bits;
        }
        absent[Z] = Z < g.n ? rep : keep(Z, rep);
        if (Z < g.n && rep != NONE) presBytes.addAndGet(bytesOf(rep));
        presCount[Z] = present;
    }

    // ---- bounded memory for the MGDL presence sets ----------------------------------------------------------------
    /** A cluster whose absent-tree set is not stored: it is recomputed when needed as the union of its species'
     *  present-tree bitsets (exactly the stored set: a tree contains Z iff it contains a species of Z). */
    static final Object VIRT = new Object();
    static final int VIRT_WORDS = Integer.getInteger("dynadupx.virtWords", 256);   // trees / 64 from which tiny clusters are always recomputed
    private final java.util.concurrent.atomic.AtomicLong presBytes = new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.LongAdder nVirt = new java.util.concurrent.atomic.LongAdder(), nVirtUse = new java.util.concurrent.atomic.LongAdder();
    private long presBudget = Long.MAX_VALUE, presPeak;
    private volatile long[][] spPres;                       // species -> present-tree bitset (built when first needed)
    static long bytesOf(Object a) { return a instanceof int[] l ? 16L + 4L * l.length : a instanceof long[] b ? 16L + 8L * b.length : 0L; }
    /** the representation to store for Z: rep itself while the stored sets fit into the budget, else VIRT (small sets
     *  are always kept) */
    private Object keep(int Z, Object rep) {
        if (rep == NONE || rep == null) return rep;
        final long b = bytesOf(rep);
        // many trees: a bitset of a cluster of at most 3 species is not stored at all (its recomputation reads at most 3
        // species bitsets), so the budget is left to larger clusters, whose recomputation costs more
        if (b > 256 && (presBytes.get() + b > presBudget || (words >= VIRT_WORDS && rep instanceof long[] && ct.size[Z] <= 3))) { nVirt.increment(); return VIRT; }
        final long now = presBytes.addAndGet(b);
        if (PROF && now > presPeak) presPeak = now;
        return rep;
    }
    private void unkeep(Object a) { if (a != null && a != VIRT && a != NONE) presBytes.addAndGet(-bytesOf(a)); }
    /** the stored absent-tree set of Z, or for VIRT its present-tree bitset in scratch buffer `slot` of sc */
    private Object absOf(int Z, Scratch sc, int slot) {
        final Object a = absent[Z];
        return a == VIRT ? virtualPresence(Z, sc, slot) : a;
    }
    private Object virtualPresence(int Z, Scratch sc, int slot) {
        long[][] sp = spPres;
        if (sp == null) synchronized (this) {
            sp = spPres;
            if (sp == null) {
                sp = new long[g.n][words];
                for (int v = 0; v < g.n; v++) for (int t : g.occTree[v]) sp[v][t >>> 6] |= 1L << (t & 63);
                spPres = sp;
            }
        }
        if (sc.vbuf == null) sc.vbuf = new long[3][];
        long[] out = sc.vbuf[slot];
        if (out == null) out = sc.vbuf[slot] = new long[words];
        else java.util.Arrays.fill(out, 0L);
        final int W = words;
        if (ct.bits != null) {
            final int BW = ct.BW; final long[] bits = ct.bits;
            for (int w = 0, o = BW * Z; w < BW; w++) { long x = bits[o + w]; while (x != 0) { final long[] p = sp[(w << 6) + Long.numberOfTrailingZeros(x)]; x &= x - 1;
                for (int j = 0; j < W; j++) out[j] |= p[j]; } }
        } else for (int v : ct.taxa(Z)) { final long[] p = sp[v]; for (int j = 0; j < W; j++) out[j] |= p[j]; }
        if (PROF) nVirtUse.increment();
        return out;
    }

    private void loadRow(int[] rows, int off, Scratch sc) {
        long tot = 0; IntList touched = sc.touched; touched.n = 0;
        for (int t = 0; t < g.k; t++) { int c = rows[off + t]; sc.comps[t] = c; if (c > 0) { tot += c; touched.add(t); } }
        sc.totalComps = tot; sc.present = touched.n;
    }

    private void clearComps(Scratch sc) {
        for (int j = 0; j < sc.touched.n; j++) sc.comps[sc.touched.a[j]] = 0;
        sc.touched.n = 0;
    }

    // ------------------------------------------------------------------------------------------------ events
    /** {W_dom, W_xl, B} of candidate c of state si (used once per chosen candidate to report events). */
    long[] components(int si, int c, int[] cA, int[] cB) {
        if (evSc == null) evSc = new Scratch(g.n, g.k);
        Scratch sc = evSc;
        int V = ss.states.a[si], A = cA[c], B = cB[c];
        int stampV = sc.stamp++;
        for (int v : ct.taxa(V)) sc.mark[v] = stampV;
        long w;
        if (gpuW != null) w = gpuW[c];
        else if (g.adjY != null) w = weight(A, B, stampV, sc);
        else { collectInside(V, sc); weightsBPW(V, c, c + 1, cA, cB, sc); w = sc.wbuf[0]; }
        computeComps(V, sc);
        long[] t = terms(A, B, sc);
        long xl = sc.totalComps - t[0] - t[2];
        clearComps(sc); sc.insideFor = -1;
        return new long[]{w, xl, t[2] + t[0] + t[1]};
    }

    long singletonXL(int id) { return singleXL[id]; }

    // ------------------------------------------------------------------------------------------------ weights
    /** MGD domination weight of (A|B); marks of V = A u B carry stampV on entry and on exit. */
    long weight(int A, int B, int stampV, Scratch sc) {
        if (g.W > 0) return weightBits(A, B, sc);
        int Sd = ct.size[A] <= ct.size[B] ? A : B, Od = Sd == A ? B : A;
        int stampS = sc.stamp++;
        int[] taxaS = ct.taxa(Sd);
        for (int v : taxaS) sc.mark[v] = stampS;
        int sizeS = taxaS.length; long w = 0;
        // K inside S:  sig(K) in sig(S)  and  csig(S) in csig(K);   other side inside O = V - S: the same with O
        final long[] SG = ct.sig, CS = ct.csig;
        final int so = 4 * Sd, oo = 4 * Od;
        final long n0 = ~SG[so], n1 = ~SG[so + 1], n2 = ~SG[so + 2], n3 = ~SG[so + 3];
        final long c0 = CS[so], c1 = CS[so + 1], c2 = CS[so + 2], c3 = CS[so + 3];
        final long m0 = ~SG[oo], m1 = ~SG[oo + 1], m2 = ~SG[oo + 2], m3 = ~SG[oo + 3];
        final long d0 = CS[oo], d1 = CS[oo + 1], d2 = CS[oo + 2], d3 = CS[oo + 3];
        for (int v : taxaS) {
            final int[] ids = g.sideBucket[v], zs = g.sideSize[v]; final long[] sg = g.sideSig[v];
            for (int i = 0; i < ids.length; i++) {
                if (zs[i] > sizeS) break;
                int p = 4 * i;
                if (((sg[p] & n0) | (sg[p + 1] & n1) | (sg[p + 2] & n2) | (sg[p + 3] & n3)) != 0) continue;
                int K = ids[i], pk = 4 * K;
                if (((c0 & ~CS[pk]) | (c1 & ~CS[pk + 1]) | (c2 & ~CS[pk + 2]) | (c3 & ~CS[pk + 3])) != 0) continue;
                if (!ct.allMarked(K, sc.mark, stampS)) continue;
                for (int y : g.adjY[K]) {
                    int other = g.yA[y] == K ? g.yB[y] : g.yA[y], q = 4 * other;
                    if (((SG[q] & m0) | (SG[q + 1] & m1) | (SG[q + 2] & m2) | (SG[q + 3] & m3)
                       | (d0 & ~CS[q]) | (d1 & ~CS[q + 1]) | (d2 & ~CS[q + 2]) | (d3 & ~CS[q + 3])) != 0) continue;
                    if (ct.allMarked(other, sc.mark, stampV)) w += g.yF[y];     // inside V and outside S
                }
            }
        }
        for (int v : taxaS) sc.mark[v] = stampV;
        return w;
    }

    /** weight() with exact species bitsets: subset tests are W-word AND-NOTs instead of leaf-range scans. */
    private long weightBits(int A, int B, Scratch sc) {
        int Sd = ct.size[A] <= ct.size[B] ? A : B, Od = Sd == A ? B : A;
        final int W = g.W;
        final long[] bS = sc.bitsS(W), bO = sc.bitsO(W);
        int[] taxaS = ct.taxa(Sd);
        System.arraycopy(ct.bits, W * Sd, bS, 0, W);          // exact species sets from the cluster table
        System.arraycopy(ct.bits, W * Od, bO, 0, W);
        final int sizeS = taxaS.length; long w = 0;
        for (int v : taxaS) {
            final int[] ids = g.sideBucket[v], zs = g.sideSize[v]; final long[] sb = g.sideBits[v];
            scan:
            for (int i = 0, p = 0; i < ids.length; i++, p += W) {
                if (zs[i] > sizeS) break;
                if (PROF) sc.cScan++;
                for (int j = 0; j < W; j++) if ((sb[p + j] & ~bS[j]) != 0) continue scan;
                int K = ids[i];
                if (PROF) { sc.cSub++; sc.cAdj += g.adjY[K].length; }
                final int[] ys = g.adjY[K]; final long[] ab = g.adjBits[K];
                adj:
                for (int e = 0, q = 0; e < ys.length; e++, q += W) {
                    for (int j = 0; j < W; j++) if ((ab[q + j] & ~bO[j]) != 0) continue adj;
                    w += g.yF[ys[e]];
                }
            }
        }
        return w;
    }

    // ------------------------------------------------------------------------------------------------ comps
    /**
     * comps_t(id) for every tree into sc.comps (non-zero entries listed in sc.touched), sc.totalComps = DLC(id),
     * sc.present = |T_id|.
     */
    static final int COMPS_MODE = Integer.getInteger("dynadupx.compsMode", 0);   // 0 = size rule, 1 = always normal, 2 = always complement
    void computeComps(int id, Scratch sc) {
        boolean comp = COMPS_MODE == 0 ? 8 * ct.size[id] > 5 * g.n : COMPS_MODE == 2;   // measured crossover ~ 5/8 of the species
        computeCompsImpl(id, sc, comp);
    }
    private void computeCompsImpl(int id, Scratch sc, boolean comp) {
        if (comp && g.parIntAll != null && !NO_COMPLEMENT_PATH) { computeCompsComplement(id, sc); return; }
        int[] taxa = ct.taxa(id);
        int st = sc.stamp++;
        IntList touched = sc.touched; touched.n = 0;
        long[] sig = new long[4];
        for (int v : taxa) {
            sc.mark2[v] = st; long h = ct.H1[v]; sig[(int) ((h >>> 6) & 3)] |= 1L << (h & 63);
            int[] tr = g.occTree[v], cn = g.occCnt[v];
            for (int e = 0; e < tr.length; e++) { if (sc.Lk[tr[e]] == 0 && sc.Ik[tr[e]] == 0) touched.add(tr[e]); sc.Lk[tr[e]] += cn[e]; }
        }
        int size = taxa.length;
        if (g.W > 0 && g.prQ != null) {                         // node clusters inside id from the pair index + size-1 ones
            if (sc.insideFor != id) collectInside(id, sc);
            final IntList ins = sc.inside;
            for (int j = 0; j < ins.n; j++) {
                int K = g.ndId[ins.a[j]];
                int[] tr = g.nodeTree[K], cn = g.nodeCnt[K];
                for (int e = 0; e < tr.length; e++) sc.Ik[tr[e]] += cn[e];
            }
            for (int v : taxa) {
                int[] tr = g.nodeTree[v]; if (tr == null) continue;
                int[] cn = g.nodeCnt[v];
                for (int e = 0; e < tr.length; e++) sc.Ik[tr[e]] += cn[e];
            }
        } else {
        final long n0 = ~sig[0], n1 = ~sig[1], n2 = ~sig[2], n3 = ~sig[3];
        final int W = g.W; final long[] bV = W > 0 ? sc.bitsV(W) : null;
        if (W > 0) System.arraycopy(ct.bits, W * id, bV, 0, W);
        for (int v : taxa) {
            final int[] ids = g.nodeBucket[v], zs = g.nodeSize[v]; final long[] sg = g.nodeSig[v];
            final long[] nb = W > 0 ? g.nodeBits[v] : null;
            node:
            for (int i = 0; i < ids.length; i++) {
                if (zs[i] > size) break;
                int K = ids[i];
                if (W > 0) {
                    int p = i * W;
                    for (int j = 0; j < W; j++) if ((nb[p + j] & ~bV[j]) != 0) continue node;
                } else {
                    int p = 4 * i;
                    if (((sg[p] & n0) | (sg[p + 1] & n1) | (sg[p + 2] & n2) | (sg[p + 3] & n3)) != 0) continue;
                    if (!ct.allMarked(K, sc.mark2, st)) continue;
                }
                int[] tr = g.nodeTree[K], cn = g.nodeCnt[K];
                for (int e = 0; e < tr.length; e++) sc.Ik[tr[e]] += cn[e];
            }
        }
        }
        long tot = 0;
        for (int j = 0; j < touched.n; j++) {
            int t = touched.a[j];
            int comps = sc.Lk[t] - sc.Ik[t];
            sc.comps[t] = comps; tot += comps;
            sc.Lk[t] = 0; sc.Ik[t] = 0;
        }
        sc.totalComps = tot; sc.present = touched.n;
    }

    /**
     * Large cluster P: work from Q = species not in P. An internal node is NOT P-pure iff it lies on the
     * root path of some Q-leaf, so I_t(P) = internal_t - |union of Q-leaf root paths| (stamp-marked walk).
     */
    private void computeCompsComplement(int id, Scratch sc) {
        int[] Q;
        if (ct.compOf[id] >= 0) Q = ct.taxa(ct.compOf[id]);
        else {
            int st0 = sc.stamp++;
            for (int v : ct.taxa(id)) sc.mark2[v] = st0;
            IntList q = new IntList();
            for (int v = 0; v < g.n; v++) if (sc.mark2[v] != st0) q.add(v);
            Q = q.toArray();
        }
        compsComplementWalk(Q, sc);
    }

    /** Internal nodes on the root paths of the Q-leaves: a stamp-marked walk (each node counted once). */
    private void compsComplementWalk(int[] Q, Scratch sc) {
        if (sc.nodeBits == null) sc.nodeBits = new long[(g.totalInternal >>> 6) + 1];
        final long[] nb = sc.nodeBits;
        int[] qLeaves = sc.Lk, impure = sc.Ik;           // per-tree scratch (all zero on entry)
        for (int v : Q) {
            int[] tt = g.occPosTree[v], pp = g.occPos[v];
            for (int e = 0; e < tt.length; e++) {
                int t = tt[e];
                qLeaves[t]++;
                int u = g.parLeafAll[g.leafOff[t] + pp[e]], base = g.intOff[t];
                while (u >= 0) { final int x = base + u; final long bit = 1L << x; if ((nb[x >>> 6] & bit) != 0) break; nb[x >>> 6] |= bit; impure[t]++; u = g.parIntAll[x]; }
            }
        }
        // clear the marks: only trees with a Q-leaf have marked nodes (whole words; neighbours are cleared too, harmlessly)
        for (int t = 0; t < g.k; t++) if (impure[t] > 0) { int w0 = g.intOff[t] >>> 6, w1 = (g.intOff[t + 1] - 1) >>> 6; for (int w = w0; w <= w1; w++) nb[w] = 0; }
        finishComplement(qLeaves, impure, sc);
    }
    /** comps_t = (leaves not in Q) - (internal nodes not on a Q root path); resets the per-tree scratch. */
    private void finishComplement(int[] qLeaves, int[] impure, Scratch sc) {
        IntList touched = sc.touched; touched.n = 0;
        long tot = 0;
        for (int t = 0; t < g.k; t++) {
            int L = g.leavesPerTree[t] - qLeaves[t];
            int I = (g.intOff[t + 1] - g.intOff[t]) - impure[t];
            qLeaves[t] = 0; impure[t] = 0;
            if (L > 0) { int comps = L - I; sc.comps[t] = comps; tot += comps; touched.add(t); }
        }
        sc.totalComps = tot; sc.present = touched.n;
    }

    /** -gpu-verify for the comps rows of a GPU batch. */
    private void verifyRows(int[] cl, int[] rows) {
        Scratch sc = new Scratch(g.n, g.k);
        for (int i = 0; i < cl.length; i++) {
            computeComps(cl[i], sc);
            for (int t = 0; t < g.k; t++) {
                if (rows[i * g.k + t] != sc.comps[t])
                    throw new IllegalStateException("GPU/CPU comps mismatch for cluster " + cl[i] + " tree " + t + ": gpu=" + rows[i * g.k + t] + " cpu=" + sc.comps[t]);
            }
            clearComps(sc);
        }
    }
}
