package phylonet.coalescent.fast;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;

/**
 * Enumerates the DP state space reachable from the full species set, with each state's candidate
 * resolutions Psi(C) = {(A | C-A) : A and C-A both known clusters}.
 *
 * Psi(C) is found without bitset scans over all clusters: the side containing the bucketing species of C
 * (ClusterTable.bkey) has the same bucketing species, so only that bucket is scanned, and key(C) - key(A) is looked up
 * in the cluster hash table (O(1) per test).
 *
 * States with no resolution get DynaDup's fallback: for |C| <= 5 all bipartitions of C; otherwise
 * (A | C-A) for the largest known clusters A strictly inside C (at most 64). New clusters are registered in the
 * global table so later states can use them.
 *
 * Implementation notes (none of them changes the result):
 *  - buckets are split by size and by kind (complement / other);
 *  - complement candidates of a large C are read from an exact inverted index (species -> complements whose base
 *    cluster contains it) using one species outside C, instead of scanning every complement;
 *  - candidates are returned in a canonical order (size of A ascending, then cluster id), so the DP breaks ties
 *    deterministically;
 *  - buckets keyed by the rarest species (balanced), 16-byte prefilter entries + exact bitset tests;
 *  - speculative parallel search. The DFS must process states in its sequential order, because a fallback creates
 *    clusters that later states can use. But the expensive part of a state - scanning for its resolutions, or for the
 *    largest clusters inside it - can be computed early against the frozen table and completed later: a pair with a
 *    side created after the speculative result is found by testing just the clusters created since (ids >= N0)
 *    against C. So worker threads speculate, in waves, on the states near the top of the stack, on their future
 *    children (sides of their pairs), and on the predicted "rest" clusters C - A of fallbacks (searched from their
 *    species set before the cluster exists, and matched to it by hash key when it is created). The main thread
 *    commits states in exactly the sequential order; cluster ids, states and candidates are identical to the
 *    sequential algorithm (-Ddynadupx.specK=0).
 */
final class StateSpace {
    static final int MAX_FALLBACK = 64;
    static final boolean PROFC = System.getenv("DDX_PROFC") != null;
    static final int PAR_SCAN = Integer.getInteger("dynadupx.parScan", 16_384);      // longer scans are split over threads
    static final int INDEX_MIN = Integer.getInteger("dynadupx.compIndexMin", 20_000); // complements per bucket to use the index
    /** speculation: clusters per phase (-1 = 16 x threads, 0 = sequential), waves per phase, staleness limit */
    static final int SPEC_K = Integer.getInteger("dynadupx.specK", -1);
    static final int SPEC_WINDOW = Integer.getInteger("dynadupx.specWindow", 256);
    static final int SPEC_STALE = Integer.getInteger("dynadupx.specStale", 1 << 15);
    static final int SPEC_WAVES = Integer.getInteger("dynadupx.specWaves", 8);
    static final int SPEC_AFTER_MS = Integer.getInteger("dynadupx.specAfterMs", 2000);
    static final boolean SPEC_RESTS = !"false".equals(System.getProperty("dynadupx.specRests"));

    final ClusterTable ct;
    final IntList states = new IntList(), candA = new IntList(), candB = new IntList(), candStart = new IntList(), candEnd = new IntList();
    int[] stateOf = new int[1024];   // cluster id -> state index, -1 if not a state
    int fallbackStates = 0; long tResolve = 0, tFallback = 0;
    final LongAdder scans = new LongAdder(), nInside = new LongAdder(), nRare = new LongAdder(), nMixed = new LongAdder(), nPlain = new LongAdder(), nList = new LongAdder(), nOut = new LongAdder();
    long bucketScans;                // (profile) entries scanned
    private final int words;
    long specStates, specUsed, specVirtualUsed, fixupTests, phases;

    /** A speculative result: pairs (a0, b0, a1, b1, ...) among clusters with id < n0, and, if there are none and the
     *  cluster has more than 5 species, the largest size fz of a cluster strictly inside it with all such clusters fl. */
    private static final class Spec { final int n0; final int[] pairs; final int fz; final int[] fl;
        Spec(int n0, int[] pairs, int fz, int[] fl) { this.n0 = n0; this.pairs = pairs; this.fz = fz; this.fl = fl; } }
    private Spec[] specOf = new Spec[1024];                        // by cluster id
    private HashMap<ClusterKey, Spec> specVirtual = new HashMap<>(); // predicted rest clusters, by species-set key

    /** The cluster a search is about: an existing cluster, or a predicted rest cluster that does not exist yet. */
    private static final class Target {
        int id = -1; int size, bkey; long h1, h2, fold, cfold; long[] inC; int[] taxa; ClusterKey key;
    }

    /** Per-thread scratch (the speculation workers run in parallel). */
    private static final class Scratch {
        final int[] mark; int stamp = 0; final long[] inC;
        Scratch(int n, int words) { mark = new int[n]; inC = new long[words]; }
    }
    private final ThreadLocal<Scratch> tls;
    private final Scratch main;
    private Phases phasesExec;

    StateSpace(ClusterTable ct, int root) { this(ct, root, null, null, null); }

    /**
     * Gene-tree-split search space (-xs; the default search space of STELAR-X): a cluster that is the
     * species set of a gene-tree node with a valid split gets as candidates exactly the distinct gene-tree splits
     * (d1 | d2) of such nodes - no search. Clusters without one (the root, complements, fallback rests, clusters seen
     * only at duplication nodes) are resolved as in DynaDup's space: all pairs of known clusters, else the fallback.
     * stbStart/stbA/stbB: CSR by parent cluster id of the distinct valid gene-tree splits (null = DynaDup's space).
     */
    private final int[] stbStart, stbA, stbB;
    long stbStates;                  // (profile) states resolved from gene-tree splits
    boolean isStbState(int c) { return hasStb(c); }
    private boolean hasStb(int c) { return stbStart != null && c + 1 < stbStart.length && stbStart[c + 1] > stbStart[c]; }

    StateSpace(ClusterTable ct, int root, int[] stbStart, int[] stbA, int[] stbB) {
        this.stbStart = stbStart; this.stbA = stbA; this.stbB = stbB;
        this.ct = ct; words = (ct.nTaxa + 63) >>> 6;
        tls = ThreadLocal.withInitial(() -> new Scratch(ct.nTaxa, words));
        main = new Scratch(ct.nTaxa, words);
        java.util.Arrays.fill(stateOf, -1);
        IntList stack = new IntList(1024);
        register(root, stack);
        java.util.concurrent.ForkJoinPool fj = java.util.concurrent.ForkJoinTask.getPool();
        int par = Math.max(1, fj != null ? fj.getParallelism() : Runtime.getRuntime().availableProcessors());
        final int K = SPEC_K >= 0 ? SPEC_K : (par > 1 ? 16 * par : 0);
        // the look-ahead only pays off on long searches (on short ones its overhead dominates): it starts once the
        // search has run for SPEC_AFTER_MS; before that states are committed sequentially (same result either way)
        boolean spec = K > 0 && SPEC_AFTER_MS <= 0;
        if (spec) phasesExec = new Phases(par);
        final long tStart = System.nanoTime();
        final boolean prof = System.getenv("DDX_PROFILE") != null; long lastPrint = System.nanoTime(), done = 0;
        try {
            while (stack.n > 0) {
                if (!spec && K > 0 && (done & 255) == 0 && System.nanoTime() - tStart > SPEC_AFTER_MS * 1_000_000L) {
                    spec = true; phasesExec = new Phases(par);
                }
                done++;
                if (spec) {
                    int top = stack.a[stack.n - 1];
                    if (needsSpec(states.a[top])) speculate(stack, K);
                }
                int si = stack.a[--stack.n];
                if (prof && done % 1000 == 0 && System.nanoTime() - lastPrint > 20_000_000_000L) {
                    lastPrint = System.nanoTime();
                    System.err.printf("[profile] state space: %d states resolved, %d registered, %d candidates, speculative %d (used %d, %d of them predicted rests), phases %d%n",
                            done, states.size(), candA.size(), specStates, specUsed, specVirtualUsed, phases);
                }
                process(si, stack);
            }
        } finally { if (phasesExec != null) phasesExec.shutdown(); }
        if (PROFC) System.err.printf("[profc] scan: %d entries passed the fold + rare-species filter, %d passed the Bloom filter; other paths (fold passed): mixed %d, plain %d, list %d; pairs found %d%n", nRare.sum(), nInside.sum(), nMixed.sum(), nPlain.sum(), nList.sum(), nOut.sum());
        if (PROFC && phasesExec != null) System.err.printf("[profc] speculation: %d parallel runs, %d items; main thread %.1fs working, %.1fs waiting; commits %.1fs%n",
                phasesExec.nRuns, phasesExec.nItems, phasesExec.tWork / 1e9, phasesExec.tWait / 1e9, (tResolve + tFallback) / 1e9);
        bucketScans = scans.sum();
        specOf = null; specVirtual = null;
    }

    int state(int c) { return c < stateOf.length ? stateOf[c] : -1; }

    private void register(int c, IntList stack) {
        if (ct.size[c] <= 1 || state(c) >= 0) return;
        int si = states.size();
        if (c >= stateOf.length) { int old = stateOf.length; stateOf = java.util.Arrays.copyOf(stateOf, Math.max(c + 1, 2 * old)); java.util.Arrays.fill(stateOf, old, stateOf.length, -1); }
        states.add(c); candStart.add(0); candEnd.add(0); stateOf[c] = si; stack.add(si);
    }

    // ------------------------------------------------------------------------------------------ speculation
    /** Speculative result of cluster c; a result computed for a predicted rest is moved to the cluster once it exists. */
    private Spec spec(int c) {
        Spec s = c < specOf.length ? specOf[c] : null;
        if (s == null && !specVirtual.isEmpty() && ct.isExplicit(c)) {
            s = specVirtual.remove(ct.key(c));
            if (s != null) { setSpec(c, s); specVirtualUsed++; }
        }
        return s;
    }
    private boolean needsSpec(int c) { if (hasStb(c)) return false; Spec s = spec(c); return s == null || ct.n - s.n0 > SPEC_STALE; }
    private void setSpec(int c, Spec s) {
        if (c >= specOf.length) specOf = Arrays.copyOf(specOf, Math.max(c + 1, 2 * specOf.length));
        specOf[c] = s;
    }

    private Target realTarget(int c) {
        Target t = new Target(); t.id = c; t.size = ct.size[c]; t.bkey = ct.bkey[c]; t.h1 = ct.h1[c]; t.h2 = ct.h2[c];
        t.fold = ct.fold[c]; t.cfold = ct.cfold[c];
        t.inC = ct.bits != null ? Arrays.copyOfRange(ct.bits, ct.BW * c, ct.BW * c + ct.BW) : null;
        return t;
    }
    /** A predicted rest cluster (sorted species array), not yet in the table. */
    private Target virtualTarget(int[] sp, ClusterKey key) {
        Target t = new Target(); t.taxa = sp; t.size = sp.length; t.key = key; t.h1 = key.h1(); t.h2 = key.h2();
        long[] own = new long[4], comp = new long[4];
        boolean[] in = new boolean[ct.nTaxa];
        for (int v : sp) { in[v] = true; long h = ct.H1[v]; own[(int) ((h >>> 6) & 3)] |= 1L << (h & 63); }
        for (int v = 0; v < ct.nTaxa; v++) if (!in[v]) { long h = ct.H1[v]; comp[(int) ((h >>> 6) & 3)] |= 1L << (h & 63); }
        t.fold = own[0] | own[1] | own[2] | own[3]; t.cfold = comp[0] | comp[1] | comp[2] | comp[3];
        int best = -1; for (int v : sp) if (best < 0 || ct.rank[v] < ct.rank[best]) best = v;
        t.bkey = best;
        if (ct.bits != null) { t.inC = new long[ct.BW]; for (int v : sp) t.inC[v >>> 6] |= 1L << (v & 63); }
        return t;
    }
    private int[] taxaOf(Target t) { return t.taxa != null ? t.taxa : ct.taxa(t.id); }

    /** Speculates, in parallel against the frozen table and in waves, on up to K clusters: states near the top of the
     *  stack without a fresh result, then the future children of the results computed (sides of their pairs that are
     *  not states yet, and the predicted rests of their fallbacks). */
    private void speculate(IntList stack, int K) {
        ArrayList<Target> wave = new ArrayList<>();
        HashSet<Integer> picked = new HashSet<>();
        HashSet<ClusterKey> pickedV = new HashSet<>();
        for (int i = stack.n - 1, seen = 0; i >= 0 && seen < SPEC_WINDOW && wave.size() < K; i--, seen++) {
            int c = states.a[stack.a[i]];
            if (needsSpec(c) && picked.add(c)) wave.add(realTarget(c));
        }
        final int N0 = ct.n;
        phases++;
        int total = 0;
        for (int w = 0; w < SPEC_WAVES && !wave.isEmpty(); w++) {
            final Target[] p = wave.toArray(new Target[0]);
            final Spec[] res = new Spec[p.length];
            total += p.length;
            phasesExec.run(p.length, q -> {
                Target t = p[q];
                Scratch sc = tls.get();
                IntList a = new IntList(8), b = new IntList(8);
                resolveRaw(t, sc, a, b, false);
                int[] pairs = new int[2 * a.n];
                for (int i = 0; i < a.n; i++) { pairs[2 * i] = a.a[i]; pairs[2 * i + 1] = b.a[i]; }
                int fz = 0; int[] fl = null;
                if (a.n == 0 && t.size > 5) { IntList l = new IntList(4); fz = fallbackScan(t, sc, l, false); fl = l.toArray(); }
                res[q] = new Spec(N0, pairs, fz, fl);
            });
            for (int q = 0; q < p.length; q++) { if (p[q].id >= 0) setSpec(p[q].id, res[q]); else specVirtual.put(p[q].key, res[q]); }
            if (total >= K) break;
            ArrayList<Target> nxt = new ArrayList<>();
            for (int q = 0; q < p.length && total + nxt.size() < K; q++) {
                Spec r = res[q];
                for (int i = 0; i < r.pairs.length && total + nxt.size() < K; i++) {
                    int x = r.pairs[i];
                    if (ct.size[x] > 1 && state(x) < 0 && needsSpec(x) && picked.add(x)) nxt.add(realTarget(x));
                }
                if (SPEC_RESTS && r.fl != null && r.fl.length > 0) {    // predicted fallback: rests C - A
                    int[] C = taxaOf(p[q]).clone(); Arrays.sort(C);
                    IntList best = canonicalFallback(new IntList(r.fl));
                    int[] mk = main.mark;
                    for (int j = 0; j < best.n && total + nxt.size() < K; j++) {
                        int st = ++main.stamp;
                        for (int v : ct.taxa(best.a[j])) mk[v] = st;
                        IntList rest = new IntList(C.length);
                        for (int v : C) if (mk[v] != st) rest.add(v);
                        if (rest.n <= 1) continue;
                        int[] sp = rest.toArray();
                        ClusterKey key = ClusterKey.of(sp);
                        int ex = ct.find(key);
                        if (ex >= 0) { if (state(ex) < 0 && needsSpec(ex) && picked.add(ex)) nxt.add(realTarget(ex)); }
                        else if (!specVirtual.containsKey(key) && pickedV.add(key)) nxt.add(virtualTarget(sp, key));
                    }
                }
            }
            wave = nxt;
        }
        specStates += total;
    }

    /** Commits state si in the sequential DFS order. */
    private void process(int si, IntList stack) {
        final int c = states.a[si];
        candStart.set(si, candA.size());
        IntList outA = new IntList(8), outB = new IntList(8);
        long a0 = System.nanoTime();
        if (hasStb(c)) {                                   // gene-tree splits only (-xs)
            for (int e = stbStart[c], f = stbStart[c + 1]; e < f; e++) { outA.add(stbA[e]); outB.add(stbB[e]); }
            commitPairs(c, outA, outB);
            candEnd.set(si, candA.size());
            stbStates++; tResolve += System.nanoTime() - a0;
            for (int j = candStart.get(si); j < candA.size(); j++) { register(candA.get(j), stack); register(candB.get(j), stack); }
            return;
        }
        Spec sp = spec(c);
        if (sp != null) {
            specUsed++;
            for (int i = 0; i < sp.pairs.length; i += 2) { outA.add(sp.pairs[i]); outB.add(sp.pairs[i + 1]); }
            fixupResolve(c, sp.n0, outA, outB);
        } else resolveRaw(realTarget(c), main, outA, outB, true);
        commitPairs(c, outA, outB);
        long a1 = System.nanoTime(); tResolve += a1 - a0;
        if (candA.size() == candStart.get(si)) {
            fallbackStates++;
            if (sp != null && ct.size[c] > 5) fallbackCommit(c, sp.fz, sp.fl, sp.n0);
            else fallback(c);
            tFallback += System.nanoTime() - a1;
        }
        candEnd.set(si, candA.size());
        if (c < specOf.length) specOf[c] = null;
        for (int j = candStart.get(si); j < candA.size(); j++) { register(candA.get(j), stack); register(candB.get(j), stack); }
    }

    // ------------------------------------------------------------------------------------------ resolutions
    /** A unit of scanning work: packed entries [lo, hi) of one bucket list, or a slice of an index list. */
    private record Job(ClusterTable.Bucket b, int[] list, int lo, int hi) {}

    /** All pairs (A, C - A) of known clusters with A in C's bucket (unsorted). Read-only on the table. */
    private void resolveRaw(Target t, Scratch sc, IntList outA, IntList outB, boolean allowParallel) {
        final int sz = t.size, min = t.bkey;
        ArrayList<Job> jobs = new ArrayList<>();
        long total = 0;
        // non-complement clusters: packed scan (of the pair buckets (bkey(C), b) for the species b of C)
        long bySize = 0, byPair = Long.MAX_VALUE;
        final ClusterTable.SizeRow nrow = ct.row(min); final int nEnd = nrow == null ? 0 : nrow.below(sz);
        for (int k = 0; k < nEnd; k++) bySize += nrow.b[k].n;
        final ClusterTable.Bucket[] prow = ct.pbucket != null && t.inC != null ? ct.pbucket[min] : null;
        if (prow != null) {
            byPair = 0;
            final long[] inC = t.inC;
            for (int w = 0; w < inC.length && byPair < bySize; w++) { long x = inC[w]; while (x != 0) { int v = (w << 6) + Long.numberOfTrailingZeros(x); x &= x - 1;
                ClusterTable.Bucket b = prow[v]; if (b != null) byPair += b.n; } }
        }
        if (byPair < bySize) {
            final long[] inC = t.inC;
            for (int w = 0; w < inC.length; w++) { long x = inC[w]; while (x != 0) { int v = (w << 6) + Long.numberOfTrailingZeros(x); x &= x - 1;
                ClusterTable.Bucket b = prow[v]; if (b != null) { jobs.add(new Job(b, null, 0, b.n)); total += b.n; } } }
        } else for (int k = 0; k < nEnd; k++) { ClusterTable.Bucket b = nrow.b[k]; jobs.add(new Job(b, null, 0, b.n)); total += b.n; }
        // complement clusters: index lookup through one species outside C, or packed scan
        int[] best = null;
        long packedComp = 0;
        final ClusterTable.SizeRow crow = ct.crow(min); final int cEnd = crow == null ? 0 : crow.below(sz);
        for (int k = 0; k < cEnd; k++) packedComp += crow.b[k].n;
        if (ct.compCount[min] >= INDEX_MIN && sz < ct.nTaxa && packedComp >= INDEX_MIN) {
            int[][] idx;
            synchronized (ct) { idx = ct.compIndex(min); }
            int st = ++sc.stamp;
            for (int v : taxaOf(t)) sc.mark[v] = st;
            for (int x = 0; x < ct.nTaxa; x++) if (sc.mark[x] != st && (best == null || idx[x].length < best.length)) best = idx[x];
        }
        if (best != null && best.length < packedComp) { jobs.add(new Job(null, best, 0, best.length)); total += best.length; }
        else for (int k = 0; k < cEnd; k++) { ClusterTable.Bucket b = crow.b[k]; jobs.add(new Job(b, null, 0, b.n)); total += b.n; }
        scans.add(total);
        // exact membership of C: bitset, or marks (only without bitsets)
        int[] mk = null; int st = 0;
        if (t.inC == null) { mk = sc.mark; st = ++sc.stamp; for (int v : taxaOf(t)) mk[v] = st; }
        if (!allowParallel || total < PAR_SCAN || t.inC == null) {
            for (Job j : jobs) scan(t, j, mk, st, outA, outB);
        } else {
            java.util.concurrent.ForkJoinPool fj = java.util.concurrent.ForkJoinTask.getPool();
            int par = fj != null ? fj.getParallelism() : Runtime.getRuntime().availableProcessors();
            int per = (int) Math.max(4_096, total / (4L * par));
            ArrayList<Job> split = new ArrayList<>();
            for (Job j : jobs) for (int lo = j.lo; lo < j.hi; lo += per) split.add(new Job(j.b, j.list, lo, Math.min(j.hi, lo + per)));
            IntList[][] res = new IntList[split.size()][];
            java.util.stream.IntStream.range(0, split.size()).parallel().forEach(q -> {
                IntList a = new IntList(4), b = new IntList(4);
                scan(t, split.get(q), null, 0, a, b);
                res[q] = new IntList[]{a, b};
            });
            for (IntList[] r : res) for (int i = 0; i < r[0].n; i++) { outA.add(r[0].a[i]); outB.add(r[1].a[i]); }
        }
    }

    /** Pairs of state c with at least one side created after the speculative result (ids >= N0). */
    private void fixupResolve(int c, int N0, IntList outA, IntList outB) {
        final int N = ct.n, sz = ct.size[c];
        if (N0 >= N) return;
        final long c1 = ct.h1[c], c2 = ct.h2[c];
        final long[] inC = ct.bits != null ? Arrays.copyOfRange(ct.bits, ct.BW * c, ct.BW * c + ct.BW) : null;
        int[] mk = null; int st = 0;
        if (inC == null) { mk = main.mark; st = ++main.stamp; for (int v : ct.taxa(c)) mk[v] = st; }
        final long nf = ~ct.fold[c], cf = ct.cfold[c]; final long[] fold = ct.fold, cfold = ct.cfold;
        for (int x = N0; x < N; x++) {
            final int zx = ct.size[x];
            if (zx >= sz) continue;
            if (((fold[x] & nf) | (cf & ~cfold[x])) != 0) continue;    // 128-bit prefilter (as in the bucket scans)
            fixupTests++;
            if (!ct.inside(x, inC, mk, st)) continue;
            int y = ct.find(c1 - ct.h1[x], c2 ^ ct.h2[x], sz - zx);
            if (y < 0) continue;
            if (y >= N0 && y < x) continue;                 // both sides new: keep one of the two symmetric finds
            outA.add(x); outB.add(y);
        }
    }

    /** Canonical order: A = the side containing the smallest species id of C; size of A ascending, then
     *  insertion order (= cluster id). Sides never share an id, so the key is unique. */
    private void commitPairs(int c, IntList outA, IntList outB) {
        int m = outA.n;
        final int idmin = ct.minT[c];
        for (int i = 0; i < m; i++) if (ct.minT[outA.a[i]] != idmin) { int t = outA.a[i]; outA.a[i] = outB.a[i]; outB.a[i] = t; }
        if (m > 1) {
            long[] key = new long[m];
            for (int i = 0; i < m; i++) key[i] = ((long) ct.size[outA.a[i]] << 32) | outA.a[i];
            Integer[] ord = new Integer[m];
            for (int i = 0; i < m; i++) ord[i] = i;
            Arrays.sort(ord, Comparator.comparingLong(i -> key[i]));
            for (int i : ord) { candA.add(outA.a[i]); candB.add(outB.a[i]); }
        } else if (m == 1) { candA.add(outA.a[0]); candB.add(outB.a[0]); }
    }

    /** One scanning job for target C: 128-bit prefilter, exact subset test, then the key index for C - A. */
    private void scan(Target t, Job j, int[] mk, int st, IntList outA, IntList outB) {
        final long c1 = t.h1, c2 = t.h2;
        final int sz = t.size;
        final long nf = ~t.fold, cf = t.cfold;
        final KeyIndex idx = ct.index;
        final long[] inC = t.inC;
        final long[] H1 = ct.h1, H2 = ct.h2; final int[] size = ct.size;
        if (j.b != null && j.b.rare && inC != null) {       // fold + four rare species of A in C
            long cRare = 0, cBloom = 0;
            final long[] d = j.b.d; final int[] ids = j.b.ids;
            final boolean mixed = j.b.mixed; final int zz0 = mixed ? 0 : sz - size[ids[0]];
            for (int i = j.lo, p = 2 * j.lo; i < j.hi; i++, p += 2) {
                if ((d[p] & nf) != 0 || !ClusterTable.rareIn(d[p + 1], inC)) continue;
                if (PROFC) cRare++;
                final int a = ids[i];
                final int zz = mixed ? sz - size[a] : zz0;
                if (zz <= 0) continue;
                // the key of C - A is tested in the (cache-resident) Bloom filter before A's bitset is read: almost all
                // clusters inside C have no known rest, and the bitset lives in a large array
                final long k1 = c1 - H1[a], k2 = c2 ^ H2[a];
                if (!idx.mightContain(k1, k2, zz)) continue;
                if (PROFC) cBloom++;
                if (!ct.inside(a, inC, mk, st)) continue;
                int other = idx.get(k1, k2, zz);
                if (other >= 0) { outA.add(a); outB.add(other); }
            }
            if (PROFC) { nRare.add(cRare); nInside.add(cBloom); }
        } else if (j.b != null && j.b.mixed) {
            final long[] d = j.b.d; final int[] ids = j.b.ids;
            for (int i = j.lo, p = 2 * j.lo; i < j.hi; i++, p += 2) {
                if (((d[p] & nf) | (cf & ~d[p + 1])) != 0) continue;
                final int a = ids[i], z = size[a];
                if (z >= sz) continue;
                if (PROFC) nMixed.increment();
                final long k1 = c1 - H1[a], k2 = c2 ^ H2[a];                // Bloom filter before the bitset
                if (!idx.mightContain(k1, k2, sz - z) || !ct.inside(a, inC, mk, st)) continue;
                int other = idx.get(k1, k2, sz - z);
                if (other >= 0) { outA.add(a); outB.add(other); }
            }
        } else if (j.b != null) {
            final long[] d = j.b.d; final int[] ids = j.b.ids;
            final int zz = sz - size[ids[0]];
            for (int i = j.lo, p = 2 * j.lo; i < j.hi; i++, p += 2) {
                if (((d[p] & nf) | (cf & ~d[p + 1])) != 0) continue;
                final int a = ids[i];
                if (PROFC) nPlain.increment();
                final long k1 = c1 - H1[a], k2 = c2 ^ H2[a];
                if (!idx.mightContain(k1, k2, zz) || !ct.inside(a, inC, mk, st)) continue;
                int other = idx.get(k1, k2, zz);
                if (other >= 0) { outA.add(a); outB.add(other); }
            }
        } else {
            final int[] list = j.list;
            for (int i = j.lo; i < j.hi; i++) {
                int a = list[i], z = size[a];
                if (z >= sz) continue;
                if (((ct.fold[a] & nf) | (cf & ~ct.cfold[a])) != 0) continue;
                if (PROFC) nList.increment();
                final long k1 = c1 - H1[a], k2 = c2 ^ H2[a];
                if (!idx.mightContain(k1, k2, sz - z) || !ct.inside(a, inC, mk, st)) continue;
                int other = idx.get(k1, k2, sz - z);
                if (other >= 0) { outA.add(a); outB.add(other); }
            }
        }
    }

    // ------------------------------------------------------------------------------------------ fallback
    /** All non-complement clusters of size z strictly inside C (read-only; C given as inC, and marked with st). */
    private void fallbackSize(Target t, int z, long[] inC, int[] mark, int st, boolean useBits, IntList out) {
        long[] nz = ct.sizeMask[z];
        if (nz == null) return;                            // no cluster of size z
        final long nf = ~t.fold, cf = t.cfold;
        for (int w = 0; w < words; w++) {
            long bits = nz[w] & inC[w];
            while (bits != 0) {
                int v = (w << 6) + Long.numberOfTrailingZeros(bits); bits &= bits - 1;
                ClusterTable.Bucket b = ct.bucket(v, z);
                final long[] d = b.d; final int[] ids = b.ids;
                final boolean rare = b.rare;
                for (int i = 0, p = 0; i < b.n; i++, p += 2) {
                    if (rare ? ((d[p] & nf) != 0 || !ClusterTable.rareIn(d[p + 1], inC)) : ((d[p] & nf) | (cf & ~d[p + 1])) != 0) continue;
                    int id = ids[i];
                    if (useBits ? !ct.bitsIn(id, inC) : !ct.allMarked(id, mark, st)) continue;
                    out.add(id);
                }
            }
        }
    }

    /** Largest size z of a known non-complement cluster strictly inside C (|C| > 5) and all clusters of that size ->
     *  out. Sizes are examined from |C|-1 downwards; after the first few, blocks of sizes are examined in parallel
     *  (when allowed) and the largest size with a match wins, so the result is the same as the sequential search. */
    private int fallbackScan(Target t, Scratch sc, IntList out, boolean allowParallel) {
        int s = t.size;
        final long[] inC = sc.inC; Arrays.fill(inC, 0L);
        final int st = ++sc.stamp;
        for (int v : taxaOf(t)) { sc.mark[v] = st; inC[v >>> 6] |= 1L << (v & 63); }
        final boolean useBits = ct.bits != null;
        int z = s - 1;
        for (int seq = 0; z >= 1 && (seq < 4 || !allowParallel); z--, seq++) { fallbackSize(t, z, inC, sc.mark, st, useBits, out); if (out.size() > 0) return z; }
        java.util.concurrent.ForkJoinPool fj = java.util.concurrent.ForkJoinTask.getPool();
        int par = Math.max(1, fj != null ? fj.getParallelism() : Runtime.getRuntime().availableProcessors());
        int block = Math.max(8, 2 * par);
        final long[] inCf = inC.clone();
        while (z >= 1) {
            final int hiZ = z, loZ = Math.max(1, z - block + 1);
            IntList[] res = new IntList[hiZ - loZ + 1];
            final int[] mk = sc.mark;
            java.util.stream.IntStream.rangeClosed(loZ, hiZ).parallel().forEach(zz -> {
                IntList r = new IntList(4); fallbackSize(t, zz, inCf, mk, st, useBits, r); res[hiZ - zz] = r;
            });
            for (int i = 0; i < res.length; i++) if (res[i].size() > 0) { for (int k = 0; k < res[i].n; k++) out.add(res[i].a[k]); return hiZ - i; }
            z = loZ - 1;
        }
        return 0;
    }

    /** Order of fallback sides: smallest species id ascending, then newest (largest id) first; at most MAX_FALLBACK. */
    private IntList canonicalFallback(IntList all) {
        int m = all.size();
        long[] key = new long[m];
        for (int i = 0; i < m; i++) { int id = all.get(i); key[i] = ((long) ct.minT[id] << 32) | (0x7fffffffL - id); }
        Arrays.sort(key);
        IntList out = new IntList(Math.min(m, MAX_FALLBACK));
        for (int i = 0; i < m && i < MAX_FALLBACK; i++) out.add((int) (0x7fffffffL - (key[i] & 0xffffffffL)));
        return out;
    }

    /** Sequential fallback (no speculative result). */
    private void fallback(int c) {
        int s = ct.size[c];
        if (s <= 5) { fallbackSmall(c); return; }
        IntList all = new IntList(4);
        fallbackScan(realTarget(c), main, all, true);
        createRests(c, canonicalFallback(all));
    }

    /** Fallback from a speculative search (largest size z0 and its clusters, among ids < N0) completed with the
     *  clusters created since. */
    private void fallbackCommit(int c, int z0, int[] L0, int N0) {
        final int s = ct.size[c], N = ct.n;
        IntList all = new IntList(4);
        int z1 = 0; IntList nw = new IntList(4);
        if (N0 < N) {
            final long[] inC = ct.bits != null ? Arrays.copyOfRange(ct.bits, ct.BW * c, ct.BW * c + ct.BW) : null;
            int[] mk = null; int st = 0;
            if (inC == null) { mk = main.mark; st = ++main.stamp; for (int v : ct.taxa(c)) mk[v] = st; }
            for (int x = N0; x < N; x++) {
                if (ct.compOf[x] >= 0) continue;
                int zx = ct.size[x];
                if (zx >= s || zx < Math.max(z0, z1)) continue;
                fixupTests++;
                if (!ct.inside(x, inC, mk, st)) continue;
                if (zx > z1) { z1 = zx; nw.n = 0; }
                nw.add(x);
            }
        }
        int z = Math.max(z0, z1);
        if (z0 == z && L0 != null) for (int id : L0) all.add(id);
        if (z1 == z) for (int i = 0; i < nw.n; i++) all.add(nw.a[i]);
        createRests(c, canonicalFallback(all));
    }

    private void fallbackSmall(int c) {
        int[] C = ct.taxa(c).clone();
        Arrays.sort(C);
        int s = C.length;
        for (int mask = 1; mask < (1 << (s - 1)); mask++) {     // taxon C[s-1] always on side B
            IntList a = new IntList(), bb = new IntList();
            for (int j = 0; j < s; j++) if ((mask >> j & 1) != 0) a.add(C[j]); else bb.add(C[j]);
            candA.add(ct.addExplicit(a.toArray())); candB.add(ct.addExplicit(bb.toArray()));
        }
    }

    /** Candidates (A | C - A) for the chosen fallback sides A (in order); C - A is registered as a cluster. */
    private void createRests(int c, IntList best) {
        int[] C = ct.taxa(c).clone();
        Arrays.sort(C);
        final int[] mark = main.mark;
        for (int j = 0; j < best.size(); j++) {
            int a = best.get(j);
            int st2 = ++main.stamp;
            for (int v : ct.taxa(a)) mark[v] = st2;
            IntList rest = new IntList();
            for (int v : C) if (mark[v] != st2) rest.add(v);
            candA.add(a); candB.add(ct.addExplicit(rest.toArray()));
        }
    }

    /** Low-latency phase executor for the speculation: T-1 spinning worker threads plus the calling thread share the
     *  items of a phase through an atomic counter (a parallel stream costs tens of microseconds per phase; phases can
     *  hold only a few items). Each phase has its own counters, so a worker finishing an old phase cannot take items
     *  of a new one. */
    private static final class Phases {
        interface Task { void run(int i); }
        private static final class Job {
            final Task t; final int m; final AtomicInteger next = new AtomicInteger(), done = new AtomicInteger();
            volatile Throwable err;
            Job(Task t, int m) { this.t = t; this.m = m; }
        }
        private final Thread[] workers; private volatile boolean stop = false;
        private volatile Job cur;
        Phases(int threads) {
            workers = new Thread[threads - 1];
            for (int w = 0; w < workers.length; w++) {
                workers[w] = new Thread(this::loop, "dynadup-x-ss-" + w); workers[w].setDaemon(true); workers[w].start();
            }
        }
        private void loop() {
            Job last = null; long idle = 0;
            while (!stop) {
                Job j = cur;
                if (j == null || j == last) { if (++idle < 20_000) Thread.onSpinWait(); else java.util.concurrent.locks.LockSupport.parkNanos(20_000); continue; }
                last = j; idle = 0;
                work(j);
            }
        }
        private static void work(Job j) {
            int i;
            while ((i = j.next.getAndIncrement()) < j.m) {
                try { j.t.run(i); } catch (Throwable x) { j.err = x; }
                j.done.incrementAndGet();
            }
        }
        long tWork, tWait, nRuns, nItems;       // (profile) main thread: working on items / waiting for the others
        void run(int m, Task t) {
            if (m == 0) return;
            Job j = new Job(t, m);
            cur = j;                            // publish
            long a = PROFC ? System.nanoTime() : 0;
            work(j);
            long b = PROFC ? System.nanoTime() : 0;
            while (j.done.get() < m) Thread.onSpinWait();
            if (PROFC) { long e = System.nanoTime(); tWork += b - a; tWait += e - b; nRuns++; nItems += m; }
            if (j.err != null) throw new RuntimeException(j.err);
        }
        void shutdown() { stop = true; }
    }
}
