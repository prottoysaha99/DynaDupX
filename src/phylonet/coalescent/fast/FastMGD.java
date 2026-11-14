package phylonet.coalescent.fast;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * DynaDup-X: fast MGD (minimize gene duplications) species-tree inference following the
 * STELAR-X design, for rooted single- or multi-copy gene trees.
 *
 * Objective (identical to DynaDup): for a candidate species-tree bipartition x = (X|Y)
 *     w(x) = sum of f(y) over VALID gene-tree bipartitions y = (A|B) dominated by x,
 *            i.e. (A subset X and B subset Y) or (A subset Y and B subset X)
 * (each such gene node is a speciation node mapped to x); overlapping nodes (A and B share a
 * species) are always duplications. total cost = #gene internal nodes - max sum_x w(x).
 *
 * Pipeline (STELAR-X structure):
 *   1. parse -> post-order species arrays + (l,m,r) tuples                  (parallel over trees)
 *   2. distinct-species double hashing of every node / child range           (parallel over trees)
 *   3. frequency map of valid bipartitions (UGB) + cluster table (search space)
 *   4. DP state-space enumeration: Psi(C) by hash subtraction over bucket[min(C)]
 *   5. batched weight precomputation for all candidate bipartitions         (parallel; GPU kernel in native/)
 *   6. bottom-up DP over clusters sorted by size
 */
public final class FastMGD {
    static final double GPU_MAX_WORK = Double.parseDouble(System.getProperty("dynadupx.gpu.maxWork", "2e11"));
    /** -xs: gene-tree-split search space (see StateSpace) */
    static boolean stbSpace = false;
    /** dev experiment (off): with -xs a duplication node gives its cluster only the splits (A-B | B) and (A | B-A) instead
     *  of all pairs of known clusters. Measured on 1,000 species x 5,000 multi-copy trees: 1.8x faster, but 3.8x more
     *  wrong clusters (unrooted FN 0.138 vs 0.036), so the default keeps all pairs. */
    static final boolean DUP_SPLITS = "true".equals(System.getProperty("dynadupx.dupSplits"));
    static final boolean CHECK_KEYS = System.getenv("DDX_CHECK_KEYS") != null;   // debug: UGB ids from children == side lookups
    private static final Set<String> SUPPORTED = new HashSet<>(Arrays.asList("-i", "-a", "-o", "-threads", "-fast", "-xc", "-x0", "-xs", "-dl", "-lambda", "-mu", "-gpu", "-gpu-verify"));

    /** True if these arguments can be served by the fast engine (plain MGD, no legacy-only options). */
    public static boolean handles(String[] args) {
        for (String a : args) {
            if (a.equals("-legacy")) return false;
            if (a.startsWith("-") && !SUPPORTED.contains(a) && !a.equals("/delete/") && !isNumber(a)) return false;
        }
        return true;
    }

    private static boolean isNumber(String a) { try { Double.parseDouble(a); return true; } catch (NumberFormatException e) { return false; } }

    public static void main(String[] args) throws Exception {
        String in = null, out = null, mapFile = null, pattern = null, rep = null;
        int threads = Runtime.getRuntime().availableProcessors();
        Boolean xc = null;          // null = mode default: DynaDup space for MGD, extended space for MGDL
        Double wd = null;           // null = MGD; else MGDL with this loss weight (NaN = auto)
        Double lambdaArg = null; double muArg = 0;
        boolean gpu = false, gpuVerify = false;    // -lambda: minimise Dup + lambda * XL (weighted MGDL/MGDLX family)
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-i" -> in = args[++i];
                case "-o" -> out = args[++i];
                case "-threads" -> threads = Integer.parseInt(args[++i]);
                case "-fast" -> { }
                case "-xc" -> xc = true;
                case "-x0" -> xc = false;
                case "-xs" -> { xc = false; stbSpace = true; }
                case "-gpu" -> gpu = true;
                case "-gpu-verify" -> { gpu = true; gpuVerify = true; }
                case "-mu" -> { muArg = Double.parseDouble(args[++i]); }
                case "-lambda" -> { lambdaArg = Double.parseDouble(args[++i]); if (wd == null) wd = 0.0; if (lambdaArg < 0) throw new IllegalArgumentException("-lambda must be >= 0"); }
                case "-dl" -> {
                    String v = args[++i];
                    wd = v.equals("auto") ? Double.NaN : Double.parseDouble(v);
                    if (!Double.isNaN(wd) && wd < 0) throw new IllegalArgumentException("-dl weight must be >= 0 or auto");
                }
                case "-a" -> {
                    String v1 = args[++i];
                    if (i + 1 < args.length && (!args[i + 1].startsWith("-") || args[i + 1].equals("/delete/"))) {
                        pattern = v1; rep = args[++i]; if (rep.equals("/delete/")) rep = "";
                    } else mapFile = v1;
                }
                default -> throw new IllegalArgumentException("unsupported option for fast engine: " + args[i]);
            }
        }
        if (in == null) throw new IllegalArgumentException("-i <gene tree file> is required");
        Map<String, String> map = mapFile == null ? null : readMapping(mapFile);
        final String fin = in, fout = out, fp = pattern, fr = rep; final int T = threads; final Double fwd = wd; final Double flam = lambdaArg; final double fmu = muArg; final boolean fgpu = gpu, fgpuV = gpuVerify; final boolean fxc = xc != null ? xc : (wd != null && lambdaArg == null);
        ForkJoinPool pool = new ForkJoinPool(T);
        Throwable[] err = new Throwable[1];
        Thread th = new Thread(null, () -> {
            try { pool.submit(() -> { run(fin, fout, map, fp, fr, T, fxc, fwd, flam, fmu, fgpu, fgpuV); return null; }).get(); }
            catch (Throwable e) { err[0] = e instanceof ExecutionException ? e.getCause() : e; }
        }, "dynadup-x", 1L << 30);
        th.start(); th.join(); pool.shutdown();
        if (err[0] != null) { err[0].printStackTrace(); System.exit(1); }
    }

    static Map<String, String> readMapping(String file) throws IOException {
        Map<String, String> m = new HashMap<>();
        for (String line : Files.readAllLines(Paths.get(file))) {
            for (String s : line.trim().split(";")) {
                if (s.isBlank()) continue;
                int c = s.indexOf(':');
                String species = s.substring(0, c).trim();
                for (String g : s.substring(c + 1).split(",")) {
                    g = g.trim(); if (g.isEmpty()) continue;
                    if (m.put(g, species) != null) throw new IllegalArgumentException("gene name mapped twice: " + g);
                }
            }
        }
        return m;
    }

    // ------------------------------------------------------------------------------------------
    static final String VERSION = "DynaDup-X 1.3.3";
    static void run(String in, String out, Map<String, String> map, String pattern, String rep, int threads, boolean allComplements, Double wdArg, Double lambdaArg, double muArg, boolean useGpu, boolean gpuVerify) throws IOException {
        long t0 = System.nanoTime();
        System.err.println(VERSION + " (fast engine, " + threads + " threads, max heap " + (Runtime.getRuntime().maxMemory() >> 20) + " MB)");
        // tree strings are dropped as soon as they are parsed (they are about as large as the input file)
        final String[] lines;
        { List<String> ls = new ArrayList<>();
          try (BufferedReader r = Files.newBufferedReader(Paths.get(in), StandardCharsets.UTF_8)) {
              for (String s; (s = r.readLine()) != null; ) if (!s.isBlank()) ls.add(s.trim()); }
          lines = ls.toArray(new String[0]); }
        if (lines.length == 0) throw new IllegalArgumentException("empty or null list of trees");

        // 1. parse; leaves carry SPECIES ids (gene copies of one species share the id)
        ConcurrentHashMap<String, Integer> prov = new ConcurrentHashMap<>();
        AtomicInteger next = new AtomicInteger();
        final java.util.regex.Pattern pat = pattern != null ? java.util.regex.Pattern.compile(pattern) : null;   // = String.replaceAll, compiled once
        NewickParser.ParsedTree[] parsed = new NewickParser.ParsedTree[lines.length];
        IntStream.range(0, lines.length).parallel().forEach(t -> {
            try {
                String line = lines[t]; lines[t] = null;
                parsed[t] = NewickParser.parse(line, label -> {
                    String sp = map != null ? map.get(label) : pat != null ? pat.matcher(label).replaceAll(rep) : label;
                    if (sp == null) throw new IllegalArgumentException("leaf " + label + " not defined in the mapping file");
                    return prov.computeIfAbsent(sp, k -> next.getAndIncrement());
                });
            } catch (RuntimeException e) { throw new IllegalArgumentException("Failed to parse tree number " + (t + 1) + ": " + e.getMessage(), e); }
        });
        final int n = prov.size(), k = parsed.length;
        String[] names = prov.keySet().toArray(new String[0]);
        Arrays.sort(names);
        int[] remap = new int[n];
        for (int i = 0; i < n; i++) remap[prov.get(names[i])] = i;
        long[] H1 = new long[n], H2 = new long[n];
        for (int i = 0; i < n; i++) { H1[i] = Hashing.h1(i); H2[i] = Hashing.h2(i); }

        // 2.+3. per-tree scans (parallel) and, in tree order, the cluster table, the valid gene STBs (UGB) with their
        // frequencies and the node-cluster occurrences. Done in chunks of trees, so the per-node hash keys
        // of only one chunk exist at a time (they are ~5x larger than the trees themselves).
        int[][] leaves = new int[k][], prevs = new int[k][];
        final int[][] triples = new int[k][];
        long totalInternal = 0; for (NewickParser.ParsedTree pt : parsed) totalInternal += pt.internalCount();
        ClusterTable ct = new ClusterTable(n, leaves, prevs, H1);
        for (int v = 0; v < n; v++) ct.addExplicit(new int[]{v});
        // UGB index: a valid STB is identified by the (unordered) pair of its sides' cluster ids = its children's clusters
        LongIntMap ugIndex = new LongIntMap(1 << 16);
        IntList yA = new IntList(), yB = new IntList();
        long[] yF = new long[1024];
        // per-tree occurrence counts (species and node clusters per gene tree) are only used by the MGDL loss terms (and by the
        // GPU runner, which uploads them); plain MGD needs only which clusters are gene-node clusters
        final boolean lossData = wdArg != null || useGpu;
        IntList occPid = new IntList((int) Math.min(Integer.MAX_VALUE - 8, Math.max(16, totalInternal))),
                occTid = new IntList(lossData ? (int) Math.min(Integer.MAX_VALUE - 8, Math.max(16, totalInternal)) : 1);   // (node cluster id, tree) per internal node
        long sigmaN = 0, validNodes = 0, invalidNodes = 0, polys = 0, weightedConst = 0;
        int[] speciesPerTree = new int[k], leavesPerTree = new int[k];
        IntList[] occT = new IntList[n], occC = new IntList[n];
        for (int v = 0; v < n; v++) { occT[v] = new IntList(4); occC[v] = new IntList(4); }
        int[] cnt = new int[n];
        // For every internal-node cluster P one decomposition P = d1 u d2 into strictly smaller clusters
        // (children of one of its gene nodes), and the parent node cluster of every valid STB (bit-parallel weights)
        int[] decA = new int[1 << 12], decB = new int[1 << 12]; Arrays.fill(decA, -1);
        final java.util.BitSet decValid = new java.util.BitSet();
        IntList yPar = new IntList(1 << 12);
        final IntList dsPar = new IntList(16), dsA = new IntList(16), dsB = new IntList(16);   // -xs splits of duplication nodes
        final int chunk = Math.max(64, 8 * threads);
        final GeneScan[] scans = new GeneScan[Math.min(k, chunk)];
        for (int tc = 0; tc < k; tc += chunk) {
            final int c0 = tc, c1 = Math.min(k, tc + chunk);
            IntStream.range(c0, c1).parallel().forEach(t -> {
                int[] L = parsed[t].leaves;
                for (int p = 0; p < L.length; p++) L[p] = remap[L[p]];
                scans[t - c0] = new GeneScan(L, parsed[t].triples, n, H1, H2);
            });
            for (int t = c0; t < c1; t++) { GeneScan s = scans[t - c0]; leaves[t] = s.leaves; prevs[t] = s.prev; triples[t] = s.triples; }
            for (int t = c0; t < c1; t++) {
                GeneScan s = scans[t - c0]; int[] tr = s.triples;
                int[] pidOf = new int[tr.length / 3];
                sigmaN += tr.length / 3; polys += parsed[t].polytomies;
                leavesPerTree[t] = s.leaves.length;
                for (int v : s.leaves) if (cnt[v]++ == 0) speciesPerTree[t]++;
                if (lossData) { for (int v : s.leaves) if (cnt[v] > 0) { occT[v].add(t); occC[v].add(cnt[v]); cnt[v] = 0; } }
                else for (int v : s.leaves) cnt[v] = 0;
                weightedConst += 2L * (speciesPerTree[t] - 1);
                for (int u = 0; u < tr.length / 3; u++) {
                    int pid = ct.addRange(s.kP(u), s.minP[u], t, tr[3 * u], tr[3 * u + 2]);
                    occPid.add(pid); if (lossData) occTid.add(t);
                    pidOf[u] = pid;
                    if (pid >= decA.length) { int old = decA.length, cap = Math.max(pid + 1, old + (old >> 1));
                        decA = Arrays.copyOf(decA, cap); decB = Arrays.copyOf(decB, cap); Arrays.fill(decA, old, cap, -1); }
                    int l = tr[3 * u], mid = tr[3 * u + 1], r = tr[3 * u + 2];     // post-order: right child u-1, left child u-(r-mid)
                    int childA = mid > l ? pidOf[u - (r - mid)] : s.leaves[l], childB = r > mid + 1 ? pidOf[u - 1] : s.leaves[r];
                    // prefer the split of a valid node (then the decomposition is itself a gene-tree split of the cluster,
                    // which the sparse weight passes test with the parts they have just read); any decomposition gives the same weights
                    if ((decA[pid] < 0 || (s.valid[u] && !decValid.get(pid))) && s.zA[u] < s.zP[u] && s.zB[u] < s.zP[u]) {
                        decA[pid] = childA; decB[pid] = childB; if (s.valid[u]) decValid.set(pid); }
                    if (!s.valid[u]) { invalidNodes++; continue; }
                    validNodes++;
                    if (CHECK_KEYS && (childA != ct.find(s.kA(u)) || childB != ct.find(s.kB(u)))) throw new IllegalStateException("child cluster != side cluster");
                    long key = childA < childB ? ((long) childA << 32) | childB : ((long) childB << 32) | childA;
                    int id = ugIndex.getOrPut(key, yA.size());
                    if (id < 0) {
                        id = yA.size();
                        yA.add(childA); yB.add(childB); yPar.add(pid);
                        if (id == yF.length) yF = Arrays.copyOf(yF, id * 2);
                    }
                    yF[id]++;
                }
                if (stbSpace && DUP_SPLITS && s.exA != null) {
                    // -xs: a duplication node (children A, B sharing species) gives its cluster A u B the splits
                    // (A-B | B) and (A | B-A)
                    int[] exId = new int[s.extra.size()];
                    for (int i = 0; i < exId.length; i++) exId[i] = ct.addExplicit(s.extra.get(i));
                    for (int u = 0; u < tr.length / 3; u++) {
                        if (s.valid[u]) continue;
                        int l = tr[3 * u], mid = tr[3 * u + 1], r = tr[3 * u + 2];
                        int childA = mid > l ? pidOf[u - (r - mid)] : s.leaves[l], childB = r > mid + 1 ? pidOf[u - 1] : s.leaves[r];
                        if (s.exA[u] >= 0) { dsPar.add(pidOf[u]); dsA.add(exId[s.exA[u]]); dsB.add(childB); }
                        if (s.exB[u] >= 0) { dsPar.add(pidOf[u]); dsA.add(childA); dsB.add(exId[s.exB[u]]); }
                    }
                } else for (int[] x : s.extra) ct.addExplicit(x);
                parsed[t] = null; scans[t - c0] = null;
            }
        }
        ugIndex = null;
        int[] allTaxa = new int[n];
        for (int v = 0; v < n; v++) allTaxa[v] = v;
        ClusterKey kAll = ClusterKey.of(allTaxa);
        ct.all = kAll;
        int root = ct.addExplicit(kAll, 0, allTaxa);
        // Complementary clusters (w.r.t. all species). Default: complements of EVERY cluster (larger search space,
        // better optima). -x0: DynaDup's own space (with cs = cd = 1 only the largest-size group).
        int base = ct.n; int[] scratch = new int[n];
        int largest = 0;
        for (int id = 0; id < base; id++) if (ct.size[id] < n) largest = Math.max(largest, ct.size[id]);
        for (int id = 0; id < base; id++) {
            int z = ct.size[id];
            if (z >= n || (!allComplements && z != largest)) continue;
            ClusterKey ck = kAll.minus(ct.key(id));
            if (ct.find(ck) >= 0) continue;
            ct.addComplement(ck, ct.minOfComplement(id, scratch, id + 1), id);
        }
        final int U = yA.size();
        long t1 = System.nanoTime();
        if (System.getenv("DDX_PROFILE") != null) System.err.printf("[profile] prep (parse, scan, cluster table, complements) %.1fs%n", (t1 - t0) / 1e9);
        boolean dl = wdArg != null;
        double wd = dl ? wdArg : 0;
        if (dl && Double.isNaN(wd)) {   // DynaDup "-dl auto": same expression as DuplicationWeightCounter
            wd = (weightedConst + 2 * k + 0.0D) / 2 * ((double) k * n);
            System.err.println("Estimated bd weight = " + wd);
        }
        System.err.println("Number of taxa: " + n);
        System.err.printf("Gene trees: %d, internal nodes: %d (%d valid STBs, %d overlapping), polytomies resolved: %d%n", k, sigmaN, validNodes, invalidNodes, polys);
        System.err.println("STBs in gene trees (count): " + U);
        System.err.println("STBs in gene trees (sum): " + validNodes);

        if (System.getenv("DDX_PROFILE") != null) {
            long[] baseB = new long[Math.min(n, 8)], compB = new long[Math.min(n, 8)]; long baseAll = 0, compAll = 0;
            for (int id = 0; id < ct.n; id++) {
                boolean comp = ct.compOf[id] >= 0; if (comp) compAll++; else baseAll++;
                int m = ct.minT[id]; if (m < baseB.length) { if (comp) compB[m]++; else baseB[m]++; }
            }
            System.err.printf("[profile] clusters: base %d, complements %d; bucket[min=0..7] base %s complement %s%n", baseAll, compAll,
                    Arrays.toString(baseB), Arrays.toString(compB));
        }
        // 4. DP state-space enumeration (Psi) ------------------------------------------------
        ct.enableBuckets();
        StateSpace ss;
        if (stbSpace) {                                   // CSR of the distinct valid gene-tree splits by parent cluster
            final int Nc = ct.n, U0 = yA.size(); final int[] par = yPar.a;
            int[] st = new int[Nc + 1];
            for (int y = 0; y < U0; y++) st[par[y] + 1]++;
            // splits of duplication nodes (-xs): distinct per cluster, and only for clusters without a valid split
            long[] dk = null; int D = 0;
            if (dsPar.n > 0) {
                dk = new long[dsPar.n];
                for (int i = 0; i < dsPar.n; i++) {
                    int p = dsPar.a[i]; if (st[p + 1] > 0) continue;          // (counts only so far: st[p + 1] = valid splits of p)
                    dk[D++] = i;                                             // index; sorted below by (p, smaller side, larger side)
                }
                final int[] idx = new int[D];
                for (int j = 0; j < D; j++) idx[j] = (int) dk[j];
                Integer[] ord = new Integer[D]; for (int j = 0; j < D; j++) ord[j] = j;
                Arrays.sort(ord, (x, y) -> {
                    int ix = idx[x], iy = idx[y];
                    if (dsPar.a[ix] != dsPar.a[iy]) return Integer.compare(dsPar.a[ix], dsPar.a[iy]);
                    int ax = Math.min(dsA.a[ix], dsB.a[ix]), ay = Math.min(dsA.a[iy], dsB.a[iy]);
                    if (ax != ay) return Integer.compare(ax, ay);
                    return Integer.compare(Math.max(dsA.a[ix], dsB.a[ix]), Math.max(dsA.a[iy], dsB.a[iy]));
                });
                int m = 0;
                for (int j = 0; j < D; j++) {
                    int i = idx[ord[j]];
                    if (m > 0) { int pi = (int) dk[m - 1];
                        if (dsPar.a[pi] == dsPar.a[i] && Math.min(dsA.a[pi], dsB.a[pi]) == Math.min(dsA.a[i], dsB.a[i]) && Math.max(dsA.a[pi], dsB.a[pi]) == Math.max(dsA.a[i], dsB.a[i])) continue; }
                    dk[m++] = i;
                }
                D = m;
                for (int j = 0; j < D; j++) st[dsPar.a[(int) dk[j]] + 1]++;
            }
            for (int c = 0; c < Nc; c++) st[c + 1] += st[c];
            int[] fill = Arrays.copyOf(st, Nc), sa = new int[U0 + D], sb2 = new int[U0 + D];
            for (int y = 0; y < U0; y++) { int e = fill[par[y]]++; sa[e] = yA.a[y]; sb2[e] = yB.a[y]; }
            for (int j = 0; j < D; j++) { int i = (int) dk[j]; int e = fill[dsPar.a[i]]++; sa[e] = dsA.a[i]; sb2[e] = dsB.a[i]; }
            if (D > 0) System.err.printf("Search space: %d distinct splits of duplication nodes for clusters without a speciation split%n", D);
            ss = new StateSpace(ct, root, st, sa, sb2);
        } else ss = new StateSpace(ct, root);
        long t2 = System.nanoTime();
        System.err.println("Number of Clusters: " + ct.n);
        System.err.printf("DP states: %d, candidate bipartitions: %d, fallback states: %d%n", ss.states.size(), ss.candA.size(), ss.fallbackStates);
        if (stbSpace) System.err.printf("Search space: gene-tree splits (-xs): %d of %d states resolved by their gene-tree splits%n", ss.stbStates, ss.states.size());
        if (StateSpace.PROFC) {       // dev: states and candidates by size decile, gene-split states vs the others
            long[] sS = new long[10], sC = new long[10], oS = new long[10], oC = new long[10];
            for (int si = 0; si < ss.states.size(); si++) {
                int c = ss.states.a[si], d = Math.min(9, 10 * ct.size[c] / n), m = ss.candEnd.a[si] - ss.candStart.a[si];
                if (ss.isStbState(c)) { sS[d]++; sC[d] += m; } else { oS[d]++; oC[d] += m; }
            }
            System.err.println("[profc] by size decile: gene-split states " + Arrays.toString(sS) + " cands " + Arrays.toString(sC));
            System.err.println("[profc] by size decile: other states      " + Arrays.toString(oS) + " cands " + Arrays.toString(oC));
            long[] kS = new long[3], kC = new long[3];       // non-split states by kind: gene-tree node cluster, explicit, complement
            for (int si = 0; si < ss.states.size(); si++) {
                int c = ss.states.a[si]; if (ss.isStbState(c)) continue;
                int kd = ct.compOf[c] >= 0 ? 2 : ct.repTree[c] >= 0 ? 0 : 1;
                kS[kd]++; kC[kd] += ss.candEnd.a[si] - ss.candStart.a[si];
            }
            System.err.println("[profc] other states by kind (gene-tree cluster, explicit, complement): " + Arrays.toString(kS) + " cands " + Arrays.toString(kC));
        }
        if (StateSpace.PROFC) System.err.printf("[profc] entries scanned %d, candidates %d; speculative %d (used %d, %d of them predicted rests), phases %d, fix-up tests %d%n",
                ss.bucketScans, ss.candA.size(), ss.specStates, ss.specUsed, ss.specVirtualUsed, ss.phases, ss.fixupTests);
        if (System.getenv("DDX_PROFILE") != null) {
            long explInts = 0, nExpl = 0, nComp = 0, nRange = 0;
            for (int id = 0; id < ct.n; id++) { if (ct.isExplicit(id)) { nExpl++; explInts += ct.size[id]; } else if (ct.compOf[id] >= 0) nComp++; else nRange++; }
            Runtime rt = Runtime.getRuntime(); System.gc();
            System.err.printf("[profile] clusters=%d (range %d, complement %d, explicit %d holding %d ints); bucket scans=%d; resolve %.1fs, fallback %.1fs; heap used %.0f MB%n",
                    ct.n, nRange, nComp, nExpl, explInts, ss.bucketScans, ss.tResolve / 1e9, ss.tFallback / 1e9, (rt.totalMemory() - rt.freeMemory()) / 1048576.0);
        }

        // the hash index and buckets are only needed to build the state space
        // gene-side structures (need final cluster ids). The per-candidate weight structures (side buckets,
        // adjacency) are only built when that method is used: GPU runs, -Ddynadupx.bpw=0, DDX_CHECK_W.
        final boolean oldWeights = Scorer.BPW_MODE == 0 || useGpu || Scorer.CHECK_W
                || System.getProperty("dynadupx.gpu.packcheck") != null || System.getProperty("dynadupx.gpu.simulate") != null;
        ct.releaseSearch(useGpu || System.getProperty("dynadupx.gpu.simulate") != null || System.getProperty("dynadupx.gpu.packcheck") != null);
        final int N = ct.n;
        IntList[] adj = oldWeights ? new IntList[N] : null;
        if (oldWeights) for (int y = 0; y < U; y++) {
            int a = yA.get(y), b = yB.get(y);
            if (adj[a] == null) adj[a] = new IntList(2);
            if (adj[b] == null) adj[b] = new IntList(2);
            adj[a].add(y); adj[b].add(y);
        }
        int[][] adjY = oldWeights ? new int[N][] : null; int[] EMPTY = new int[0];
        IntList[] sb = new IntList[n], nb = new IntList[n];
        for (int v = 0; v < n; v++) { sb[v] = new IntList(2); nb[v] = new IntList(2); }
        int[][] nodeTree = new int[N][], nodeCnt = new int[N][];
        // CSR of (node cluster -> trees): internal nodes were visited tree by tree, so each cluster's list is sorted
        if (!lossData) {                                   // MGD: only mark the gene-node clusters
            int m = occPid.size(); int[] pa = occPid.a;
            for (int i = 0; i < m; i++) { nodeTree[pa[i]] = EMPTY; nodeCnt[pa[i]] = EMPTY; }
            occPid = null; occTid = null;
        }
        else { int[] occStart = new int[N + 1];
        { int m = occPid.size(); int[] pa = occPid.a;
          for (int i = 0; i < m; i++) occStart[pa[i] + 1]++;
          for (int i = 0; i < N; i++) occStart[i + 1] += occStart[i];
          int[] fill = Arrays.copyOf(occStart, N), occT2 = new int[m];
          for (int i = 0; i < m; i++) occT2[fill[pa[i]]++] = occTid.a[i];
          occPid = null; occTid = null;
          for (int id = 0; id < N; id++) {
              int a = occStart[id], b = occStart[id + 1];
              if (a == b) continue;
              int d = 1; for (int j = a + 1; j < b; j++) if (occT2[j] != occT2[j - 1]) d++;
              int[] tt = new int[d], cc = new int[d]; int q = -1;
              for (int j = a; j < b; j++) { if (j == a || occT2[j] != occT2[j - 1]) { tt[++q] = occT2[j]; cc[q] = 1; } else cc[q]++; }
              nodeTree[id] = tt; nodeCnt[id] = cc;
          }
        } }
        int[] bySize = new int[N];                         // ids sorted by size (stable counting sort)
        { int[] c2 = new int[n + 2]; for (int i = 0; i < N; i++) c2[ct.size[i] + 1]++; for (int z = 0; z <= n; z++) c2[z + 1] += c2[z];
          for (int i = 0; i < N; i++) bySize[c2[ct.size[i]]++] = i; }
        for (int id : bySize) {
            if (oldWeights) adjY[id] = adj[id] == null ? EMPTY : adj[id].toArray();
            if (oldWeights && adj[id] != null) sb[ct.minT[id]].add(id);
            if (nodeTree[id] != null) nb[ct.minT[id]].add(id);
        }
        adj = null;
        if (!oldWeights) adjY = null;
        int[][] sideBucket = oldWeights ? new int[n][] : null, nodeBucket = new int[n][], occTree = new int[n][], occCnt = new int[n][];
        for (int v = 0; v < n; v++) { if (oldWeights) sideBucket[v] = sb[v].toArray(); nodeBucket[v] = nb[v].toArray(); occTree[v] = occT[v].toArray(); occCnt[v] = occC[v].toArray(); }
        GeneData gd = new GeneData(n, k, yA.toArray(), yB.toArray(), Arrays.copyOf(yF, U), adjY, sideBucket, nodeTree, nodeCnt,
                nodeBucket, occTree, occCnt, speciesPerTree, leavesPerTree);
        gd.pack(ct);
        gd.setDecompositions(ct, decA, decB, yPar.trimmed());
        decA = null; decB = null;
        if (Scorer.PAIR_INDEX) {
            long tp = System.nanoTime();
            gd.buildPairIndex(ct, dl || !Scorer.SPARSE_PASS || Scorer.PREFIX_MODE == 2);   // MGD with sparse passes: exact tests only for the report
            if (System.getenv("DDX_PROFILE") != null && gd.prQ != null)
                System.err.printf("[profile] pair index: %d node clusters under %d species pairs, %.2fs%n", gd.nodeCount, gd.prB.length, (System.nanoTime() - tp) / 1e9);
        }
        if (dl) gd = new GeneData(gd, triples, leaves);

        // 5.+6. scores fused with the bottom-up DP (layer by layer; no per-candidate score arrays) ---------------
        final int[] cA = ss.candA.trimmed(), cB = ss.candB.trimmed();
        Scorer scorer = new Scorer(ct, gd, ss, dl, wd);
        if (lambdaArg != null) { scorer.lambda = lambdaArg; scorer.mu = muArg; }
        scorer.verifyGpu = gpuVerify;
        if (System.getProperty("dynadupx.gpu.packcheck") != null) GpuRunner.checkSides(ct, gd.yA, gd.yB);
        if (System.getProperty("dynadupx.gpu.simulate") != null) {
            gd.attachTrees(leaves, prevs);
            GpuSim.run(ct, gd, cA, cB, Integer.parseInt(System.getProperty("dynadupx.gpu.simulate")), 42);
        }
        GpuRunner gpuRunner = null;
        if (useGpu) {
            String gpuProblem = null;
            if (Gpu.AVAILABLE) {
                try { gpuProblem = Gpu.selfTest(); } catch (Throwable t) { gpuProblem = t.toString(); }
            }
            if (!Gpu.AVAILABLE) {
                System.err.println("[gpu] CUDA library not loaded (" + Gpu.loadError + "); continuing on the CPU. "
                        + "Build it with ./build_gpu.sh and pass -Djava.library.path=<dir> or -Ddynadupx.lib=<file>.");
            } else if (gpuProblem != null) {
                System.err.println("[gpu] disabled: " + gpuProblem + " (continuing on the CPU; results are the same)");
            } else if ((double) cA.length * gd.yA.length > GPU_MAX_WORK) {
                // the kernels are brute force (every gene STB for every candidate; every node cluster for every
                // state), so on very large inputs they would be slower than the CPU's indexed enumeration
                System.err.printf("[gpu] skipped: %d candidates x %d gene bipartitions = %.2g pair tests exceeds %.2g "
                        + "(-Ddynadupx.gpu.maxWork); the CPU path is faster for an input of this size%n",
                        cA.length, gd.yA.length, (double) cA.length * gd.yA.length, GPU_MAX_WORK);
            } else {
                long g0 = System.nanoTime();
                try {
                    gd.attachTrees(leaves, prevs);
                    gpuRunner = new GpuRunner(ct, gd, gd.yA, gd.yB, gd.yF);
                    scorer.gpuW = gpuRunner.weights(cA, cB);
                    long nodeClusters = 0; for (int[] x : gd.nodeTree) if (x != null) nodeClusters++;
                    if (dl && (double) ss.states.size() * nodeClusters <= GPU_MAX_WORK) scorer.gpuComps = gpuRunner;   // per DP layer batch
                    else if (dl) System.err.printf("[gpu] comps stay on the CPU: %d states x %d node clusters exceeds %.2g%n", ss.states.size(), nodeClusters, GPU_MAX_WORK);
                    System.err.printf("[gpu] %s: weights for %d candidates in %.2fs%s%n", Gpu.deviceName(), cA.length,
                            (System.nanoTime() - g0) / 1e9, scorer.gpuComps != null ? "; comps computed per DP layer on the GPU" : "");
                } catch (Throwable t) {
                    System.err.println("[gpu] disabled: " + t + " (continuing on the CPU; results are the same)");
                    scorer.gpuW = null; scorer.gpuComps = null;
                    if (gpuRunner != null) { try { gpuRunner.close(); } catch (Throwable ignored) {} gpuRunner = null; }
                }
            }
        }
        if (dl && System.getenv("DDX_BENCH_COMPS") != null) { scorer.benchComps(); return; }
        long t3 = System.nanoTime();
        Scorer.Result res;
        try { res = scorer.solve(cA, cB); }
        catch (RuntimeException e) {
            if (scorer.gpuComps == null) throw e;
            System.err.println("[gpu] comps disabled: " + e + " (rerunning the DP on the CPU)");
            scorer.gpuComps = null;
            res = scorer.solve(cA, cB);
        }
        final double[] V = res.V; final int[] best = res.best;
        double score = V[ss.state(root)];
        double sigma0;
        double sigma = dl ? sigmaN + (1 - wd) * weightedConst : sigmaN;
        double cost = sigma - score;
        sigma0 = sigma;
        String tree = newick(ct, ss, best, cA, cB, root, names);
        if (dl) {   // decompose the chosen tree into duplications, extra lineages and losses (Zhang 2011 identity)
            long sumW = 0, sumXL = 0, lossConst = 0, sumB = 0;
            ArrayDeque<Integer> stk = new ArrayDeque<>(); stk.push(root);
            while (!stk.isEmpty()) {
                int c = stk.pop();
                if (ct.size[c] == 1) { sumXL += scorer.singletonXL(c); continue; }
                int si = ss.state(c), b = best[si];
                long[] comp = scorer.components(si, b, cA, cB);
                sumW += comp[0]; sumXL += comp[1]; sumB += comp[2]; stk.push(cA[b]); stk.push(cB[b]);
            }
            for (int t = 0; t < k; t++) lossConst -= 2L * (leavesPerTree[t] - speciesPerTree[t]);   // loss = XL + 2 Dup - 2 sum(n_i - |L_i|), checked against an independent reconciliation
            long dup = sigmaN - sumW;
            System.err.println("EVENTS dup=" + dup + " xl=" + sumXL + " loss=" + (sumXL + 2 * dup + lossConst) + " B=" + sumB);
            if (lambdaArg != null) { cost = dup + lambdaArg * sumXL + muArg * sumB; score = -cost; }
        }
        if (gpuRunner != null) gpuRunner.close();
        long t4 = System.nanoTime();

        System.err.println("Sigma N: " + sigma);
        System.err.println("total cost: " + cost);
        System.err.printf("DynaDup-X (%s, %s search space) timings: prep %.2fs, state space %.2fs, setup %.2fs, scores+DP %.2fs (threads=%d)%n",
                dl ? "MGDL wd=" + wd : "MGD", stbSpace ? "gene-tree split" : allComplements ? "extended" : "DynaDup", (t1 - t0) / 1e9, (t2 - t1) / 1e9, (t3 - t2) / 1e9, (t4 - t3) / 1e9, threads);
        if (System.getenv("DDX_PROFILE") != null) { Runtime rt = Runtime.getRuntime(); System.err.printf("[profile] heap in use at end %.0f MB (max %.0f MB)%n", (rt.totalMemory() - rt.freeMemory()) / 1048576.0, rt.maxMemory() / 1048576.0); }
        System.err.println("Optimal tree inferred in " + (t4 - t0) / 1e9 + " secs");
        if (out != null) Files.writeString(Paths.get(out), tree + " \n");
        else System.out.println(tree + " \n" + cost + (dl ? " duplication+loss" : " duplications") + " in total");
    }

    /** y = (leaves[l..m] | leaves[m+1..r]) dominated by the marked bipartition (sa-side | sb-side)? */
    static boolean dominated(int[] L, int[] P, int l, int m, int r, int[] mark, int sa, int sb) {
        int s1 = mark[L[l]];
        if (s1 != sa && s1 != sb) return false;
        int s2 = s1 == sa ? sb : sa;
        if (mark[L[m + 1]] != s2) return false;
        for (int p = l + 1; p <= m; p++) if (P[p] < l && mark[L[p]] != s1) return false;
        for (int p = m + 2; p <= r; p++) if (P[p] <= m && mark[L[p]] != s2) return false;
        return true;
    }

    private static String newick(ClusterTable ct, StateSpace ss, int[] best, int[] cA, int[] cB, int root, String[] names) {
        // iterative: emit tokens in order
        StringBuilder sb = new StringBuilder();
        ArrayDeque<int[]> st = new ArrayDeque<>();   // {cluster, phase}
        st.push(new int[]{root, 0});
        while (!st.isEmpty()) {
            int[] f = st.pop(); int c = f[0];
            if (c < 0) { sb.append(c == -1 ? ',' : ')'); continue; }
            if (ct.size[c] == 1) { sb.append(quote(names[ct.taxa(c)[0]])); continue; }
            int b = best[ss.state(c)];
            sb.append('(');
            st.push(new int[]{-2}); st.push(new int[]{cB[b], 0}); st.push(new int[]{-1}); st.push(new int[]{cA[b], 0});
        }
        return sb.append(';').toString();
    }

    static String quote(String s) {
        for (int i = 0; i < s.length(); i++) if (",():;[]' \t".indexOf(s.charAt(i)) >= 0) return "'" + s.replace("'", "''") + "'";
        return s;
    }
}
