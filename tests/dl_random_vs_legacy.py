#!/usr/bin/env python3
"""Random multi-copy MGDL datasets: DynaDup-X vs original DynaDup (-legacy), wd in {0, 1, 0.5}.
fast(-x0) = same search space as legacy (equal cost expected, or lower where legacy hits its -1 sentinel bug);
-st recount = legacy reconciliation; extended = default MGDL search space (never worse than -x0)."""
import os, random, re, subprocess, tempfile, sys

JAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dynadupx.jar")
FAST = ["java", "-Xmx400m", "-jar", JAR]
LEG = ["java", "-Xmx400m", "-jar", JAR, "-legacy"]
A = ["-a", "_[0-9]+$", "/delete/"]
def rand_tree(labels):
    nodes = list(labels)
    while len(nodes) > 1:
        i, j = sorted(random.sample(range(len(nodes)), 2)); b = nodes.pop(j); a = nodes.pop(i); nodes.append(f"({a},{b})")
    return nodes[0]
def gene_tree(n, inc):
    sp = [f"S{i}" for i in range(n)]
    if random.random() < inc: sp = random.sample(sp, max(3, n - random.randint(1, 3)))
    labels, cnt = [], {}
    for s in sp + random.choices(sp, k=random.randint(0, n)):
        cnt[s] = cnt.get(s, 0) + 1; labels.append(f"{s}_{cnt[s]}")
    random.shuffle(labels); return rand_tree(labels) + ";"
def cost(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=200)
    m = re.search(r"total cost: (-?[0-9.]+)", r.stderr + r.stdout); assert m, (r.stderr + r.stdout)[-800:]
    return float(m.group(1))
def stscore(gt, st, wd):
    r = subprocess.run(LEG + ["-i", gt, *A, "-st", st], capture_output=True, text=True, timeout=200).stdout
    d = int(re.search(r"duplications is: (\d+)", r).group(1)); lb = int(re.search(r"losses \(bd\) is: (\d+)", r).group(1)); ls = int(re.search(r"losses \(std\) is: (\d+)", r).group(1))
    return d + ls + wd * (lb - ls)
res = []
for seed in range(int(os.environ.get("N", 10))):
    random.seed(seed); n, k = random.randint(5, 10), random.randint(4, 12); inc = [0.0, 0.4][seed % 2]
    d = tempfile.mkdtemp(); gt = f"{d}/g.tre"; open(gt, "w").write("\n".join(gene_tree(n, inc) for _ in range(k)) + "\n")
    for wd in (0, 1, 0.5):
        cf = cost(FAST + ["-i", gt, *A, "-x0", "-dl", str(wd), "-o", f"{d}/f.tre"])
        cl = cost(LEG + ["-i", gt, *A, "-dl", str(wd), "-o", f"{d}/l.tre"])
        sf, sl = stscore(gt, f"{d}/f.tre", wd), stscore(gt, f"{d}/l.tre", wd)
        cx = cost(FAST + ["-i", gt, *A, "-dl", str(wd), "-o", f"{d}/x.tre"])
        res.append((seed, n, k, inc, wd, cf, cl, sf, sl, cx))
        print(f"seed {seed} n={n} k={k} incomplete={inc} wd={wd}: fast(-x0)={cf} legacy={cl} | -st recount fast={sf} legacy={sl} | fast(extended)={cx}", flush=True)
print("equal fast/legacy DP cost:", sum(r[5] == r[6] for r in res), "/", len(res))
print("DP cost == -st recount (fast):", sum(abs(r[5] - r[7]) < 1e-6 for r in res), "/", len(res), " (legacy):", sum(abs(r[6] - r[8]) < 1e-6 for r in res))
print("extended <= -x0:", sum(r[9] <= r[5] + 1e-9 for r in res), "/", len(res))
