#!/usr/bin/env python3
"""Random multi-copy MGD datasets: DynaDup-X (fast, default space) vs original DynaDup (-legacy).
Per case:  (1) legacy's own -st reconciliation of the fast tree == fast reported cost
           (2) fast cost vs legacy cost (equal expected in the default search space)
           (3) -xc cost <= default cost"""
import os, random, re, subprocess, sys, tempfile
JAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dynadupx.jar")
A = ["-a", "_[0-9]+$", "/delete/"]

def rand_tree(labels):
    nodes = list(labels)
    while len(nodes) > 1:
        i, j = sorted(random.sample(range(len(nodes)), 2)); b = nodes.pop(j); a = nodes.pop(i); nodes.append(f"({a},{b})")
    return nodes[0]

def gene_tree(n):
    sp = [f"S{i}" for i in range(n)]
    if random.random() < 0.3: sp = random.sample(sp, max(3, n - random.randint(1, 3)))   # incomplete
    labels, cnt = [], {}
    for s in sp + random.choices(sp, k=random.randint(0, n)):                              # extra copies
        cnt[s] = cnt.get(s, 0) + 1; labels.append(f"{s}_{cnt[s]}")
    random.shuffle(labels)
    return rand_tree(labels) + ";"

def cost(args):
    r = subprocess.run(["java", "-jar", JAR, *args], capture_output=True, text=True, timeout=120)
    m = re.search(r"total cost: ([0-9.]+)", r.stderr + r.stdout)
    assert m, r.stderr[-1500:]
    return float(m.group(1))

def score(gt, st):
    r = subprocess.run(["java", "-jar", JAR, "-i", gt, *A, "-st", st], capture_output=True, text=True, timeout=120)
    return float(re.search(r"Total number of duplications is: ([0-9]+)", r.stdout + r.stderr).group(1))

stats = {"cases": 0, "equal_to_legacy": 0, "better_than_legacy": 0, "worse_than_legacy": 0, "xc_improves": 0}
for seed in range(int(os.environ.get("N", 15))):
    random.seed(seed)
    n, k = random.randint(5, 14), random.randint(4, 25)
    d = tempfile.mkdtemp(); gt = f"{d}/g.tre"
    open(gt, "w").write("\n".join(gene_tree(n) for _ in range(k)) + "\n")
    cf = cost(["-i", gt, *A, "-o", f"{d}/f.tre"])
    cl = cost(["-i", gt, *A, "-legacy", "-o", f"{d}/l.tre"])
    cx = cost(["-i", gt, *A, "-xc", "-o", f"{d}/x.tre"])
    sf = score(gt, f"{d}/f.tre")
    assert sf == cf, f"seed {seed}: fast reports {cf} but legacy reconciliation gives {sf}"
    assert score(gt, f"{d}/x.tre") == cx
    assert cx <= cf, f"seed {seed}: -xc {cx} worse than default {cf}"
    stats["cases"] += 1
    stats["equal_to_legacy" if cf == cl else "better_than_legacy" if cf < cl else "worse_than_legacy"] += 1
    stats["xc_improves"] += cx < cf
    print(f"seed {seed}: n={n} k={k} fast={cf} legacy={cl} -xc={cx}", flush=True)
print("RESULT", stats)
