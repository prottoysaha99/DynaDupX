#!/usr/bin/env python3
"""Fuzz test for tools/root_gene_trees.py.

Generates many random and degenerate gene trees (1-3 leaves, unary nodes, zero/missing branch lengths, support values,
[comments], quoted labels, polytomies, multi-copy species, trees with only / without outgroup copies) and checks for
every tree and every rooting mode:
  1. no exception;
  2. the output is a rooted binary-at-the-root Newick (exactly 2 children at the top, or a single leaf);
  3. the multiset of leaf labels is unchanged;
  4. the unrooted topology (set of non-trivial splits) is unchanged, after suppressing unary nodes;
  5. the total branch length is unchanged (up to rounding);
  6. the root is in the right place: --outgroup -> no other edge misplaces fewer leaves; --midpoint -> the two
     sides of the root reach equally far (the root is the middle of the longest leaf-to-leaf path).
Also a messy-file test (glued trees, multi-line trees, CRLF, missing ';', malformed trees).
The checker has its own Newick parser (independent from the script's).
  python3 test/fuzz_root_gene_trees.py [n_random_trees]
"""
import os, random, re, subprocess, sys, tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "tools", "root_gene_trees.py")

# ------------------------------------------------------------------ independent recursive parser
class N:
    __slots__ = ("ch", "lab", "len")
    def __init__(s): s.ch, s.lab, s.len = [], None, None

def parse(s):
    s = re.sub(r"\[[^\]]*\]", "", s.strip())
    if s.endswith(";"): s = s[:-1]
    pos = [0]
    def name():
        i = pos[0]
        if i < len(s) and s[i] == "'":
            j = s.index("'", i + 1); pos[0] = j + 1; return s[i:j + 1]
        while pos[0] < len(s) and s[pos[0]] not in ",():;": pos[0] += 1
        return s[i:pos[0]].strip()
    def node():
        n = N()
        if s[pos[0]] == "(":
            pos[0] += 1
            n.ch.append(node())
            while s[pos[0]] == ",": pos[0] += 1; n.ch.append(node())
            assert s[pos[0]] == ")"; pos[0] += 1
            name()                                   # internal label dropped
        else:
            n.lab = name()
        if pos[0] < len(s) and s[pos[0]] == ":":
            pos[0] += 1; i = pos[0]
            while pos[0] < len(s) and s[pos[0]] not in ",()": pos[0] += 1
            try: n.len = float(s[i:pos[0]])
            except ValueError: n.len = None
        return n
    t = node()
    assert pos[0] == len(s), "trailing text: %r" % s[pos[0]:]
    return t

def leaves(t):
    out, st = [], [t]
    while st:
        u = st.pop()
        if not u.ch: out.append(u.lab)
        st.extend(u.ch)
    return out

def total_len(t, top=True):
    if top:                          # a stem above a unary root leads to no leaf: it is not a branch of the unrooted tree
        while len(t.ch) == 1 and t.ch[0].ch: t = t.ch[0]
        if len(t.ch) == 1: return 0.0
    return (0.0 if top or t.len is None else t.len) + sum(total_len(c, False) for c in t.ch)

def splits(t):
    """Non-trivial unrooted splits, each as the frozenset of leaf *ids* on the side not containing leaf id 0."""
    ids, lab = {}, []
    def number(u):
        if not u.ch: ids[id(u)] = len(lab); lab.append(u.lab)
        for c in u.ch: number(c)
    number(t)
    n = len(lab); allset = frozenset(range(n)); out = set()
    def below(u):
        if not u.ch: return frozenset([ids[id(u)]])
        return frozenset().union(*[below(c) for c in u.ch]) if u.ch else frozenset()
    def rec(u):
        b = below(u)
        if 1 < len(b) < n - 1:
            out.add(b if 0 not in b else allset - b)
        for c in u.ch: rec(c)
    rec(t)
    # express splits with labels + duplicates disambiguated by the order-independent multiset of each side
    return out, lab

def canon_splits(t):
    """Order-independent split set: each split as a pair of sorted label multisets (smaller side first)."""
    s, lab = splits(t)
    res = Counter()
    n = len(lab)
    for b in s:
        a = tuple(sorted(lab[i] for i in b)); c = tuple(sorted(lab[i] for i in range(n) if i not in b))
        res[min(a, c) + ("|",) + max(a, c)] += 1
    return res

# ------------------------------------------------------------------ random tree generator
def rand_label(rng, sp, copy, quote):
    l = "%d_%d_0" % (sp, copy)
    return "'%s'" % l if quote else l

def rand_tree(rng):
    k = rng.choice([1, 2, 2, 3, 3, 4, 5, 6, 8, 12, 20, 40])
    nsp = rng.randint(1, max(1, k))
    outmode = rng.choice(["none", "some", "all", "one"])
    labs, cnt = [], Counter()
    for i in range(k):
        if outmode == "all" or (outmode == "one" and i == 0) or (outmode == "some" and rng.random() < 0.3): sp = 0
        else: sp = rng.randint(1, nsp)
        labs.append((sp, cnt[sp])); cnt[sp] += 1
    rng.shuffle(labs)
    quote = rng.random() < 0.1
    lenmode = rng.choice(["normal", "zero", "missing", "mixed", "tiny", "sci"])
    def blen():
        if lenmode == "normal": return ":%.6f" % rng.expovariate(10)
        if lenmode == "zero": return ":0.0"
        if lenmode == "missing": return ""
        if lenmode == "tiny": return ":%.3e" % (rng.random() * 1e-9)
        if lenmode == "sci": return ":%.4e" % rng.expovariate(1)
        return rng.choice([":0", "", ":%.5f" % rng.random()])
    def sup():
        r = rng.random()
        return "" if r < 0.6 else ("%.3f" % rng.random() if r < 0.8 else ("[&c=1]" if r < 0.9 else "%d" % rng.randint(0, 100)))
    nodes = [rand_label(rng, sp, c, quote) + blen() for sp, c in labs]
    # random agglomeration; allow polytomies (3-4 way) and unary wrappers
    while len(nodes) > 1:
        r = rng.random()
        m = 2 if r < 0.75 or len(nodes) < 3 else min(len(nodes), rng.choice([3, 4]))
        if len(nodes) == m: top = True
        else: top = False
        pick = rng.sample(range(len(nodes)), m)
        grp = [nodes[i] for i in pick]
        for i in sorted(pick, reverse=True): nodes.pop(i)
        s = "(" + ",".join(grp) + ")" + sup()
        if rng.random() < 0.05: s = "(" + s + blen() + ")"          # unary node
        nodes.append(s + ("" if top else blen()))
    t = nodes[0]
    if k == 1 and rng.random() < 0.5: t = "(" + t + ")"
    # unrooted style: make the top a trifurcation sometimes (already random)
    return t + ";"

EXTRA = [
    "A;", "'A';", "(A);", "((A:1):2);", "(A:1,B:2);", "(0_0_0:0.0,0_1_0:0.0);", "(479_1_0:1.035251981,0_0_0:1.035251981);",
    "((A,B));", "(A,B,C);", "(A:0,B:0,C:0);", "((A:1,B:1):0,C:1);", "(0_0_0,0_1_0,0_2_0);", "(0_0_0:1,(1_0_0:1,2_0_0:1):1,3_0_0:1);",
    "((0_0_0:1,0_1_0:1)0.9:1,(1_0_0:1,2_0_0:1)1:1);", "(((((A:1)))),B:4);",
    "(A:1[&x],B:2)[&R];", "[&U](A,B,(C,D));", "(A:1e-3,B:2E+1,C:0.5);", "((A,B)'lab':1,(C,D)99:2);",
    "(0_0_0:5,(1_0_0:1,(2_0_0:1,0_1_0:1):1):1,3_0_0:1);", "((((0_0_0:1,1_0_0:1):1,2_0_0:1):1,3_0_0:1):1,0_1_0:1);",
]

def run(trees, mode):
    with tempfile.TemporaryDirectory() as d:
        fi, fo = os.path.join(d, "in"), os.path.join(d, "out")
        open(fi, "w").write("\n".join(trees) + "\n")
        args = [sys.executable, SCRIPT, fi, fo] + (["--midpoint"] if mode == "mid" else ["--outgroup", "0"])
        p = subprocess.run(args, capture_output=True, text=True)
        if p.returncode != 0: return None, p.stderr
        return open(fo).read().split("\n")[:-1], p.stderr

def strip_q(l): return l[1:-1] if l and l[0] == "'" and l[-1] == "'" else l

def is_out(l): return re.sub(r"_[0-9]+_[0-9]+$", "", strip_q(l)) == "0"

def counts(u):
    """(#outgroup leaves, #other leaves) below u"""
    if not u.ch: return (1, 0) if is_out(u.lab) else (0, 1)
    o = g = 0
    for c in u.ch: a, b = counts(c); o += a; g += b
    return o, g

def depth_max(u):
    if not u.ch: return 0.0
    return max((c.len or 0.0) + depth_max(c) for c in u.ch)

def check_position(a, b, mode):
    """midpoint: both root sides reach equally far; outgroup: the root edge has the fewest misplaced leaves."""
    nout, nin = counts(a)
    if nout + nin <= 2: return None
    if mode == "og" and 0 < nout < nout + nin:
        best, st = None, [a]
        while st:
            u = st.pop(); st.extend(u.ch)
            if u is a: continue
            o, g = counts(u); e = min(g + (nout - o), (nin - g) + o)
            best = e if best is None else min(best, e)
        (o1, g1), (o2, g2) = counts(b.ch[0]), counts(b.ch[1])
        e = min(g1 + o2, g2 + o1)
        return None if e == best else "outgroup root has %d misplaced leaves, best edge has %d" % (e, best)
    l = (b.ch[0].len or 0.0) + depth_max(b.ch[0]); r = (b.ch[1].len or 0.0) + depth_max(b.ch[1])
    if l + r == 0.0: return None
    return None if abs(l - r) <= 1e-6 * max(1.0, l + r) else "midpoint: sides reach %g and %g" % (l, r)

def check(inp, out, mode):
    a, b = parse(inp), parse(out)
    la, lb = Counter(map(strip_q, leaves(a))), Counter(map(strip_q, leaves(b)))
    if la != lb: return "leaf multiset changed"
    if sum(la.values()) >= 2 and len(b.ch) != 2: return "root of output has %d children" % len(b.ch)
    if sum(la.values()) == 1 and b.ch: return "one-leaf tree not written as a leaf"
    # every internal non-root node of the output must be at least binary (unary suppressed)
    st = list(b.ch)
    while st:
        u = st.pop()
        if len(u.ch) == 1: return "unary node in output"
        st.extend(u.ch)
    if canon_splits(a) != canon_splits(b): return "unrooted topology changed"
    if sum(la.values()) == 1: return None
    ta, tb = total_len(a), total_len(b)
    if abs(ta - tb) > 1e-6 * max(1.0, abs(ta)): return "total length %g -> %g" % (ta, tb)
    return check_position(a, b, mode)

def file_format_test(trees, rng):
    """Messy files: trees glued on one line, a tree over several lines, CRLF, blank lines, missing last ';',
    ';' inside [comments] / 'quotes'. The result must equal rooting one clean tree per line."""
    clean, _ = run(trees, "og")
    parts = []
    for t in trees:
        r = rng.random()
        if r < 0.15: parts.append(t)                                     # glued to the next tree
        elif r < 0.25 and "," in t:
            i = t.index(","); parts.append(t[:i + 1] + "\n" + t[i + 1:] + "\n")   # broken over two lines
        elif r < 0.30: parts.append(t + "\r\n\n")                        # CRLF + blank line
        elif r < 0.35: parts.append("[&R;x] " + t + "\n")                  # comment with a ';' in front
        else: parts.append(t + "\n")
    messy = "".join(parts).rstrip()
    if messy.endswith(";"): messy = messy[:-1]                          # last ';' missing
    extra = ["(A:1,'x;y':2,B:3);"]
    with tempfile.TemporaryDirectory() as d:
        fi, fo = os.path.join(d, "in"), os.path.join(d, "out")
        open(fi, "w").write("\n".join(extra) + "\n" + messy + "\n")
        p = subprocess.run([sys.executable, SCRIPT, fi, fo, "--outgroup", "0"], capture_output=True, text=True)
        got = open(fo).read().split("\n")[:-1] if p.returncode == 0 else None
        e1, _ = run(extra, "og")
        # malformed trees must give a clean message naming the tree, not a traceback
        msgs = []
        for badtree in ["((A,B),(C,D);", "(A,B));", "(A,B)(C,D);"]:
            open(fi, "w").write("(A,B,C);\n" + badtree + "\n")
            q = subprocess.run([sys.executable, SCRIPT, fi, fo, "--midpoint"], capture_output=True, text=True)
            msgs.append(q.returncode != 0 and "tree 2 (line 2)" in q.stderr and "Traceback" not in q.stderr)
    ok = got is not None and got == e1 + clean and all(msgs)
    print("file format test: %s   [%s]" % ("ok" if ok else "PROBLEM", p.stderr.strip().split("\n")[0][:150]))
    if not ok and got is not None:
        print("  %d trees expected, %d written; malformed-tree messages ok: %s" % (len(clean) + 1, len(got), msgs))
    return 0 if ok else 1

def main():
    nrand = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rng = random.Random(12345)
    trees = EXTRA + [rand_tree(rng) for _ in range(nrand)]
    bad = 0
    for mode in ("og", "mid"):
        out, err = run(trees, mode)
        if out is None: print("MODE %s: script failed:\n%s" % (mode, err)); bad += 1; continue
        if len(out) != len(trees): print("MODE %s: %d trees in, %d out" % (mode, len(trees), len(out))); bad += 1; continue
        nb = 0
        for i, (x, y) in enumerate(zip(trees, out), 1):
            try: why = check(x, y, mode)
            except Exception as e: why = "checker error %s: %s" % (type(e).__name__, e)
            if why:
                nb += 1
                if nb <= 10: print("MODE %s tree %d: %s\n  in : %s\n  out: %s" % (mode, i, why, x[:300], y[:300]))
        print("mode %-3s: %d trees, %d problems   [%s]" % (mode, len(trees), nb, err.strip()))
        bad += nb
    bad += file_format_test(trees[:400], rng)
    print("ALL OK" if bad == 0 else "FAILED (%d)" % bad)
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
