#!/usr/bin/env python3
"""Root gene trees before running DynaDup-X (MGD/MGDL/MGDLX need ROOTED gene trees).

Estimated gene trees (FastTree, RAxML, IQ-TREE ...) are unrooted: their Newick has a 3-way split at the top.
DynaDup-X would otherwise turn that 3-way split into an arbitrary 2-way root.

  python3 root_gene_trees.py in.trees out.trees --outgroup 0 [--map "_[0-9]+_[0-9]+$"]   # outgroup species
  python3 root_gene_trees.py in.trees out.trees --midpoint                               # midpoint rooting

--outgroup S : root on the edge that best separates the copies of species S (multi-copy aware): the edge whose
               lower side X minimises (#non-S leaves in X) + (#S leaves outside X); if all copies of S form one clade
               this is exactly the edge above that clade. Trees without S are midpoint-rooted (reported).
--midpoint   : midpoint of the longest leaf-to-leaf path (if all branch lengths on it are 0, its middle edge).
Small trees: a tree with one leaf is written unchanged; a tree with two leaves is rooted on the edge between them.
Unary nodes are removed (their branch lengths are added up), internal labels / support values are dropped.
Leaf labels are mapped to species with the regular expression --map (default: SimPhy style "_[0-9]+_[0-9]+$" is
deleted). Branch lengths are kept (the rooted edge is split in two). Output: one rooted tree per line, same order as the input.
Trees are read up to each ';', so two trees glued on one line (a missing line break) are both rooted (reported).
"""
import argparse, re, sys

_TOKL = re.compile(r"\(|\)|,|;|:[^,();\[]*|\[[^\]]*\]|'[^']*'|[^,();:\[\]']+")

def parse(s):
    """-> (children, label, length); node 0 is the root. Internal labels are dropped (label None)."""
    children, label, length = [[]], [None], [None]
    stack, last, first = [], None, True
    for t in _TOKL.findall(s.strip()):
        c = t[0]
        if c == "(":
            if first: stack.append(0); first = False
            elif not stack: raise ValueError("text after the end of the tree (two trees without ';' between them?)")
            else:
                children.append([]); label.append(None); length.append(None)
                v = len(children) - 1; children[stack[-1]].append(v); stack.append(v)
            last = None
        elif c == ")":
            if not stack: raise ValueError("unbalanced parentheses: a ')' without its '('")
            last = stack.pop()
        elif c == ",": last = None
        elif c == ":":
            if last is not None and t[1:].strip():
                try: length[last] = float(t[1:])
                except ValueError: pass
        elif c in ";[": continue
        else:
            if t.strip() == "": continue
            if last is not None: continue      # internal label / support value after ")": dropped; its ":length" follows
            if not stack: continue
            children.append([]); label.append(t.strip()); length.append(None)
            v = len(children) - 1; children[stack[-1]].append(v); last = v
    if stack: raise ValueError("unbalanced parentheses: %d '(' never closed (tree cut off?)" % len(stack))
    return children, label, length

def read_trees(f):
    """Yield (line number, Newick) for every tree, split at ';' (outside 'quotes' and [comments]), so two trees
    glued on one line become two trees and a tree spread over several lines becomes one tree."""
    buf, first, q, br = [], None, False, False
    for ln, line in enumerate(f, 1):
        line = line.rstrip("\r\n")
        if not q and not br and "'" not in line and "[" not in line:          # fast path
            parts = line.split(";")
            for piece in parts[:-1]:
                buf.append(piece); t = "".join(buf).strip()
                if t: yield (first or ln), t + ";"
                buf, first = [], None
            if parts[-1].strip(): buf.append(parts[-1]); first = first or ln
            continue
        st = 0
        for k, c in enumerate(line):
            if q:
                if c == "'": q = False
            elif br:
                if c == "]": br = False
            elif c == "'": q = True
            elif c == "[": br = True
            elif c == ";":
                buf.append(line[st:k]); t = "".join(buf).strip()
                if t: yield (first or ln), t + ";"
                buf, first, st = [], None, k + 1
        if line[st:].strip() or q or br: buf.append(line[st:]); first = first or ln
    t = "".join(buf).strip()
    if t: yield first, t                                                     # last tree without ';'

def undirected(children, label, length):
    """Unrooted adjacency {neighbour: branch length}; unlabeled dead ends are removed and unlabeled degree-2 nodes
    (the old root, unary nodes) are suppressed by adding up their two branch lengths."""
    adj = [dict() for _ in children]
    for u, ch in enumerate(children):
        for v in ch:
            w = length[v] if length[v] is not None else 0.0
            adj[u][v] = w; adj[v][u] = w
    changed = True
    while changed:
        changed = False
        for u in range(len(adj)):
            if label[u] is not None: continue
            d = len(adj[u])
            if d == 1:
                (v, _), = adj[u].items(); del adj[v][u]; adj[u] = {}; changed = True
            elif d == 2:
                (a, wa), (b, wb) = adj[u].items()
                del adj[a][u]; del adj[b][u]; adj[u] = {}
                adj[a][b] = wa + wb; adj[b][a] = wa + wb; changed = True
    return adj

def write(adj, label, root, parent):
    out, st = [], [(root, parent, 0)]
    while st:
        u, p, s = st.pop()
        if s == 2: out.append(","); continue
        if s == 0:
            ch = [v for v in adj[u] if v != p]
            if ch:
                st.append((u, p, 1)); out.append("(")
                for i, v in enumerate(reversed(ch)):
                    st.append((v, u, 0))
                    if i < len(ch) - 1: st.append((None, None, 2))
                continue
            out.append(label[u])
        else:
            out.append(")")
        if p is not None and p >= 0: out.append(":%.10g" % adj[u][p])
    return "".join(out)

def root_on_edge(adj, label, a, b, frac=0.5):
    w = adj[a][b]
    del adj[a][b]; del adj[b][a]
    s = "(%s:%.10g,%s:%.10g);" % (write(adj, label, a, -1), w * frac, write(adj, label, b, -1), w * (1 - frac))
    adj[a][b] = w; adj[b][a] = w
    return s

def leaves_of(adj, label):
    return [u for u in range(len(adj)) if label[u] is not None and len(adj[u]) <= 1]

def midpoint(adj, label):
    """Midpoint of the longest leaf-to-leaf path. If every branch on that path has length 0 (or lengths are missing),
    the middle edge of the path with the most edges is used instead."""
    leaves = leaves_of(adj, label)
    def far(src, hops):
        dist, prev, st = {src: 0.0}, {src: -1}, [src]
        while st:
            u = st.pop()
            for v, w in adj[u].items():
                if v not in dist: dist[v] = dist[u] + (1.0 if hops else max(w, 0.0)); prev[v] = u; st.append(v)
        return max(leaves, key=lambda x: dist[x]), dist, prev
    hops = False
    b, _, _ = far(leaves[0], False); c, dist, prev = far(b, False)
    if dist[c] <= 0.0:                                          # all lengths zero: use the number of edges
        hops = True
        b, _, _ = far(leaves[0], True); c, dist, prev = far(b, True)
    half = dist[c] / 2.0; u = c
    while prev[u] != -1 and dist[prev[u]] > half: u = prev[u]
    p = prev[u]
    if hops: return root_on_edge(adj, label, u, p, 0.5)
    w = max(adj[u][p], 0.0)
    return root_on_edge(adj, label, u, p, min(1.0, max(0.0, (dist[u] - half) / w)) if w > 0 else 0.5)

def outgroup_root(adj, label, is_out):
    leaves = leaves_of(adj, label)
    nout = sum(1 for u in leaves if is_out(label[u]))               # caller guarantees 0 < nout < #leaves, #leaves >= 3
    anchor = next(u for u in range(len(adj)) if len(adj[u]) >= 3)   # exists: >= 3 leaves and no degree-2 nodes
    order, par, st = [], {anchor: -1}, [anchor]
    while st:
        u = st.pop(); order.append(u)
        for v in adj[u]:
            if v != par[u]: par[v] = u; st.append(v)
    o, g = {}, {}
    for u in reversed(order):
        if label[u] is not None and len(adj[u]) <= 1: o[u] = 1 if is_out(label[u]) else 0; g[u] = 1 - o[u]
        else:
            o[u] = sum(o[v] for v in adj[u] if v != par[u]); g[u] = sum(g[v] for v in adj[u] if v != par[u])
    nin = len(leaves) - nout
    best = None
    for u in order:
        if par[u] == -1: continue
        # outgroup below u: errors = ingroup below + outgroup above ; outgroup above u: the mirror
        e = min(g[u] + (nout - o[u]), (nin - g[u]) + o[u])
        if best is None or e < best[0]: best = (e, u, par[u])
    return root_on_edge(adj, label, best[1], best[2])

def root_line(line, is_out):
    """Rooted Newick for one input tree; returns (newick, how) with how in {outgroup, midpoint, single, pair}."""
    line = line.strip()
    if "(" not in line:                                        # a one-leaf tree has nothing to root
        return (line if line.endswith(";") else line + ";"), "single"
    ch, lab, ln = parse(line)
    adj = undirected(ch, lab, ln)
    leaves = leaves_of(adj, lab)
    if len(leaves) == 0: return line, "single"
    if len(leaves) == 1: return lab[leaves[0]] + ";", "single"
    if len(leaves) == 2:                                        # the two leaves are joined by one edge
        a, b = leaves
        return root_on_edge(adj, lab, a, b, 0.5), "pair"
    if is_out is not None:
        nout = sum(1 for u in leaves if is_out(lab[u]))
        if nout == 0: return midpoint(adj, lab), "mid_none"               # no copy of the outgroup
        if nout == len(leaves): return midpoint(adj, lab), "mid_all"      # only copies of the outgroup
        return outgroup_root(adj, lab, is_out), "outgroup"
    return midpoint(adj, lab), "midpoint"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--outgroup"); g.add_argument("--midpoint", action="store_true")
    ap.add_argument("--map", default=r"_[0-9]+_[0-9]+$", help="regex deleted from a leaf label to get its species")
    a = ap.parse_args()
    rx = re.compile(a.map)
    unq = lambda l: l[1:-1] if len(l) >= 2 and l[0] == l[-1] == "'" else l    # 'quoted' labels
    is_out = (lambda l: rx.sub("", unq(l)) == a.outgroup) if a.outgroup is not None else None
    counts = {"outgroup": 0, "midpoint": 0, "mid_none": 0, "mid_all": 0, "single": 0, "pair": 0}
    per_line = {}
    with open(a.inp) as fi, open(a.out, "w") as fo:
        for i, (ln, tree) in enumerate(read_trees(fi), 1):
            per_line[ln] = per_line.get(ln, 0) + 1
            try:
                r, how = root_line(tree, is_out)
            except Exception as e:
                sys.exit("tree %d (line %d) could not be rooted (%s: %s): %s" % (
                    i, ln, type(e).__name__, e, tree[:200]))
            fo.write(r + "\n"); counts[how] += 1
    glued = sorted(l for l, c in per_line.items() if c > 1)
    if glued:
        print("note: %d line(s) of the input held more than one tree (a missing line break), e.g. line %s; "
              "each tree is written on its own line" % (len(glued), ", ".join(map(str, glued[:5]))), file=sys.stderr)
    n = sum(counts.values())
    if is_out is not None:
        msg = ("rooted %d trees: %d by the outgroup, %d midpoint (no copy of %s), %d midpoint (only copies of %s), "
               "%d with two leaves (rooted on their one edge), %d with one leaf" % (n, counts["outgroup"],
               counts["mid_none"], a.outgroup, counts["mid_all"], a.outgroup, counts["pair"], counts["single"]))
    else:
        msg = "rooted %d trees: %d midpoint, %d with two leaves (rooted on their one edge), %d with one leaf" % (
            n, counts["midpoint"], counts["pair"], counts["single"])
    print(msg, file=sys.stderr)

if __name__ == "__main__":
    main()
