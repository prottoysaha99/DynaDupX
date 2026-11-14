package phylonet.coalescent.fast;

import java.util.ArrayDeque;
import java.util.function.ToIntFunction;

/**
 * Iterative Newick parser for rooted, possibly MULTI-COPY gene trees.
 *
 * Output (ParsedTree):
 *   leaves   : taxon ids in post-order (the leaf array of the tree). A taxon may
 *              occur several times (multi-copy / paralogs).
 *   triples  : for every internal node in post-order, (l, m, r) such that its
 *              left child covers leaves[l..m] and its right child covers leaves[m+1..r].
 *              This is the tuple (i, l1, r1, l2, r2) with l2 = r1 + 1.
 *
 * Polytomies are resolved deterministically as left-deep caterpillars; unary
 * nodes are collapsed. Branch lengths, support values, internal labels and
 * [comments] are ignored. Quoted labels are supported.
 */
final class NewickParser {
    static final class ParsedTree {
        final int[] leaves; final int[] triples; final int polytomies;
        ParsedTree(int[] l, int[] t, int p) { leaves = l; triples = t; polytomies = p; }
        int internalCount() { return triples.length / 3; }
    }

    private final String s; private int i;
    private final IntList left = new IntList(), right = new IntList(), leafId = new IntList();
    private int polytomies = 0;

    private NewickParser(String s) { this.s = s; }

    static ParsedTree parse(String newick, ToIntFunction<String> taxonId) {
        return new NewickParser(newick).run(taxonId);
    }

    private int newLeaf(int taxon) { left.add(-1); right.add(-1); leafId.add(taxon); return left.size() - 1; }
    private int newInternal(int a, int b) { left.add(a); right.add(b); leafId.add(-1); return left.size() - 1; }

    private int close(IntList ch) {
        if (ch.size() == 0) throw new IllegalArgumentException("empty clade '()' in tree");
        if (ch.size() == 1) return ch.get(0);                 // collapse unary node
        if (ch.size() > 2) polytomies++;
        int node = ch.get(0);
        for (int j = 1; j < ch.size(); j++) node = newInternal(node, ch.get(j));
        return node;
    }

    /** Skip internal label / branch length / comments up to the next structural delimiter. */
    private void skipTrailer() {
        int n = s.length();
        while (i < n) {
            char c = s.charAt(i);
            if (c == ',' || c == ')' || c == ';' || c == '(') return;
            if (c == '[') { int e = s.indexOf(']', i); i = e < 0 ? n : e + 1; continue; }
            if (c == '\'') { i++; while (i < n) { if (s.charAt(i) == '\'') { if (i + 1 < n && s.charAt(i + 1) == '\'') i += 2; else { i++; break; } } else i++; } continue; }
            i++;
        }
    }

    private String readLabel() {
        int n = s.length();
        if (s.charAt(i) == '\'') {
            StringBuilder sb = new StringBuilder(); i++;
            while (i < n) {
                char c = s.charAt(i);
                if (c == '\'') { if (i + 1 < n && s.charAt(i + 1) == '\'') { sb.append('\''); i += 2; continue; } i++; break; }
                sb.append(c); i++;
            }
            return sb.toString();
        }
        int st = i;
        while (i < n) {
            char c = s.charAt(i);
            if (c == ',' || c == ')' || c == '(' || c == ';' || c == ':' || c == '[' || Character.isWhitespace(c)) break;
            i++;
        }
        return s.substring(st, i);
    }

    private ParsedTree run(ToIntFunction<String> taxonId) {
        ArrayDeque<IntList> stack = new ArrayDeque<>();
        IntList top = new IntList(2), cur = top;
        int n = s.length();
        while (i < n) {
            char c = s.charAt(i);
            if (c == '(') { stack.push(cur); cur = new IntList(); i++; }
            else if (c == ',') i++;
            else if (c == ')') {
                if (stack.isEmpty()) throw new IllegalArgumentException("unbalanced ')'");
                int node = close(cur); cur = stack.pop(); cur.add(node); i++; skipTrailer();
            }
            else if (c == ';') break;
            else if (Character.isWhitespace(c)) i++;
            else if (c == '[') { int e = s.indexOf(']', i); i = e < 0 ? n : e + 1; }
            else {
                String label = readLabel();
                if (label.isEmpty()) throw new IllegalArgumentException("empty leaf label near position " + i);
                cur.add(newLeaf(taxonId.applyAsInt(label))); skipTrailer();
            }
        }
        if (!stack.isEmpty()) throw new IllegalArgumentException("unbalanced '('");
        if (top.size() != 1) throw new IllegalArgumentException("expected exactly one top-level tree");
        int root = top.get(0);
        return postorder(root);
    }

    private ParsedTree postorder(int root) {
        int total = left.size();
        // Reverse of (node, right, left) pre-order = (left, right, node) post-order.
        int[] order = new int[total]; int on = 0;
        int[] st = new int[total]; int sp = 0; st[sp++] = root;
        while (sp > 0) {
            int u = st[--sp]; order[on++] = u;
            if (left.get(u) >= 0) { st[sp++] = left.get(u); st[sp++] = right.get(u); }
        }
        int[] lo = new int[total], hi = new int[total];
        IntList leaves = new IntList(total / 2 + 1), triples = new IntList(total * 3 / 2 + 1);
        int pos = 0;
        for (int k = on - 1; k >= 0; k--) {
            int u = order[k];
            if (left.get(u) < 0) { lo[u] = hi[u] = pos++; leaves.add(leafId.get(u)); }
            else {
                int a = left.get(u), b = right.get(u);
                lo[u] = lo[a]; hi[u] = hi[b];
                triples.add(lo[a]); triples.add(hi[a]); triples.add(hi[b]);
            }
        }
        return new ParsedTree(leaves.toArray(), triples.toArray(), polytomies);
    }
}
