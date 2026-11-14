package phylonet.coalescent.fast;

import java.util.Arrays;

final class IntList {
    int[] a; int n;
    IntList() { a = new int[8]; }
    IntList(int cap) { a = new int[Math.max(cap, 1)]; }
    IntList(int[] src) { a = src.length == 0 ? new int[1] : src.clone(); n = src.length; }
    void add(int v) { if (n == a.length) a = Arrays.copyOf(a, n < 1 << 16 ? 2 * n : n + (n >> 1)); a[n++] = v; }
    int get(int i) { return a[i]; }
    void set(int i, int v) { a[i] = v; }
    int size() { return n; }
    int[] toArray() { return Arrays.copyOf(a, n); }
    /** The backing array trimmed to the size (shared, no second copy kept). */
    int[] trimmed() { if (a.length != n) a = Arrays.copyOf(a, n); return a; }
}
