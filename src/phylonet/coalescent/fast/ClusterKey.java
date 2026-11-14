package phylonet.coalescent.fast;

/** Order-independent signature of a SET of taxa: (sum of h1, xor of h2, set size). */
public record ClusterKey(long h1, long h2, int size) implements Comparable<ClusterKey> {
    ClusterKey union(ClusterKey o) {            // valid only for disjoint sets
        return new ClusterKey(h1 + o.h1, h2 ^ o.h2, size + o.size);
    }
    ClusterKey minus(ClusterKey sub) {          // valid only if sub is a subset
        return new ClusterKey(h1 - sub.h1, h2 ^ sub.h2, size - sub.size);
    }
    @Override public int compareTo(ClusterKey o) {
        if (size != o.size) return Integer.compare(size, o.size);
        if (h1 != o.h1) return Long.compare(h1, o.h1);
        return Long.compare(h2, o.h2);
    }
    static ClusterKey of(int[] taxa) {
        long a = 0, b = 0;
        for (int t : taxa) { a += Hashing.h1(t); b ^= Hashing.h2(t); }
        return new ClusterKey(a, b, taxa.length);
    }
}
