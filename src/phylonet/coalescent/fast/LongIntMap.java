package phylonet.coalescent.fast;

import java.util.Arrays;

/** Allocation-free open-addressing map from non-negative long keys to int values (linear probing, no deletion). */
final class LongIntMap {
    private long[] keys; private int[] vals; private int mask, count;

    LongIntMap(int expected) {
        int cap = Integer.highestOneBit(Math.max(16, 2 * expected) - 1) << 1;
        keys = new long[cap]; Arrays.fill(keys, -1L); vals = new int[cap]; mask = cap - 1;
    }
    private static int home(long key) { long z = key * 0x9E3779B97F4A7C15L; return (int) (z ^ (z >>> 29)); }

    /** The value of key; if absent, inserts (key, v) and returns -1. */
    int getOrPut(long key, int v) {
        int i = home(key) & mask;
        for (long x; (x = keys[i]) != -1L; i = (i + 1) & mask) if (x == key) return vals[i];
        keys[i] = key; vals[i] = v;
        if (4L * ++count > 3L * (mask + 1)) grow();
        return -1;
    }
    private void grow() {
        long[] ok = keys; int[] ov = vals;
        int cap = 2 * (mask + 1);
        keys = new long[cap]; Arrays.fill(keys, -1L); vals = new int[cap]; mask = cap - 1;
        for (int j = 0; j < ok.length; j++) {
            if (ok[j] == -1L) continue;
            int i = home(ok[j]) & mask;
            while (keys[i] != -1L) i = (i + 1) & mask;
            keys[i] = ok[j]; vals[i] = ov[j];
        }
    }
}
