package phylonet.coalescent.fast;

/**
 * Allocation-free open-addressing map (h1, h2, size) -> cluster id, used for O(1) cluster lookups.
 * A slot holds only h1 (8 bytes) and id + 1 (4 bytes); h2 and size are verified in the cluster table
 * (only when h1 matches, i.e. almost only for true hits). A Bloom-style bit filter answers most negative lookups (the
 * common case while enumerating DP resolutions) from a small, cache-resident array without touching the table.
 * Inserts never move existing entries (linear probing, no deletion), so lookups of existing keys are unaffected by
 * later inserts.
 */
final class KeyIndex {
    private final ClusterTable ct;
    private long[] keyH1; private int[] slotId; private int mask, count;
    private long[] bloom; private int bloomMask;

    KeyIndex(ClusterTable ct, int expected) {
        this.ct = ct;
        int cap = Integer.highestOneBit(Math.max(16, expected * 2) - 1) << 1;
        keyH1 = new long[cap]; slotId = new int[cap]; mask = cap - 1;
        newBloom(Math.max(expected, 1024));
    }
    private static long mix(long h1, long h2, int size) {
        long z = h1 ^ (h2 * 0x9E3779B97F4A7C15L) ^ ((long) size * 0xC2B2AE3D27D4EB4FL);
        z ^= z >>> 31; z *= 0xBF58476D1CE4E5B9L; z ^= z >>> 29;
        return z;
    }
    /** Slot position from h1 only (h2/size are not stored, so rehashing needs only h1). */
    private static int home(long h1) { long z = h1 * 0x9E3779B97F4A7C15L; z ^= z >>> 29; return (int) (z >>> 17); }
    private void newBloom(int expected) {
        long bits = Long.highestOneBit(Math.max(1L << 16, 16L * expected) - 1) << 1;   // >= 16 bits per key, 2 probes
        bits = Math.min(bits, 1L << 34);
        bloom = new long[(int) (bits >>> 6)]; bloomMask = (int) ((bits >>> 6) - 1);
    }
    private void bloomAdd(long z) {
        int w1 = (int) (z >>> 6) & bloomMask, w2 = (int) (z >>> 38) & bloomMask;
        bloom[w1] |= 1L << (z & 63); bloom[w2] |= 1L << ((z >>> 32) & 63);
    }
    /** false = certainly absent; true = maybe present. */
    boolean mightContain(long h1, long h2, int size) {
        long z = mix(h1, h2, size);
        return (bloom[(int) (z >>> 6) & bloomMask] >>> (z & 63) & 1L) != 0 && (bloom[(int) (z >>> 38) & bloomMask] >>> ((z >>> 32) & 63) & 1L) != 0;
    }

    int get(long h1, long h2, int size) {
        long z = mix(h1, h2, size);
        final long[] bl = bloom; final int bm = bloomMask;
        if ((bl[(int) (z >>> 6) & bm] >>> (z & 63) & 1L) == 0 || (bl[(int) (z >>> 38) & bm] >>> ((z >>> 32) & 63) & 1L) == 0) return -1;
        final long[] kh = keyH1; final int[] sid = slotId; final int msk = kh.length - 1;
        int i = home(h1) & msk;
        while (true) {
            int id = sid[i] - 1;
            if (id < 0) return -1;
            if (kh[i] == h1 && ct.h2[id] == h2 && ct.size[id] == size) return id;
            i = (i + 1) & msk;
        }
    }
    void put(long h1, long h2, int size, int id) {
        if (4L * (count + 1) > 3L * (mask + 1)) grow();          // load factor <= 3/4
        if (16L * (count + 1) > 64L * bloom.length) rebuildBloom(2 * (count + 1));
        int i = home(h1) & mask;
        while (slotId[i] != 0) {
            int o = slotId[i] - 1;
            if (keyH1[i] == h1 && ct.h2[o] == h2 && ct.size[o] == size) { slotId[i] = id + 1; return; }
            i = (i + 1) & mask;
        }
        keyH1[i] = h1; slotId[i] = id + 1; count++;
        bloomAdd(mix(h1, h2, size));
    }
    private void grow() {
        long[] oldH = keyH1; int[] oldId = slotId;
        int cap = 2 * (mask + 1);
        long[] nh = new long[cap]; int[] ni = new int[cap]; int nm = cap - 1;
        for (int j = 0; j < oldH.length; j++) {
            if (oldId[j] == 0) continue;
            int i = home(oldH[j]) & nm;
            while (ni[i] != 0) i = (i + 1) & nm;
            nh[i] = oldH[j]; ni[i] = oldId[j];
        }
        keyH1 = nh; slotId = ni; mask = nm;
    }
    private void rebuildBloom(int expected) {
        newBloom(expected);
        for (int i = 0; i < keyH1.length; i++) if (slotId[i] != 0) { int id = slotId[i] - 1; bloomAdd(mix(keyH1[i], ct.h2[id], ct.size[id])); }
    }
}
