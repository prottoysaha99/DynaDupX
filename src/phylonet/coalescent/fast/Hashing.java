package phylonet.coalescent.fast;

/**
 * Per-taxon hashing. Two independently seeded 64-bit mixers feed the two
 * permutation-invariant set hashes (phi1 = sum mod 2^64, phi2 = XOR).
 * Using different seeds for phi1 and phi2 makes the independence assumption
 * of the collision bound (see the manuscript, Methods) hold by construction.
 *
 * IMPORTANT (multicopy): these set hashes are only correct over SETS of taxa.
 * Over a multiset, XOR cancels duplicate copies and SUM double counts them, so
 * callers must feed each distinct taxon exactly once (see GeneScan).
 */
final class Hashing {
    private Hashing() {}

    private static long fmix64(long z) {
        z = (z ^ (z >>> 33)) * 0xff51afd7ed558ccdL;
        z = (z ^ (z >>> 33)) * 0xc4ceb9fe1a85ec53L;
        z ^= (z >>> 33);
        return z == 0 ? 1 : z;
    }

    static long h1(int id) { return fmix64((id + 1L) * 0x9E3779B97F4A7C15L + 0x243F6A8885A308D3L); }
    static long h2(int id) { return fmix64((id + 1L) * 0xC2B2AE3D27D4EB4FL + 0x13198A2E03707344L); }
}
