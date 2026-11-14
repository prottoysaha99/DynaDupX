# DynaDup-X

DynaDup-X infers a rooted species tree from rooted **multi-copy gene family trees** by gene tree parsimony. It solves, exactly within a constrained search space, the problems

* **MGD** – minimize the number of gene duplications,
* **MGDL** – minimize the number of gene duplications and losses (restriction-based losses, missing-taxon losses, or a weighted mixture of both),
* **weighted MGDL / MGDLX** – minimize duplications + λ · extra lineages + μ · (missing-taxon term).

DynaDup-X uses the subtree-bipartition formulation and the dynamic programming (DP) algorithm of DynaDup (Bayzid, Mirarab and Warnow, PSB 2013). It replaces the set-based data structures of the original implementation with compact encodings and hashing of clusters, a rarest-species index for enumerating the DP state space, and bit-parallel computation of the subtree-bipartition weights. With these changes it handles inputs with 100,000 species or 100,000 gene trees on a single machine.

Version: **1.3.3**

---

## Requirements

* Java Development Kit **17 or newer** (to build and to run)
* Linux, macOS or Windows
* Optional: an NVIDIA GPU with the CUDA toolkit for the `-gpu` back end (Linux)

## Building

```sh
./build.sh
```

This creates `dynadupx.jar` in the repository root. The jar reads the original DynaDup/PhyloNet classes from `lib/main.jar` and `lib/jsr166.jar`, so keep the `lib/` folder next to the jar.

## Quick start

```sh
java -jar dynadupx.jar -i examples/genes.tre -a "_.*" /delete/ -o species.tre
```

Expected output (standard error):

```
DynaDup-X 1.3.3 (fast engine, ... threads, max heap ... MB)
Number of taxa: 6
Gene trees: 7, internal nodes: 39 (35 valid STBs, 4 overlapping), polytomies resolved: 1
STBs in gene trees (count): 21
STBs in gene trees (sum): 35
Number of Clusters: 22
DP states: 16, candidate bipartitions: 32, fallback states: 0
Sigma N: 39.0
total cost: 12.0
```

and `species.tre` contains `(((a,b),c),((d,e),f));`.

## Input

* One rooted gene tree per line, in Newick format.
* A species may occur any number of times in a gene tree (paralogs), or not at all (missing taxa).
* Polytomies are resolved deterministically as left-deep caterpillars; unary nodes are suppressed; branch lengths, support values, internal labels and `[comments]` are ignored.
* Gene trees must be rooted. Unrooted gene trees can be rooted with `scripts/root_gene_trees.py` (outgroup or midpoint rooting).

### Mapping leaf labels to species

```sh
# regular expression and replacement ("/delete/" = empty string)
java -jar dynadupx.jar -i genes.tre -a "_[0-9]+_[0-9]+$" /delete/ -o sp.tre    # SimPhy labels: species_locus_individual
# mapping file: one line per species, "species:label1,label2,...;"
java -jar dynadupx.jar -i genes.tre -a map.txt -o sp.tre
```

## Usage

```sh
# MGD (default search space -x0)
java -jar dynadupx.jar -i genes.tre -a map.txt -o sp.tre
# MGD with the gene-tree-split search space (fastest)
java -jar dynadupx.jar -i genes.tre -a map.txt -xs -o sp.tre
# MGDL, restriction-based losses (default search space -xc)
java -jar dynadupx.jar -i genes.tre -a map.txt -dl 0 -o sp.tre
# MGDL, missing taxa counted as losses
java -jar dynadupx.jar -i genes.tre -a map.txt -dl 1 -x0 -o sp.tre
# weighted objective: duplications + 1.0 * extra lineages + 0.5 * B
java -jar dynadupx.jar -i genes.tre -a map.txt -lambda 1 -mu 0.5 -o sp.tre
# large inputs: more heap, fixed number of threads
java -Xmx100g -jar dynadupx.jar -i genes.tre -a map.txt -xs -threads 32 -o sp.tre
```

| option | meaning |
|---|---|
| `-i FILE` | gene trees (required) |
| `-o FILE` | output species tree (printed to standard output if omitted) |
| `-a FILE` or `-a REGEX REPL` | leaf label → species mapping |
| `-dl W` | MGDL with loss weight `W` in [0,1] (0 = restriction-based losses, 1 = missing taxa counted as losses); `-dl auto` chooses `W` as DynaDup does |
| `-lambda L -mu M` | minimize duplications + `L`·XL + `M`·B |
| `-x0` | DynaDup search space (default for MGD) |
| `-xc` | extended search space: complements of all clusters added (default for MGDL) |
| `-xs` | gene-tree-split search space |
| `-threads N` | number of threads (default: all cores) |
| `-gpu`, `-gpu-verify` | GPU back end; `-gpu-verify` also recomputes on the CPU and stops at the first difference |
| `-legacy` | run the original DynaDup implementation (also used for its other options `-st`, `-ex`, `-xt`, `-cs`, `-cd`, `-dll`) |

### Search spaces

All three search spaces start from the same set of clusters: every species, the species set of every internal gene tree node, the two difference sets of every node whose children share a species, the set of all species, and complements.

* `-x0`: a cluster *C* may be resolved by every subtree-bipartition *A* | *C*−*A* whose two parts are clusters of the set; only the complements of the largest clusters are added. A cluster without such a resolution is resolved by DynaDup's fallback rule.
* `-xc`: as `-x0`, with the complement of every cluster added.
* `-xs`: a cluster that is the species set of a speciation-compatible (disjoint-children) gene tree node may only use the subtree-bipartitions of such nodes; all other clusters use the `-x0` rule.

The DP returns an optimal species tree within the chosen search space. Ties are broken by a fixed canonical order, so the result does not depend on the number of threads.

### Output

| line | meaning |
|---|---|
| `Number of Clusters` | clusters of the search space |
| `DP states`, `candidate bipartitions`, `fallback states` | size of the DP state space |
| `EVENTS dup=.. xl=.. loss=.. B=..` | (MGDL) events of the inferred species tree |
| `Sigma N` | constant term of the cost |
| `total cost` | duplications (MGD) or duplications + losses (MGDL) of the inferred tree |
| `timings` | wall-clock time of the four phases |

A line starting with `[memory]` reports that the weight computation runs within a reduced memory budget; the result is identical, only slower. Increase `-Xmx` or decrease `-threads` to avoid it.

## Memory and performance

* Java uses at most the heap given with `-Xmx` (default: one quarter of the physical memory). Give the program most of the machine's memory for large inputs.
* Memory grows linearly with the number of gene tree nodes and clusters; no structure is quadratic in the number of species.
* Up to 2,048 species exact bitsets are used for clusters; beyond that, clusters are compared through their signatures and species lists.
* All memory budgets (weight rows, MGDL presence sets, DP layer chunks) only change the speed, never the result.

## GPU back end (optional)

```sh
./build_gpu.sh                       # builds libdynadupx.so with nvcc and checks it on this machine's GPU
java -Djava.library.path=$PWD -jar dynadupx.jar -i genes.tre ... -gpu
./test_gpu.sh genes.tre -a map.txt   # compares GPU and CPU results, then times both
```

The GPU computes the MGD weights of all candidate subtree-bipartitions and, for MGDL, the per-tree lineage counts of each DP layer. The program falls back to the CPU (with a message) if the library or a suitable GPU is not available.

## Scripts

* `scripts/root_gene_trees.py` – roots gene trees by an outgroup or at the midpoint.
* `scripts/run_dynadupx_all.sh` – runs MGD and MGDL on every data folder `DynaDupX-SIM_mc_*` and records time and peak memory.

## Tests

```sh
tests/run_example.sh                    # the example of this README, compared with examples/expected_output.txt
N=20 python3 tests/random_vs_legacy.py  # random multi-copy inputs: DynaDup-X vs the original DynaDup implementation (MGD)
python3 tests/dl_random_vs_legacy.py    # the same for MGDL with w = 0, 0.5, 1
python3 tests/fuzz_root_gene_trees.py   # tests of the rooting script
tests/gpu_emulator/run.sh genes.tre ... # runs the CUDA kernels on the CPU through a fake CUDA runtime
```

Debug cross-checks (environment variables): `DDX_CHECK_W=1` recomputes every weight by a second method, `DDX_CHECK_SPARSE=1` recomputes every MGDL term from complete per-tree rows; `DDX_PROFILE=1` prints progress and memory use.

## Citation

If you use DynaDup-X, please cite:

* P. Saha and M. S. Bayzid. DynaDup-X: scaling species tree inference under gene duplication and loss to ultra-large datasets. (manuscript in preparation)
* M. S. Bayzid, S. Mirarab and T. Warnow. Inferring optimal species trees under gene duplication and loss. *Pacific Symposium on Biocomputing* 18:250–261, 2013.

## License

DynaDup-X is distributed under the GNU General Public License v3.0 (see `LICENSE`). It builds on DynaDup and PhyloNet; see `NOTICE.md` for third-party components.
