# Third-party components

DynaDup-X builds on the following software. Their original authors retain their copyrights.

* **DynaDup** (M. S. Bayzid, S. Mirarab, T. Warnow; https://github.com/smirarab/DynaDup) – command-line front end
  (`src/phylonet/coalescent/MGDInference_DP.java` and the other classes in `src/phylonet/coalescent/`), the original
  implementation that is run with `-legacy`, and the compiled classes in `lib/main.jar`.
* **PhyloNet** (Rice University; https://bioinfocs.rice.edu/phylonet) – tree data structures used by the DynaDup front
  end (`lib/main.jar`, `src/phylonet/tree`, `src/phylonet/util`).
* **jsr166** concurrency utilities (public domain; `lib/jsr166.jar`).

The fast engine (`src/phylonet/coalescent/fast/`), the CUDA back end (`native/`), the scripts and the tests are part of
DynaDup-X.
