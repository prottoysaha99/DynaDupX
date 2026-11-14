#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run DynaDup-X MGD (-xs) and MGDL (-xs -dl 1) on every folder whose name
# starts with "DynaDupX-SIM_mc_*", using gene_trees.tre inside each folder.
# Results go to a SEPARATE output directory (one sub-folder per dataset).
#
# Usage:
#   bash run_dynadupx_all.sh [DATA_DIR] [OUT_DIR]
#     DATA_DIR : where the DynaDupX-SIM_mc_* folders are   (default: .)
#     OUT_DIR  : where results will be written             (default: ./DynaDupX_results)
# ---------------------------------------------------------------------------

# ======================= EDIT THESE IF NEEDED ==============================
JAR="$(cd "$(dirname "$0")/.." && pwd)/dynadupx.jar"   # path to the DynaDup-X jar
MEM="55g"                          # Java heap (-Xmx)
root="rooted"                      # label used in output file names
version=""                         # optional label used in output file names
SKIP_DONE=1                        # 1 = skip a run if its output tree already exists
# ===========================================================================

DATA_DIR="${1:-.}"
OUT_DIR="${2:-./DynaDupX_results}"

# --- basic checks ----------------------------------------------------------
[ -f "$JAR" ]          || { echo "ERROR: jar not found: $JAR"; exit 1; }
[ -x /usr/bin/time ]   || { echo "ERROR: /usr/bin/time not found (install package 'time')"; exit 1; }
command -v java >/dev/null || { echo "ERROR: java not found in PATH"; exit 1; }

mkdir -p "$OUT_DIR"
shopt -s nullglob

dirs=( "$DATA_DIR"/DynaDupX-SIM_mc_*/ )
[ ${#dirs[@]} -gt 0 ] || { echo "No DynaDupX-SIM_mc_* folders found in $DATA_DIR"; exit 1; }

for d in "${dirs[@]}"; do
    d="${d%/}"                         # remove trailing slash
    name="$(basename "$d")"            # e.g. DynaDupX-SIM_mc_t25000_g1000
    indir="$d/gene_trees.tre"

    if [ ! -f "$indir" ]; then
        echo "[$name] gene_trees.tre not found -> skipped"
        continue
    fi

    # Number of gene trees: from "_g1000" in the folder name, else count lines
    if [[ "$name" =~ _g([0-9]+) ]]; then
        Numberofgt="${BASH_REMATCH[1]}"
    else
        Numberofgt="$(grep -c ';' "$indir")"
    fi

    outdir="$OUT_DIR/$name"
    mkdir -p "$outdir"

    echo "=================================================================="
    echo "[$name]  input: $indir   gene trees: $Numberofgt"
    echo "         output: $outdir"
    echo "=================================================================="

    # ------------------------------ MGD (-xs) ------------------------------
    out_mgd="$outdir/${Numberofgt}mgdXcpu${root}${version}.tree"
    if [ "$SKIP_DONE" = 1 ] && [ -s "$out_mgd" ]; then
        echo "[$name] MGD already done -> skipped"
    else
        echo "[$name] MGD started: $(date)"
        /usr/bin/time -f "Time: %E\nPeak RAM: %M KB" \
            -o "$outdir/${Numberofgt}mgdXcpu${root}${version}_ramusage.tree" \
            java -Xmx"$MEM" -jar "$JAR" -i "${indir}" \
            -o "$out_mgd" \
            -a "_.*" "/delete/" -xs \
            > "$outdir/${Numberofgt}mgdXcpu${root}${version}.log" 2>&1
        echo "[$name] MGD finished (exit $?): $(date)"
    fi

    # --------------------------- MGDL (-xs -dl 1) --------------------------
    out_mgdl="$outdir/${Numberofgt}mgdlXcpu${root}${version}.tree"
    if [ "$SKIP_DONE" = 1 ] && [ -s "$out_mgdl" ]; then
        echo "[$name] MGDL already done -> skipped"
    else
        echo "[$name] MGDL started: $(date)"
        /usr/bin/time -f "Time: %E\nPeak RAM: %M KB" \
            -o "$outdir/${Numberofgt}mgdlXcpu${root}${version}_ramusage.tree" \
            java -Xmx"$MEM" -jar "$JAR" -i "${indir}" \
            -o "$out_mgdl" \
            -a "_.*" "/delete/" -xs -dl 1 \
            > "$outdir/${Numberofgt}mgdlXcpu${root}${version}.log" 2>&1
        echo "[$name] MGDL finished (exit $?): $(date)"
    fi
done

echo "All done. Results are in: $OUT_DIR"
