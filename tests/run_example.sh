#!/bin/sh
# Runs the README example and compares the result with examples/expected_output.txt.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${TMPDIR:-/tmp}/ddx_example; mkdir -p "$OUT"
java -jar "$ROOT/dynadupx.jar" -i "$ROOT/examples/genes.tre" -a "_.*" /delete/ -threads 1 -o "$OUT/sp.tre" 2> "$OUT/log.txt"
grep -E '^(Number|Gene trees|STBs|DP states|Sigma|total cost)' "$OUT/log.txt" > "$OUT/got.txt"
cat "$OUT/sp.tre" >> "$OUT/got.txt"
if diff "$ROOT/examples/expected_output.txt" "$OUT/got.txt"; then echo "example: OK"; else echo "example: DIFFERENT"; exit 1; fi
