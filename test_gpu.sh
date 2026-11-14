#!/bin/sh
# Verifies the GPU back end against the CPU, then times it. Usage: ./test_gpu.sh genes.tre [-a regex /delete/]
set -e
GT=$1; shift
J="java -Djava.library.path=$PWD -jar dynadupx.jar"
echo "== correctness (-gpu-verify runs both and aborts on any mismatch)"
$J -i "$GT" "$@" -x0 -gpu-verify -o /tmp/gpu_mgd.tre
$J -i "$GT" "$@" -x0 -dl 0 -gpu-verify -o /tmp/gpu_mgdl.tre
echo "== timing"
for mode in "-x0" "-x0 -dl 0" "-x0 -dl 0 -xc"; do
  for gpu in "" "-gpu"; do
    s=$(date +%s.%N)
    $J -i "$GT" "$@" $mode $gpu -o /tmp/t.tre 2> /tmp/t.log
    e=$(date +%s.%N)
    printf "%-18s %-5s %8.2fs   cost=%s\n" "$mode" "${gpu:-cpu}" "$(echo "$e-$s" | bc)" "$(grep -o 'total cost: [0-9.]*' /tmp/t.log | cut -d' ' -f3)"
  done
done
