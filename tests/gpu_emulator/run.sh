#!/bin/sh
# Runs the CUDA kernels of native/dynadupx_jni.cu on the CPU, without a GPU, through a fake CUDA runtime
# (fakecuda.cpp): every kernel launch becomes a loop over its blocks, one thread per block, which is a valid
# schedule for these kernels. DynaDup-X then runs with -gpu-verify, which compares every GPU value with the CPU.
#   test/gpu_emulator/run.sh genes.tre [options]     e.g.  -a "_[0-9]+_[0-9]+$" "/delete/" -dl 0 -x0
# FAKE_RC=222 makes the fake runtime fail like a PTX-only library on an older driver (tests the self-test message).
# Needs g++, perl and a JDK. Slow (one CPU thread plays the whole GPU): use small inputs.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
JH=${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")}
OUT=${TMPDIR:-/tmp}/ddx_gpu_emulator; mkdir -p "$OUT"
perl -0pe 's/(\w+)<<<\s*([^,>]+?)\s*,.*?>>>\s*\(/for (blockIdx.x = 0, gridDim.x = (unsigned)($2); blockIdx.x < gridDim.x; ++blockIdx.x) $1(/gs' \
  "$ROOT/native/dynadupx_jni.cu" > "$OUT/emu.cpp"
g++ -std=c++14 -O2 -shared -fPIC -w -I"$HERE" -I"$JH/include" -I"$JH/include/linux" "$OUT/emu.cpp" "$HERE/fakecuda.cpp" \
  -o "$OUT/libdynadupx.so"
GT=$1; shift
java -Djava.library.path="$OUT" -cp "$ROOT/dynadupx.jar" phylonet.coalescent.fast.GpuCheck || true
java -Djava.library.path="$OUT" -jar "$ROOT/dynadupx.jar" -i "$GT" "$@" -gpu-verify -o "$OUT/sp.tre"
