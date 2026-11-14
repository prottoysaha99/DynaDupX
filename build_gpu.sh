#!/bin/sh
# Builds the optional CUDA back end of DynaDup-X (libdynadupx.so) and tests it on this machine's GPU.
# Requires nvcc and a JDK (for jni.h). Build on the machine (or a node of the same type) that will run DynaDup-X.
#
#   ./build_gpu.sh                 machine code for the GPU(s) that nvidia-smi reports on this machine
#   ./build_gpu.sh -arch=sm_86     your own target, e.g. when building on a login node without a GPU
#                                  (86 = compute capability 8.6; see: nvidia-smi --query-gpu=name,compute_cap --format=csv)
#
# Why machine code: a library that carries only PTX (what nvcc makes by default for a GPU newer than its default
# target) must be compiled by the GPU driver when the program starts, and a driver older than the nvcc that wrote
# the PTX refuses it:
#   [gpu] disabled: CUDA error 222 (cudaErrorUnsupportedPtxVersion: the provided PTX was compiled with an
#         unsupported toolchain)
# With machine code (SASS) for the GPU nothing has to be compiled at run time, so any driver of the same CUDA major
# version works.
set -e
cd "$(dirname "$0")"
: "${JAVA_HOME:=$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")}"
echo "JAVA_HOME=$JAVA_HOME"
command -v nvcc >/dev/null 2>&1 || { echo "nvcc not found: load a CUDA module (e.g. module load cuda) or add the toolkit's bin to PATH"; exit 1; }

NVCC_VER=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)
DRV_VER=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)
echo "nvcc: CUDA ${NVCC_VER:-?}    GPU driver on this machine: CUDA ${DRV_VER:-? (no nvidia-smi or no GPU here)}"
if [ -n "$NVCC_VER" ] && [ -n "$DRV_VER" ]; then
  if [ "${NVCC_VER%%.*}" -gt "${DRV_VER%%.*}" ]; then
    echo "ERROR: nvcc is CUDA $NVCC_VER but the driver only supports CUDA $DRV_VER (older major version):"
    echo "       a library built with this nvcc cannot run here. Use a CUDA ${DRV_VER%%.*}.x toolkit"
    echo "       (e.g. module load cuda/$DRV_VER) or update the GPU driver."
    exit 1
  fi
  newest=$(printf '%s\n%s\n' "$NVCC_VER" "$DRV_VER" | sort -t. -k1,1n -k2,2n | tail -n 1)
  if [ "$newest" != "$DRV_VER" ]; then
    echo "note: nvcc ($NVCC_VER) is newer than the driver ($DRV_VER). This works only with machine code for the GPU,"
    echo "      which is what this script builds (the driver cannot compile newer PTX: CUDA error 222)."
  fi
fi

TARGET=""
case " $* " in
  *" -arch"*|*" --gpu-architecture"*|*" -gencode"*|*" --generate-code"*|*" -code"*)
    echo "target: your flags ($*)" ;;
  *)
    CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un || true)
    if [ -n "$CAPS" ]; then
      KNOWN=$(nvcc --list-gpu-code 2>/dev/null || true)
      for c in $CAPS; do
        if [ -z "$KNOWN" ] || echo "$KNOWN" | grep -qx "sm_$c"; then
          TARGET="$TARGET -gencode arch=compute_$c,code=sm_$c -gencode arch=compute_$c,code=compute_$c"
          echo "target: GPU with compute capability $c -> machine code sm_$c (+ PTX compute_$c)"
        else
          # the GPU is newer than this nvcc: ship PTX of the newest architecture nvcc knows; the driver (which knows
          # the GPU, so it is newer than nvcc) compiles it at start-up
          top=$(nvcc --list-gpu-arch 2>/dev/null | sed -n 's/^compute_\([0-9][0-9]*\)$/\1/p' | sort -n | tail -n 1)
          TARGET="$TARGET -gencode arch=compute_$top,code=compute_$top"
          echo "target: GPU with compute capability $c is newer than this nvcc -> PTX compute_$top (the driver compiles it)"
        fi
      done
    else
      TARGET="-arch=all"
      echo "target: no GPU visible here -> machine code for every GPU architecture this nvcc supports (-arch=all);"
      echo "        if this nvcc is older than 11.5, pass the target yourself: ./build_gpu.sh -arch=sm_XY"
    fi ;;
esac

# shellcheck disable=SC2086
nvcc -O3 -std=c++14 -shared -Xcompiler -fPIC \
     -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/linux" \
     $TARGET "$@" -o libdynadupx.so native/dynadupx_jni.cu \
  || { echo "nvcc failed (see above)"; exit 1; }
echo "built libdynadupx.so"
if command -v cuobjdump >/dev/null 2>&1; then
  echo "  machine code inside: $(cuobjdump --list-elf libdynadupx.so 2>/dev/null | sed -n 's/.*\.\(sm_[0-9a-z]*\)\..*/\1/p' | sort -u | tr '\n' ' ')"
  echo "  PTX inside:          $(cuobjdump --list-ptx libdynadupx.so 2>/dev/null | sed -n 's/.*\.\(sm_[0-9a-z]*\)\..*/\1/p' | sort -u | tr '\n' ' ')"
fi

if [ -n "$DRV_VER" ] && [ -f dynadupx.jar ]; then
  echo "== self-test on this machine's GPU"
  java -Djava.library.path="$PWD" -cp dynadupx.jar phylonet.coalescent.fast.GpuCheck \
    || { echo "the library does not work on this GPU (reason above)"; exit 1; }
else
  echo "(no GPU here: test the library on the GPU machine with"
  echo "   java -Djava.library.path=$PWD -cp dynadupx.jar phylonet.coalescent.fast.GpuCheck )"
fi
echo "run with:  java -Djava.library.path=$PWD -jar dynadupx.jar -i genes.tre ... -gpu"
