#!/bin/sh
# Builds dynadupx.jar (DynaDup-X fast engine + the original DynaDup command-line front end).
# Requires a JDK 17 or newer.
set -e
cd "$(dirname "$0")"
VERSION=1.3.3
rm -rf build && mkdir build
javac -nowarn -encoding UTF-8 --release 17 -classpath lib/main.jar:lib/jsr166.jar -d build \
  src/phylonet/util/BitSet.java src/phylonet/coalescent/*.java src/phylonet/coalescent/fast/*.java \
  src/phylonet/tree/model/sti/STITreeCluster.java
printf 'Main-Class: phylonet.coalescent.MGDInference_DP\nClass-Path: lib/main.jar lib/jsr166.jar\nImplementation-Title: DynaDup-X\nImplementation-Version: %s\n' "$VERSION" > build/manifest.txt
jar cfm dynadupx.jar build/manifest.txt -C build phylonet
rm -rf build
echo "built dynadupx.jar (DynaDup-X $VERSION)"
