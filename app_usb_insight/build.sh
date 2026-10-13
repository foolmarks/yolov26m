#!/usr/bin/env bash
# Cross-compile usb_insight for the Modalix DevKit (ARM64).
# Run inside the Neat SDK container from a login shell (provides SYSROOT, CC and CXX).
set -euo pipefail
cd "$(dirname "$0")"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/neat-resources/apps-src/cmake/toolchains/aarch64-modalix.cmake
cmake --build build -j"$(nproc)"
