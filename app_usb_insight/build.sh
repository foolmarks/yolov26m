#!/usr/bin/env bash
# Cross-compile the application for the Modalix ARM64 target.
# Run inside the Neat SDK container (ghcr.io-sima-neat-sdk-v2.1.3.0), which
# exports CC/CXX/SYSROOT for the aarch64 toolchain.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${APP_DIR}/build}"
TOOLCHAIN="${CMAKE_TOOLCHAIN_FILE:-/neat-resources/apps-src/cmake/toolchains/aarch64-modalix.cmake}"

if [[ ! -f "${TOOLCHAIN}" ]]; then
  echo "Cross toolchain file not found: ${TOOLCHAIN}" >&2
  echo "Run this script inside the Neat SDK container." >&2
  exit 1
fi

cmake -S "${APP_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" -j "$(nproc)"

echo
file "${BUILD_DIR}/app-usb-insight"
