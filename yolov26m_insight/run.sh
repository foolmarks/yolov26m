#!/usr/bin/env bash
# Run the application on the paired DevKit.
# Execute inside the Neat SDK container; `dk` copies nothing because /workspace
# is NFS-mounted on the DevKit, so it runs the binary in place.
set -euo pipefail

APP_DIR="${APP_DIR:-/workspace/yolov26m_insight}"
CONFIG="${CONFIG:-${APP_DIR}/config/config.yaml}"

# shellcheck source=/dev/null
[[ -f "${HOME}/.devkit-sync.rc" ]] && source "${HOME}/.devkit-sync.rc"

if ! type dk >/dev/null 2>&1; then
  echo "'dk' is not available. Run this inside the Neat SDK container with a paired DevKit." >&2
  exit 1
fi

dk "${APP_DIR}/build/yolov26m-insight" --config "${CONFIG}" "$@"
