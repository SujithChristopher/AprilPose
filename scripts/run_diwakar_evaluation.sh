#!/usr/bin/env bash
set -euo pipefail

UV_CACHE_DIR=.uv-cache rtk uv run python scripts/evaluate_mocap_orientation_error.py "$@"
