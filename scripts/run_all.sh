#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "scripts/run_all.sh is kept for compatibility."
echo "Use scripts/setup_dev.sh once, then scripts/check_all.sh for repeated checks."
bash scripts/check_all.sh
