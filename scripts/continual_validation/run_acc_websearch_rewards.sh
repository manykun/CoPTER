#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "${PYTHON_BIN:-python3}" "$ROOT/scripts/continual_validation/run_acc_websearch_rewards.py" "$@"
