#!/usr/bin/env bash
# Compatibility entry point for the A-B-A ACC/SOR effectiveness experiment.
set -Eeuo pipefail

ROOT="/root/paddlejob/workspace/yangziwen/CoPTER"
exec "$ROOT/scripts_exp/run_stage2_forgetting.sh" "$@"
