#!/usr/bin/env bash
set -Eeuo pipefail
run_dir="${1:?run_dir}"; seed="${2:-1}"; epochs="${3:-80}"; resume="${4:-0}"
ROOT=/root/paddlejob/workspace/yangziwen/CoPTER; PY=/root/miniconda3/envs/m3/bin/python; BIN="$ROOT/ns-3.33/build/scratch/copter-sim"
PORT="${ACC_PORT:-5610}"; EXP="stage1_acc_seed${seed}"; MODEL="$run_dir/checkpoints"; mkdir -p "$MODEL" "$run_dir/metrics" "$run_dir/fct"
export LD_LIBRARY_PATH="$ROOT/ns-3.33/build/lib:${LD_LIBRARY_PATH:-}"; unset NS_LOG
start=1; [[ "$resume" == 1 && -f "$run_dir/last_epoch" ]] && start=$(( $(<"$run_dir/last_epoch") + 1 ))
for ((epoch=start; epoch<=epochs; epoch++)); do
  (cd "$ROOT/simulation" && "$BIN" mix/acc_scenA.conf --port="$PORT") >"$run_dir/logs/ns3_train_${epoch}.log" 2>&1 & ns=$!
  trap 'kill $ns 2>/dev/null || true' EXIT; sleep 5
  args=(-p "$PORT" -e "$EXP" --online -m ACC -d "$MODEL" -s 4 -i 8 -b 10000 --epsilon_start 1.0 --epsilon_end 0.05 --epsilon_decay_steps 2500 --state_save_interval 1 --seed "$seed" --run_id "$EXP" --phase train_A --eval_tag train_A --tb_enable false)
  [[ "$resume" == 1 || "$epoch" -gt 1 ]] && args+=(--resume)
  (cd "$ROOT/copter" && "$PY" copter.py "${args[@]}") >"$run_dir/logs/agent_train_${epoch}.log" 2>&1
  wait "$ns"; trap - EXIT; printf '%s\n' "$epoch" > "$run_dir/last_epoch"
done
metrics="$MODEL/${EXP}_metrics.jsonl"; [[ -s "$metrics" ]] || { echo "missing metrics: $metrics" >&2; exit 2; }; cp -f "$metrics" "$run_dir/metrics/acc_metrics.jsonl"
checkpoint="$(find "$MODEL" -name '*ACC_0' -type f -print -quit)"; [[ -n "$checkpoint" ]] || { echo "missing checkpoint in $MODEL" >&2; exit 2; }; cp -f "$checkpoint" "$run_dir/checkpoints/final.pt"
