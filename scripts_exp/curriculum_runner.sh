#!/usr/bin/env bash
set -Eeuo pipefail
method="${1:?method}"; phase="${2:?phase}"; flow="${3:?flow}"; run_dir="${4:?run_dir}"; seed="${5:-1}"; epochs="${6:-30}"; resume="${7:-0}"
ROOT=/root/paddlejob/workspace/yangziwen/CoPTER; PY=/root/miniconda3/envs/m3/bin/python; BIN="$ROOT/ns-3.33/build/scratch/copter-sim"
PORT=$((5620 + ${phase#p})); [[ "$method" == SOR ]] && PORT=$((PORT + 10))
scenario=A; [[ "$flow" == *scenB* ]] && scenario=B; model="$run_dir/checkpoints/${method}"; exp="${method,,}_curriculum_seed${seed}"; mkdir -p "$model" "$run_dir/metrics" "$run_dir/fct" "$run_dir/logs"
export LD_LIBRARY_PATH="$ROOT/ns-3.33/build/lib:${LD_LIBRARY_PATH:-}"; unset NS_LOG
for ((epoch=1; epoch<=epochs; epoch++)); do
 (cd "$ROOT/simulation" && "$BIN" "mix/acc_scen${scenario}.conf" --port="$PORT") >"$run_dir/logs/${method}_${phase}_ns3_${epoch}.log" 2>&1 & ns=$!; trap 'kill $ns 2>/dev/null || true' EXIT; sleep 5
 if [[ "$method" == ACC ]]; then agent=("$PY" copter.py -p "$PORT" -e "$exp" --online -m ACC -d "$model" -s 4 -i 8 -b 10000 --epsilon_start 1.0 --epsilon_end 0.05 --epsilon_decay_steps 600 --seed "$seed" --run_id "$exp" --phase "$phase" --eval_tag "train_${phase}_${scenario}" --tb_enable false); [[ "$phase" != p1 || "$epoch" -gt 1 || "$resume" == 1 ]] && agent+=(--resume); work="$ROOT/copter"; else agent=("$PY" sor_copter.py -p "$PORT" -e "$exp" --online -d "$model" -s 4 -i 8 -b 10000 --epsilon_start 1.0 --epsilon_end 0.05 --epsilon_decay_steps 600 --seed "$seed" --eval_tag "train_${phase}_${scenario}" --tb_enable false); work="$ROOT/sor"; fi
 (cd "$work" && "${agent[@]}") >"$run_dir/logs/${method}_${phase}_agent_${epoch}.log" 2>&1; wait "$ns"; trap - EXIT
done
for eval_scenario in A B; do
 eval_flow="$ROOT/simulation/mix/scen${eval_scenario}_$([[ "$eval_scenario" == A ]] && echo throughput || echo incast).flow"; out="$run_dir/fct/${method}_${phase}_eval_${eval_scenario}.fct"; metrics="$run_dir/metrics/${method}_${phase}_eval_${eval_scenario}.jsonl"
 if [[ "$method" == ACC ]]; then "$ROOT/scripts_exp/acc_eval_runner.sh" ACC "$eval_flow" "$out" "$metrics" "$seed" "$model/final.pt"; else
  # SOR frozen evaluation uses its own policy and the same simulator configuration.
  EVAL_PORT=$((PORT+20)) "$ROOT/scripts_exp/sor_eval_runner.sh" "$exp" "$model" "$eval_scenario" "$out" "$metrics" "$seed"
 fi
done
