#!/bin/bash
# 阶段1: ACC 从零重训 (scenA), 每epoch归档FCT, 结束后做greedy eval
set -u
ROOT=/root/paddlejob/workspace/yangziwen/CoPTER
PY=/root/miniconda3/envs/m3/bin/python
NS3_BIN=$ROOT/ns-3.33/build/scratch/copter-sim
export LD_LIBRARY_PATH=$ROOT/ns-3.33/build/lib:${LD_LIBRARY_PATH:-}
unset NS_LOG

PORT=5602
EXP=scenA_acc_v2
MODEL_DIR=$ROOT/copter/models_${EXP}
OUT=$ROOT/simulation/output/${EXP}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-40}
mkdir -p "$MODEL_DIR" "$OUT"

for ((i=1; i<=TOTAL_EPOCHS; i++)); do
    echo "[orch] epoch $i/$TOTAL_EPOCHS $(date +%H:%M:%S)"
    cd "$ROOT/simulation"
    "$NS3_BIN" mix/acc_scenA.conf --port=$PORT > "$OUT/ns3_ep${i}.log" 2>&1 &
    NSPID=$!
    sleep 5
    cd "$ROOT/copter"
    "$PY" copter.py -p $PORT -e "$EXP" --online -m ACC -d "$MODEL_DIR" \
        -s 4 -i 8 -b 10000 \
        --epsilon_start 1.0 --epsilon_end 0.05 --epsilon_decay_steps 2500 \
        --state_save_interval 1 --seed 1 \
        --eval_tag "train_scenA" --tb_enable false \
        > "$OUT/agent_ep${i}.log" 2>&1
    rc=$?
    wait $NSPID 2>/dev/null
    cp -f "$ROOT/simulation/output/scenA/acc_scenA.fct" "$OUT/fct_ep${i}.fct" 2>/dev/null
    echo "[orch] epoch $i rc=$rc fct=$(wc -l < "$OUT/fct_ep${i}.fct" 2>/dev/null || echo 0)"
    tail -1 "$MODEL_DIR/${EXP}_metrics.jsonl" 2>/dev/null | head -c 400; echo
    sleep 3
done

# 最终贪心评估 x3
for j in 1 2 3; do
    echo "[orch] greedy eval $j"
    cd "$ROOT/simulation"
    "$NS3_BIN" mix/acc_scenA.conf --port=$PORT > "$OUT/ns3_eval${j}.log" 2>&1 &
    NSPID=$!
    sleep 5
    cd "$ROOT/copter"
    "$PY" copter.py -p $PORT -e "$EXP" --eval_greedy -m ACC -d "$MODEL_DIR" \
        -s 4 -b 10000 --eval_tag "eval_scenA_final" --tb_enable false \
        > "$OUT/agent_eval${j}.log" 2>&1
    wait $NSPID 2>/dev/null
    cp -f "$ROOT/simulation/output/scenA/acc_scenA.fct" "$OUT/fct_eval${j}.fct" 2>/dev/null
    sleep 3
done
echo "ACC_TRAIN_ALL_DONE"
