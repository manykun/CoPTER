#!/bin/bash
set -u
ROOT=/root/paddlejob/workspace/yangziwen/CoPTER
PY=/root/miniconda3/envs/m3/bin/python
NS3_BIN=$ROOT/ns-3.33/build/scratch/copter-sim
export LD_LIBRARY_PATH=$ROOT/ns-3.33/build/lib:${LD_LIBRARY_PATH:-}
unset NS_LOG
PORT=5601
OUT=$ROOT/simulation/output/sanity

run_one () {
    local tag="$1"; local action="$2"
    echo "===== sanity run: $tag action=$action ====="
    cd "$ROOT/simulation"
    "$NS3_BIN" mix/acc_scenA.conf --port=$PORT > "$OUT/ns3_${tag}.log" 2>&1 &
    local ns3_pid=$!
    sleep 5
    cd "$ROOT/copter"
    mkdir -p "$OUT/models_${tag}"
    "$PY" copter.py -p $PORT -e "sanity_${tag}" --eval_greedy \
        --force_action "$action" -m ACC -d "$OUT/models_${tag}" \
        -b 10000 --eval_tag "sanity_${tag}" --tb_enable false \
        > "$OUT/agent_${tag}.log" 2>&1
    local rc=$?
    wait $ns3_pid 2>/dev/null
    cp -f "$ROOT/simulation/output/scenA/acc_scenA.fct" "$OUT/fct_${tag}.fct" 2>/dev/null
    echo "run $tag done rc=$rc"
    sleep 5
}

run_one bad "5,3,0"
run_one aggr "0,0,9"
echo "===== summary ====="
for tag in good bad aggr; do
    echo "--- $tag ---"
    f="$OUT/models_${tag}/sanity_${tag}_metrics.jsonl"
    [ -f "$f" ] && cat "$f" || echo "metrics missing: $f"
done
echo "ALL_SANITY_DONE"
