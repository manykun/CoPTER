#!/bin/bash
set -u
ROOT=/root/paddlejob/workspace/yangziwen/CoPTER
PY=/root/miniconda3/envs/m3/bin/python
NS3_BIN=$ROOT/ns-3.33/build/scratch/copter-sim
export LD_LIBRARY_PATH=$ROOT/ns-3.33/build/lib:${LD_LIBRARY_PATH:-}
unset NS_LOG
PORT=5601
OUT=$ROOT/simulation/output/sanity
cd "$ROOT/simulation"
"$NS3_BIN" mix/acc_scenA.conf --port=$PORT > "$OUT/ns3_good.log" 2>&1 &
NSPID=$!
sleep 5
cd "$ROOT/copter"
mkdir -p "$OUT/models_good"
"$PY" copter.py -p $PORT -e "sanity_good" --eval_greedy \
    --force_action "2,1,1" -m ACC -d "$OUT/models_good" \
    -b 10000 --eval_tag "sanity_good" --tb_enable false \
    > "$OUT/agent_good.log" 2>&1
wait $NSPID 2>/dev/null
cp -f "$ROOT/simulation/output/scenA/acc_scenA.fct" "$OUT/fct_good.fct"
cat "$OUT/models_good/sanity_good_metrics.jsonl"
echo GOOD_DONE
