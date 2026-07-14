#!/bin/bash
# 阶段1-奖励函数动作区分度检查:
# 用 --force_action 固定"好/坏"两组动作各跑1个epoch (eval_greedy, 不训练),
# 比较 rollout_mean_reward 与 FCT, 验证奖励对动作敏感.
set -u
ROOT=/root/paddlejob/workspace/yangziwen/CoPTER
PY=/root/miniconda3/envs/m3/bin/python
NS3_BIN=$ROOT/ns-3.33/build/scratch/copter-sim
export LD_LIBRARY_PATH=$ROOT/ns-3.33/build/lib:${LD_LIBRARY_PATH:-}
unset NS_LOG

PORT=5601
OUT=$ROOT/simulation/output/sanity
mkdir -p "$OUT"

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
    echo "run $tag done rc=$rc, fct lines: $(wc -l < "$OUT/fct_${tag}.fct" 2>/dev/null || echo 0)"
    sleep 5
}

# good: 中等kmin/kmax, 低pmax (接近secn2静态最优: kmin=40/160KB kmax=160/640KB pmax=0.2)
# 动作空间归一化: kmin_norm=0.2259 -> ~981KB? 取决于OPENGYM范围(100..4000KB@25G)
# kmin idx2(0.2259), kmax idx1(0.25), pmax idx1(0.2)
run_one good "2,1,1"
# bad: kmin=1.0(4000KB), kmax=1.0(12000KB), pmax=0.1 -> 几乎不标记ECN, 深队列
run_one bad "5,3,0"
# aggressive-bad: kmin=0(100KB), kmax=0(500KB), pmax=1.0 -> 过度标记, 压吞吐
run_one aggr "0,0,9"

echo "===== summary ====="
for tag in good bad aggr; do
    echo "--- $tag ---"
    f="$OUT/models_${tag}/sanity_${tag}_metrics.jsonl"
    [ -f "$f" ] && cat "$f" || echo "metrics missing: $f"
done
echo "ALL_SANITY_DONE"
