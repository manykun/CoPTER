#!/bin/bash

NS3_DIR=../ns-3.33
NS3_BLD_DIR=$NS3_DIR/build
NS3_LIB_DIR=$(realpath $NS3_BLD_DIR)/lib
LOG_LEV=level_all

LOG_FILE="sim_log_$(date +%Y%m%d_%H%M%S).txt"

# 允许至少1个参数（配置文件），支持额外参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <RDMA_CONF> [额外参数，如--port 5555]"
    exit 1
fi
RDMA_CONF=$1  # 第一个参数为配置文件
shift  # 剩下的参数传递给ns3程序


if [ ! -f "$RDMA_CONF" ]; then
    echo "Error: Configuration file '$RDMA_CONF' not found"
    exit 1
fi

export LD_LIBRARY_PATH=$NS3_LIB_DIR:$LD_LIBRARY_PATH
# export NS_LOG=CongestionControlSimulator=$LOG_LEV
# 新增打印参数的命令（用于调试）
# echo "传递给 ns3 程序的参数: $NS3_BLD_DIR/scratch/copter-sim $RDMA_CONF $@"
# $NS3_BLD_DIR/scratch/copter-sim "$RDMA_CONF" "$@"
export NS_LOG="CongestionControlSimulator=$LOG_LEV:OpenGymInterface=$LOG_LEV"
echo "传递给 ns3 程序的参数: $NS3_BLD_DIR/scratch/copter-sim $RDMA_CONF $@"
# $NS3_BLD_DIR/scratch/copter-sim "$RDMA_CONF" "$@" > "$LOG_FILE" 2>&1
$NS3_BLD_DIR/scratch/copter-sim "$RDMA_CONF" "$@"
echo "log save to:$LOG_FILE"