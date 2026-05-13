#!/bin/bash

# 循环执行100次
for i in {1..20}
do
    echo "=============================="
    echo "开始第 $i 次执行 (共100次)"
    echo "=============================="
    
    # 执行命令并记录开始时间
    start_time=$(date +%s)
    # python copter.py -p 5555 -e copter_experiment_acc_webserver_t0.05_l0.7 --online
    # python copter.py -p 5557 -e experiment_acc --online
    # python copter.py -p 5555 -e second_copter_experiment_copter_webserver_t0.05_l0.7 -m CoPTER -f fmaps --online
    # python copter.py -p 5555 -e third_triplehead_copter_experiment_copter_webserver_t0.05_l0.7 -m CoPTER -f fmaps --online
    # python copter.py -p 5555 -e forth_new_copter_experiment_copter_webserver_t0.05_l0.7 -m CoPTER -f /home/ame/m3-main/parsimon-eval/expts/fig_8/analysis/fmaps_port_level_448ports/_dcqcn_07load_webserver --online
    # python copter.py -p 5555 -e fifth_new_reward_copter_experiment_copter_webserver_t0.05_l0.7 -m CoPTER -f /home/ame/m3-main/parsimon-eval/expts/fig_8/analysis/fmaps_port_level_448ports/_dcqcn_07load_webserver --online
    # python copter.py -p 5555 -e seven_new_reward_5paramStep_copter_experiment_copter_webserver_t0.05_l0.7 -m CoPTER -f /home/ame/m3-main/parsimon-eval/expts/fig_8/analysis/fmaps_port_level_448ports/_dcqcn_06load_hadoop_5paramStep --online
    # python copter.py -p 5555 -e eight_08_02_5paramStep_copter_experiment_copter_webserver_t0.05_l0.7 -m CoPTER -f /home/ame/m3-main/parsimon-eval/expts/fig_8/analysis/fmaps_port_level_448ports/_dcqcn_06load_hadoop_5paramStep --online
    # python copter.py -p 5555 -e experiment_copter -m CoPTER -f /home/ame/m3-main/parsimon-eval/expts/fig_8/analysis/fmaps_port_level_448ports/_dcqcn_06load_hadoop_5paramStep --online
    # python copter.py -p 5555 -e experiment_copter_new -m CoPTER -f /home/ame/m3-main/parsimon-eval/expts/fig_8/analysis/fmaps_port_level_448ports/new/_mix_webserver_websearch_hadoop_clusters --online
    python copter.py -p 5555 -e experiment_copter_thesis -m CoPTER -f /home/ame/m3-main/parsimon-eval/expts/fig_8/analysis/fmaps_port_level_448ports/_thesis_mix_webserver_websearch_cachefollower_random_1 --online
    # 计算命令执行时间
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "------------------------------"
    echo "命令执行完成，耗时: $duration 秒"
    
    # 如果不是最后一次执行，则等待30秒
    if [ $i -lt 20 ]; then
        echo "等待30秒后开始下一次执行..."
        sleep 30
    else
        echo "已完成所有100次执行"
    fi
done
