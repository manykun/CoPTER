import argparse
import hashlib
import json
import math
import os
import random
import sys

import numpy as np
from loguru import logger


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as exc:
        logger.warning(f"Failed to set torch random seed: {exc}")

SOR_DIR = os.path.dirname(__file__)
COPTER_DIR = os.path.abspath(os.path.join(SOR_DIR, "..", "copter"))
if COPTER_DIR not in sys.path:
    sys.path.insert(0, COPTER_DIR)
if SOR_DIR not in sys.path:
    sys.path.insert(0, SOR_DIR)

from network_helper import NetworkHelper
from fct_metrics import FCTStepTracker
from port_metrics import (
    PortMetricTracker,
    parse_forced_port_action,
    parse_watch_ports,
)
from structures import (
    ACC_ACTION_SPACES,
    AgentHelperParameters,
    DCQCNParameters,
    EPSILON_SCHEDULES,
    NetworkHelperParameters,
)
from sor_agent_helper import SORAgentHelper
from sor_replay import SORReplayConfig


def build_parser():
    parser = argparse.ArgumentParser(description="Run SOR-ACC experiment.")
    parser.add_argument("-p", "--ns3_socket", type=int, default=5555)
    parser.add_argument("-e", "--exp_name", type=str, default="sor_acc_experiment")
    parser.add_argument("-r", "--override_name", type=str, default=None)
    parser.add_argument("-o", "--online", action="store_true")
    parser.add_argument("-d", "--model_dir", type=str, default="sor_models")
    parser.add_argument("-s", "--static_steps", type=int, default=4)
    parser.add_argument("-i", "--train_intervals", type=int, default=8)
    parser.add_argument("-b", "--switch_buffer", type=int, default=400)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_global_train_steps", type=int, default=0)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=50000)
    parser.add_argument(
        "--epsilon_schedule",
        choices=EPSILON_SCHEDULES,
        default="phase",
        help="phase restarts epsilon at task boundaries; global continues one schedule across tasks.",
    )
    parser.add_argument(
        "--acc_hidden_dims",
        type=str,
        default="32,64,64,32",
        help="Comma-separated hidden layer widths; identical to ACC.",
    )
    parser.add_argument(
        "--action_space",
        choices=ACC_ACTION_SPACES,
        default="legacy",
        help="ACC/SOR categorical action parameterization.",
    )
    parser.add_argument(
        "--target_update_interval",
        type=int,
        default=100,
        help="Hard target sync interval in global optimizer updates.",
    )
    parser.add_argument(
        "--reward_weights",
        type=str,
        default="0.50,0.30,0.20",
        help="Throughput,queue,ECN weights; must sum to 1.",
    )
    parser.add_argument("--reward_profile", choices=("weighted", "tail_safe"), default="weighted")
    parser.add_argument("--reward_queue_lambda", type=float, default=5.0)
    parser.add_argument("--reward_ecn_lambda", type=float, default=5.0)
    parser.add_argument("--state_save_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--phase", type=str, default=None)
    parser.add_argument("--eval_greedy", action="store_true", help="Pure greedy evaluation: epsilon=0, no recording/training/saving.")
    parser.add_argument("--eval_tag", type=str, default="", help="Optional tag recorded into metrics (e.g. phase/task name).")
    parser.add_argument("--watch_ports", type=str, default="", help="Comma-separated port indices to measure explicitly.")
    parser.add_argument("--watch_trace_file", type=str, default="", help="Optional JSONL path for per-step metrics of --watch_ports.")
    parser.add_argument("--fct_source_file", type=str, default="", help="ns-3 FCT stream to read incrementally.")
    parser.add_argument("--fct_step_trace_file", type=str, default="", help="CSV output with one FCT record per environment step.")
    parser.add_argument("--launcher_episode", type=int, default=None, help="Launcher episode identifier stored in the FCT trace.")
    parser.add_argument("--force_port_action", type=str, default="", help="Frozen local sweep: PORT,KMIN_NORM,KMAX_NORM,PMAX.")
    parser.add_argument("--tb_enable", type=str, default="true")
    parser.add_argument("--tb_log_dir", type=str, default="tb_logs")
    parser.add_argument("--tb_flush_secs", type=int, default=30)
    parser.add_argument("--sor_recent_size", type=int, default=2000)
    parser.add_argument("--sor_boundary_size", type=int, default=20000)
    parser.add_argument("--sor_max_clusters", type=int, default=32)
    parser.add_argument("--sor_prototype_distance", type=float, default=1.0)
    parser.add_argument("--sor_prototype_eta", type=float, default=0.05)
    parser.add_argument("--sor_boundary_threshold", type=float, default=0.5)
    parser.add_argument("--sor_alpha_td", type=float, default=1.0)
    parser.add_argument("--sor_beta_under_sample", type=float, default=0.2)
    parser.add_argument("--sor_gamma_drift", type=float, default=0.5)
    parser.add_argument("--sor_rho_boundary", type=float, default=0.5)
    parser.add_argument("--sor_temperature", type=float, default=1.0)
    parser.add_argument("--sor_uniform_mix", type=float, default=0.01)
    parser.add_argument("--sor_lambda_cons", type=float, default=0.01)
    parser.add_argument("--sor_lambda_reg", type=float, default=0.001)
    parser.add_argument("--sor_drift_reg_threshold", type=float, default=0.5)
    parser.add_argument("--sor_ref_update_interval", type=int, default=256)
    parser.add_argument("--sor_sync_interval", type=int, default=8,
                        help="Run global->per-port broadcast only every N train() calls.")
    parser.add_argument("--sor_save_buffer_every", type=int, default=5,
                        help="Pickle global replay buffer only every N epochs.")
    return parser


def main():
    args = build_parser().parse_args()
    try:
        acc_hidden_dims = tuple(int(width) for width in args.acc_hidden_dims.split(","))
        if not acc_hidden_dims or any(width <= 0 for width in acc_hidden_dims):
            raise ValueError("all widths must be positive")
    except ValueError as exc:
        raise SystemExit(
            f"--acc_hidden_dims must be comma-separated positive integers: {exc}"
        )
    try:
        reward_weights = tuple(float(weight) for weight in args.reward_weights.split(","))
        if len(reward_weights) != 3 or any(weight < 0 for weight in reward_weights):
            raise ValueError("exactly three non-negative weights are required")
        if not math.isclose(sum(reward_weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"weights sum to {sum(reward_weights)}, not 1")
    except ValueError as exc:
        raise SystemExit(f"--reward_weights is invalid: {exc}")
    set_random_seed(args.seed)
    if args.reward_queue_lambda < 0 or args.reward_ecn_lambda < 0:
        raise SystemExit("tail-safe reward lambdas must be non-negative")
    if args.target_update_interval <= 0:
        raise SystemExit("--target_update_interval must be positive")
    if not 0.0 <= args.sor_uniform_mix <= 1.0:
        raise SystemExit("--sor_uniform_mix must be in [0, 1]")
    try:
        watch_ports = parse_watch_ports(args.watch_ports)
    except ValueError as exc:
        raise SystemExit(f"invalid --watch_ports: {exc}")
    try:
        forced_port_action = parse_forced_port_action(args.force_port_action)
    except ValueError as exc:
        raise SystemExit(f"invalid --force_port_action: {exc}")
    if forced_port_action is not None and not args.eval_greedy:
        raise SystemExit("--force_port_action is allowed only with --eval_greedy")
    logger.info(f"Parsed arguments: {args}")
    logger.info(f"Random seed fixed to {args.seed}")
    logger.add(args.exp_name + "_log/sor_copter_{time}.log", level="INFO", rotation="5 MB")

    network_helper_params = NetworkHelperParameters(
        port_states=6,
        port_actions=3,
        state_observations=3,
        switch_buffer_size=args.switch_buffer,
        reward_throughput_weight=reward_weights[0],
        reward_queue_weight=reward_weights[1],
        reward_ecn_weight=reward_weights[2],
        reward_profile=args.reward_profile,
        reward_queue_lambda=args.reward_queue_lambda,
        reward_ecn_lambda=args.reward_ecn_lambda,
    )
    network_helper = NetworkHelper(ns3_socket=args.ns3_socket, nhp=network_helper_params)
    port_metric_tracker = PortMetricTracker(watch_ports, args.watch_trace_file)
    try:
        fct_step_tracker = FCTStepTracker(
            args.fct_source_file,
            args.fct_step_trace_file,
            episode=args.launcher_episode,
            phase=args.phase or "",
            eval_tag=args.eval_tag,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid FCT step trace configuration: {exc}")
    try:
        port_metric_tracker.validate(network_helper.get_n_port())
    except ValueError as exc:
        raise SystemExit(str(exc))
    if (
        forced_port_action is not None
        and forced_port_action[0] >= network_helper.get_n_port()
    ):
        raise SystemExit(
            f"forced port {forced_port_action[0]} is outside "
            f"[0, {network_helper.get_n_port() - 1}]"
        )
    agent_helper_params = AgentHelperParameters(
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        epsilon_schedule=args.epsilon_schedule,
        state_save_interval=args.state_save_interval,
        target_update_interval=args.target_update_interval,
    )
    sor_config = SORReplayConfig(
        rb_size=agent_helper_params.rb_size,
        rb_size_global=agent_helper_params.rb_size_global,
        recent_size=args.sor_recent_size,
        boundary_size=args.sor_boundary_size,
        max_clusters=args.sor_max_clusters,
        prototype_distance=args.sor_prototype_distance,
        prototype_eta=args.sor_prototype_eta,
        boundary_threshold=args.sor_boundary_threshold,
        alpha_td=args.sor_alpha_td,
        beta_under_sample=args.sor_beta_under_sample,
        gamma_drift=args.sor_gamma_drift,
        rho_boundary=args.sor_rho_boundary,
        temperature=args.sor_temperature,
        uniform_mix=args.sor_uniform_mix,
    )
    agent_helper = SORAgentHelper(
        node_number=network_helper.get_n_port(),
        ahp=agent_helper_params,
        model_dir=args.model_dir,
        exp_name=args.exp_name,
        online=args.online,
        network_helper=network_helper,
        sor_config=sor_config,
        lambda_cons=args.sor_lambda_cons,
        lambda_reg=args.sor_lambda_reg,
        drift_reg_threshold=args.sor_drift_reg_threshold,
        ref_update_interval=args.sor_ref_update_interval,
        sync_interval=args.sor_sync_interval,
        save_buffer_every=args.sor_save_buffer_every,
        acc_hidden_dims=acc_hidden_dims,
        acc_action_space=args.action_space,
        run_id=args.run_id,
        phase=args.phase,
        config_hash=hashlib.sha256(
            json.dumps(vars(args), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
    )
    agent_helper.load(args.override_name)
    startup_env_step = agent_helper.global_env_step

    startup_hashes = {}
    startup_parameter_hashes = {}
    for name in ("policy_net", "target_net", "reference_net"):
        hasher = hashlib.sha256()
        parameter_hasher = hashlib.sha256()
        for agent in agent_helper.agent_pool:
            for key, tensor in sorted(getattr(agent, name).state_dict().items()):
                hasher.update(key.encode())
                hasher.update(tensor.detach().cpu().numpy().tobytes())
            for tensor in getattr(agent, name).parameters():
                parameter_hasher.update(tensor.detach().cpu().numpy().tobytes())
        startup_hashes[name] = hasher.hexdigest()
        startup_parameter_hashes[name] = parameter_hasher.hexdigest()

    tb_enabled = str(args.tb_enable).lower() in ("1", "true", "yes", "on")
    tb_writer = None
    if tb_enabled:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = os.path.join(args.tb_log_dir, args.exp_name)
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_dir, flush_secs=args.tb_flush_secs)
            hparams_text = "\n".join([f"- **{key}**: {value}" for key, value in vars(args).items()])
            tb_writer.add_text("hparams", hparams_text, global_step=agent_helper.global_train_step)
            agent_helper.attach_tb(tb_writer)
        except Exception as exc:
            logger.warning(f"Failed to initialize TensorBoard: {exc}")

    current_step = 0
    rollout_reward_sum = 0.0
    rollout_reward_count = 0
    rollout_top30_sum = 0.0
    rollout_top30_count = 0
    rollout_median_sum = 0.0
    rollout_median_count = 0
    congested_port_count_sum = 0
    congested_step_count = 0
    reward_component_keys = (
        "throughput", "queue", "ecn", "avg_tx_rate", "avg_queue",
        "peak_queue", "avg_ecn", "peak_ecn", "combined_queue",
        "combined_ecn", "queue_cost_sq", "ecn_cost_sq", "tail_safe_raw",
    )
    reward_component_sums = {key: 0.0 for key in reward_component_keys}
    reward_component_count = 0
    action_histogram = {}
    try:
        while True:
            if current_step == 0:
                done = network_helper.monitor(current_step)
                fct_step_tracker.observe(
                    current_step,
                    global_env_step=startup_env_step + current_step,
                    global_train_step=agent_helper.global_train_step,
                    terminal=done,
                )
            elif current_step < max(4, args.static_steps):
                for port_idx in range(network_helper.get_n_port()):
                    paras = network_helper.get_port_current_parameters(port_idx)
                    network_helper.configurator(current_step, port_idx, paras)
                done = network_helper.monitor(current_step)
                fct_step_tracker.observe(
                    current_step,
                    global_env_step=startup_env_step + current_step,
                    global_train_step=agent_helper.global_train_step,
                    terminal=done,
                )
            else:
                port_states = [network_helper.get_port_current_state_list(port_idx) for port_idx in range(network_helper.get_n_port())]
                current_epsilon = 0.0 if args.eval_greedy else agent_helper.get_current_epsilon()
                paras, actions = agent_helper.decide(port_states, epsi=current_epsilon)
                if forced_port_action is not None:
                    port, kmin_norm, kmax_norm, pmax = forced_port_action
                    paras[port] = DCQCNParameters(kmin_norm, kmax_norm, pmax)
                    actions[port] = (kmin_norm, kmax_norm, pmax)
                for action in actions:
                    key = ",".join(str(index) for index in action)
                    action_histogram[key] = action_histogram.get(key, 0) + 1
                for port_idx, parameter in enumerate(paras):
                    network_helper.configurator(current_step, port_idx, parameter)
                done = network_helper.monitor(current_step)
                fct_step_tracker.observe(
                    current_step,
                    global_env_step=startup_env_step + current_step,
                    global_train_step=agent_helper.global_train_step,
                    terminal=done,
                )
                if done:
                    logger.info(
                        "NS3 terminal notification received after action; "
                        "exiting before reward/record update."
                    )
                    break
                port_metric_tracker.observe(
                    current_step, actions, network_helper
                )
                try:
                    n_port = network_helper.get_n_port()
                    congested_ports = [
                        port_idx for port_idx in range(n_port)
                        if network_helper.is_port_congested(port_idx)
                    ]
                    if congested_ports:
                        rewards = []
                        for port_idx in congested_ports:
                            components = network_helper.get_port_current_reward_components(
                                port_idx
                            )
                            rewards.append(float(components["reward"]))
                            for key in reward_component_keys:
                                reward_component_sums[key] += float(components[key])
                            reward_component_count += 1
                        rollout_reward_sum += sum(rewards)
                        rollout_reward_count += len(rewards)
                        congested_port_count_sum += len(rewards)
                        congested_step_count += 1
                        ordered = sorted(rewards, reverse=True)
                        count = max(1, int(len(ordered) * 0.30))
                        rollout_top30_sum += sum(ordered[:count]) / count
                        rollout_top30_count += 1
                        middle = len(ordered) // 2
                        median = (
                            ordered[middle]
                            if len(ordered) % 2
                            else 0.5 * (ordered[middle - 1] + ordered[middle])
                        )
                        rollout_median_sum += median
                        rollout_median_count += 1
                    else:
                        topk = network_helper.get_topk_congested_ports(k=8)
                        if topk:
                            rewards = []
                            for port_idx in topk:
                                components = (
                                    network_helper.get_port_current_reward_components(
                                        port_idx
                                    )
                                )
                                rewards.append(float(components["reward"]))
                                for key in reward_component_keys:
                                    reward_component_sums[key] += float(
                                        components[key]
                                    )
                                reward_component_count += 1
                            rollout_reward_sum += sum(rewards)
                            rollout_reward_count += len(rewards)
                            rollout_top30_sum += max(rewards)
                            rollout_top30_count += 1
                            rollout_median_sum += sorted(rewards)[len(rewards) // 2]
                            rollout_median_count += 1
                except Exception as _exc:
                    logger.warning(f"rollout reward accumulation failed at step {current_step}: {_exc}")

                if args.online and not args.eval_greedy:
                    for port_idx in range(network_helper.get_n_port()):
                        current_state = network_helper.get_port_current_state_list(port_idx)
                        last_state = network_helper.get_port_last_state_list(port_idx)
                        agent_helper.record(port_idx, last_state, actions[port_idx], current_state)

                if args.online and not args.eval_greedy and current_step % args.train_intervals == 0:
                    agent_helper.maybe_sync()
                    agent_helper.train(current_step)
                    if (
                        args.max_global_train_steps > 0
                        and agent_helper.global_train_step >= args.max_global_train_steps
                    ):
                        logger.info(
                            "Reached max_global_train_steps="
                            f"{args.max_global_train_steps}; ending this phase."
                        )
                        break

            current_step += 1
            if done:
                logger.info("NS3 environment reported done; exiting this epoch cleanly.")
                break
            if args.max_steps > 0 and current_step >= args.max_steps:
                logger.info(f"Reached max_steps={args.max_steps}; exiting this epoch cleanly.")
                break
    except KeyboardInterrupt:
        print("Ctrl-C -> Exit")
    except Exception as exc:
        logger.exception(f"Unexpected error in SOR main loop: {exc}")
        raise
    finally:
        try:
            if not args.eval_greedy:
                agent_helper.save()
            agent_helper.append_epoch_metrics({
                "steps_this_epoch": current_step,
                "eval_greedy": bool(args.eval_greedy),
                "eval_tag": args.eval_tag,
                "reward_profile": args.reward_profile,
                "action_space": args.action_space,
                "target_update_interval": args.target_update_interval,
                "startup_network_hashes": startup_hashes,
                "startup_parameter_hashes": startup_parameter_hashes,
                "reward_queue_lambda": args.reward_queue_lambda,
                "reward_ecn_lambda": args.reward_ecn_lambda,
                "rollout_mean_reward": (
                    rollout_top30_sum / rollout_top30_count
                    if rollout_top30_count > 0 else None
                ),
                "rollout_all_congested_mean": (
                    rollout_reward_sum / rollout_reward_count
                    if rollout_reward_count > 0 else None
                ),
                "rollout_median_reward": (
                    rollout_median_sum / rollout_median_count
                    if rollout_median_count > 0 else None
                ),
                "congested_step_ratio": (
                    congested_step_count / current_step if current_step > 0 else 0.0
                ),
                "avg_congested_ports_per_step": (
                    congested_port_count_sum / congested_step_count
                    if congested_step_count > 0 else 0.0
                ),
                "action_histogram": action_histogram,
                "forced_port_action": (
                    list(forced_port_action)
                    if forced_port_action is not None else None
                ),
                "watch_ports_reward": {
                    port: (round(value, 6) if value is not None else None)
                    for port, value in port_metric_tracker.legacy_rewards().items()
                } if watch_ports else {},
                "watch_ports_metrics": port_metric_tracker.summary(),
                "fct_step_trace": fct_step_tracker.summary(),
                **{
                    f"reward_{key}_mean": (
                        reward_component_sums[key] / reward_component_count
                        if reward_component_count > 0 else None
                    )
                    for key in reward_component_keys
                },
            })
        except Exception as exc:
            logger.exception(f"Failed to save SOR agent state on exit: {exc}")
        try:
            port_metric_tracker.write_trace()
        except Exception as exc:
            logger.exception(f"Failed to write watch-port trace: {exc}")
        try:
            fct_step_tracker.close()
        except Exception as exc:
            logger.exception(f"Failed to close FCT step trace: {exc}")
        try:
            network_helper.close_env()
        except Exception:
            pass
        try:
            if tb_writer is not None:
                tb_writer.flush()
                tb_writer.close()
        except Exception:
            pass
        print("Done")


if __name__ == "__main__":
    main()
