
import os
import argparse
import hashlib
import json
import math
import random
import numpy as np
from loguru import logger
from network_helper import NetworkHelper
from agent_helper import AgentHelper
from port_metrics import (
    PortMetricTracker,
    parse_forced_port_action,
    parse_watch_ports,
)
from structures import (
    ACC_ACTION_SPACES,
    EPSILON_SCHEDULES,
    AgentHelperParameters,
    DCQCNParameters,
    NetworkHelperParameters,
    acc_action_from_indices,
    validate_acc_action_indices,
)


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


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the CoPTER experiment.")
    parser.add_argument("-p", "--ns3_socket", type=int, default=5555, help="ns3_socket: The port number for the ns3-gym environment.")
    parser.add_argument("-e", "--exp_name", type=str, default="copter_experiment", help="exp_name: The name of the experiment for logging.")
    parser.add_argument("-r", "--override_name", type=str, default=None, help="override_name: Override the experiment name for loading models.")
    parser.add_argument("-m", "--mode", type=str, choices=["ACC", "CoPTER"], default="ACC", help="mode: The mode of the agent, either 'ACC' or 'CoPTER'.")
    parser.add_argument("-f", "--fmap_dir", type=str, default=None, help="fmap_dir: The directory for the feature map, required if mode is 'CoPTER'.")
    parser.add_argument("-o", "--online", action="store_true", help="online: Whether to run the agent in online mode.")
    parser.add_argument("-d", "--model_dir", type=str, default="models", help="model_dir: The directory where the model is stored.")
    parser.add_argument("-s", "--static_steps", type=int, default=4, help="static_steps: The number of initial steps to keep the actions static.")
    parser.add_argument("-i", "--train_intervals", type=int, default=8, help="train_intervals: The number of intervals for training the agent.")
    parser.add_argument("-b", "--switch_buffer", type=int, default=400, help="Switch buffer size in KB. This is experiment metadata; ns-3 normalizes queue occupancy.")
    parser.add_argument("--max_steps", type=int, default=0, help="max_steps: If > 0, exit cleanly after this many steps in this run (one epoch). 0 means run until ns3 ends.")
    parser.add_argument("--max_global_train_steps", type=int, default=0, help="Stop this process after reaching this absolute optimizer-update count.")
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=50000)
    parser.add_argument(
        "--epsilon_schedule",
        choices=EPSILON_SCHEDULES,
        default="phase",
        help="phase restarts epsilon at task boundaries; global continues one schedule across tasks.",
    )
    parser.add_argument("--acc_hidden_dims", type=str, default="32,64,64,32", help="Comma-separated ACC hidden layer widths.")
    parser.add_argument(
        "--action_space",
        choices=ACC_ACTION_SPACES,
        default="legacy",
        help="ACC categorical action space. multiscale uses a valid threshold-profile head plus Pmax.",
    )
    parser.add_argument("--reward_weights", type=str, default="0.50,0.30,0.20", help="Throughput,queue,ECN reward weights; must sum to 1.")
    parser.add_argument("--reward_profile", choices=("weighted", "tail_safe"), default="weighted")
    parser.add_argument("--reward_queue_lambda", type=float, default=5.0)
    parser.add_argument("--reward_ecn_lambda", type=float, default=5.0)
    parser.add_argument("--state_save_interval", type=int, default=1)
    parser.add_argument(
        "--shared_replay",
        choices=("true", "false"),
        default="true",
        help="Enable ACC cross-port shared/global replay. false keeps only per-port FIFO replay.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for Python, NumPy, and PyTorch.")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run identifier stored with training state and metrics.")
    parser.add_argument("--phase", type=str, default=None, help="Optional training phase stored with training state and metrics.")
    parser.add_argument("--resume", action="store_true", help="Resume models, replay buffers and counters from saved state.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename override used when resuming.")
    parser.add_argument("--eval_greedy", action="store_true", help="Pure greedy evaluation: epsilon=0, no recording/training/saving.")
    parser.add_argument("--eval_tag", type=str, default="", help="Optional tag recorded into metrics (e.g. phase/task name).")
    parser.add_argument("--force_action", type=str, default="", help="Sanity-check: force a fixed action index triple 'kmin_idx,kmax_idx,pmax_idx' for ALL ports/steps (overrides the policy). Used to test reward sensitivity to actions.")
    parser.add_argument("--force_port_action", type=str, default="", help="Frozen local sweep: override one port with PORT,KMIN_NORM,KMAX_NORM,PMAX while all other ports remain greedy.")
    parser.add_argument("--watch_ports", type=str, default="", help="Comma-separated port indices to track explicitly. Their per-epoch rollout reward (EMA) is written to metrics jsonl and TensorBoard (rollout/reward_port{p}) so a fixed port's reward trajectory can be plotted across epochs.")
    parser.add_argument("--watch_trace_file", type=str, default="", help="Optional JSONL path for per-step metrics of --watch_ports.")
    # ---- tensorboard ----
    parser.add_argument("--tb_enable", type=str, default="true", help="Enable tensorboard logging: true/false")
    parser.add_argument("--tb_log_dir", type=str, default="tb_logs", help="TensorBoard root log dir; actual dir = <tb_log_dir>/<exp_name>")
    parser.add_argument("--tb_flush_secs", type=int, default=30)
    args = parser.parse_args()
    try:
        acc_hidden_dims = tuple(int(width) for width in args.acc_hidden_dims.split(","))
        if not acc_hidden_dims or any(width <= 0 for width in acc_hidden_dims):
            raise ValueError("all widths must be positive")
    except ValueError as exc:
        raise SystemExit(f"--acc_hidden_dims must be comma-separated positive integers: {exc}")
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

    # Parse optional forced action (sanity-check for reward sensitivity).
    forced_action_idx = None
    if args.force_action:
        try:
            forced_action_idx = validate_acc_action_indices(
                args.force_action.split(","), args.action_space
            )
        except Exception as exc:
            expected = (
                "kmin_idx,kmax_idx,pmax_idx"
                if args.action_space == "legacy"
                else "profile_idx,pmax_idx"
            )
            raise SystemExit(
                f"--force_action must be '{expected}' for {args.action_space}; "
                f"got {args.force_action!r} ({exc})"
            )
    try:
        forced_port_action = parse_forced_port_action(args.force_port_action)
    except ValueError as exc:
        raise SystemExit(f"invalid --force_port_action: {exc}")
    if forced_action_idx is not None and forced_port_action is not None:
        raise SystemExit("--force_action and --force_port_action are mutually exclusive")
    if forced_port_action is not None and not args.eval_greedy:
        raise SystemExit("--force_port_action is allowed only with --eval_greedy")

    # Parse optional watch-port list (fixed ports whose reward we track across epochs).
    try:
        watch_ports = parse_watch_ports(args.watch_ports)
    except ValueError as exc:
        raise SystemExit(f"invalid --watch_ports: {exc}")

    # Print the parsed arguments
    logger.info(f"Parsed arguments: {args}")
    logger.info(f"Random seed fixed to {args.seed}")

    # Initialize Logger
    logger.add(args.exp_name + "_log/copter_{time}.log", level="INFO", rotation="5 MB")
    logger.info(f"Copter Starting Running at {os.getcwd()}")

    # Initialize NetworkHelper
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
        shared_replay_enabled=args.shared_replay == "true",
    )

    config_hash = hashlib.sha256(
        json.dumps(vars(args), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    agent_helper = AgentHelper(node_number=network_helper.get_n_port(), ahp=agent_helper_params, model_dir=args.model_dir, mode=args.mode,
                               exp_name=args.exp_name, online=args.online, network_helper=network_helper, fmap_dir=args.fmap_dir,
                               run_id=args.run_id, phase=args.phase, config_hash=config_hash,
                               acc_hidden_dims=acc_hidden_dims,
                               acc_action_space=args.action_space)

    checkpoint_name = args.checkpoint if args.resume and args.checkpoint else args.override_name
    agent_helper.load(checkpoint_name)

    # ---- Initialize TensorBoard SummaryWriter (跨 epoch 续写到同一目录，曲线连续) ----
    tb_enabled = str(args.tb_enable).lower() in ("1", "true", "yes", "on")
    tb_writer = None
    if tb_enabled:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(args.tb_log_dir, args.exp_name)
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_dir, flush_secs=args.tb_flush_secs)
            # 把超参写到 text 面板，方便回查
            hparams_text = "\n".join([f"- **{k}**: {v}" for k, v in vars(args).items()])
            tb_writer.add_text("hparams", hparams_text, global_step=agent_helper.global_train_step)
            agent_helper.attach_tb(tb_writer)
            logger.info(f"TensorBoard initialized: log_dir={tb_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
            tb_writer = None

    current_step = 0
    # Accumulator for per-step rollout reward (used so eval epochs also report
    # a mean_reward; training epochs additionally have replay-batch rewards
    # tracked in agent_helper._recent_rewards).
    rollout_reward_sum = 0.0
    rollout_reward_count = 0
    # Track additional metrics for diagnosing catastrophic forgetting.
    congested_port_count_sum = 0     # total congested-port samples across the epoch
    congested_step_count = 0          # steps that had ≥1 congested port
    per_port_reward_ema = {}          # port_idx → EMA of reward (for debugging)
    # NEW: track distribution of rewards across congested ports each step.
    # rollout_top30_reward: mean of the top-30% congested-port rewards this step.
    # This filters out structurally-stuck ports (e.g. a permanently-saturated
    # bottleneck) whose reward cannot respond to DCQCN parameters and would
    # otherwise clamp the metric to a floor. It surfaces the policy's actual
    # improvement on the ports where it can matter.
    rollout_top30_sum = 0.0
    rollout_top30_count = 0
    rollout_median_sum = 0.0
    rollout_median_count = 0
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
                # Get the initial observation for all ports
                done = network_helper.monitor(current_step)

            elif current_step < max(4, args.static_steps):
                # For the first few steps, we keep their actions as what the environment provides.
                for port_idx in range(network_helper.get_n_port()):
                    paras = network_helper.get_port_current_parameters(port_idx)
                    network_helper.configurator(current_step, port_idx, paras)
                done = network_helper.monitor(current_step)

            else:
                # Get the current states for all ports. NOTE: `Observation` is different from `State` in CoPTER.
                port_states = [
                    network_helper.get_port_current_state_list(port_idx)
                    for port_idx in range(network_helper.get_n_port())
                ]

                # Decide the actions for all ports with AgenHelper
                # Use globally-decaying epsilon so that exploration carries across runs.
                current_epsilon = 0.0 if args.eval_greedy else agent_helper.get_current_epsilon()
                if forced_action_idx is not None:
                    # Sanity-check: bypass the policy and apply a fixed action to
                    # every port so we can measure reward sensitivity to actions.
                    forced_para = acc_action_from_indices(
                        forced_action_idx, args.action_space
                    )
                    n_port = network_helper.get_n_port()
                    paras = [forced_para for _ in range(n_port)]
                    actions = [forced_action_idx for _ in range(n_port)]
                else:
                    paras, actions = agent_helper.decide(port_states, epsi=current_epsilon)

                if forced_port_action is not None:
                    port, kmin_norm, kmax_norm, pmax = forced_port_action
                    paras[port] = DCQCNParameters(kmin_norm, kmax_norm, pmax)
                    # Evaluation never records this action into replay. Keeping
                    # the applied continuous values in `actions` makes traces
                    # and action histograms auditable.
                    actions[port] = (kmin_norm, kmax_norm, pmax)

                for action in actions:
                    action_key = ",".join(str(index) for index in action)
                    action_histogram[action_key] = action_histogram.get(action_key, 0) + 1

                # Cofigure the actions for each port
                for port_idx, parameter in enumerate(paras):
                    network_helper.configurator(current_step, port_idx, parameter)

                # Enforce the updated parameters
                done = network_helper.monitor(current_step)
                if done:
                    logger.info("NS3 terminal notification received after action; exiting before reward/record update.")
                    break
                port_metric_tracker.observe(
                    current_step, actions, network_helper
                )
                # Accumulate per-port reward for this step.
                #
                # KEY DESIGN: aggregate reward ONLY over CONGESTED ports.
                # An idle port always returns a near-constant "everything is
                # fine" reward regardless of DCQCN parameters. Averaging idle
                # ports (which dominate numerically) washes out the policy
                # signal — the rollout mean stays glued to ~0.72 no matter
                # what the agent does. By restricting the average to
                # congested ports we surface the true policy performance.
                #
                # Fallback chain:
                #   1. Congested ports (queue built up OR ECN marked)  → best signal
                #   2. Top-K congested by score (if none pass threshold) → still concentrated
                #   3. All active ports (last resort)
                try:
                    n_port = network_helper.get_n_port()
                    congested_ports = [p for p in range(n_port)
                                       if network_helper.is_port_congested(p)]
                    if len(congested_ports) > 0:
                        rewards = []
                        for p in congested_ports:
                            components = network_helper.get_port_current_reward_components(p)
                            r = float(components["reward"])
                            rewards.append(r)
                            for key in reward_component_keys:
                                reward_component_sums[key] += float(components[key])
                            reward_component_count += 1
                            # Update per-port EMA for diagnostics
                            prev = per_port_reward_ema.get(p, r)
                            per_port_reward_ema[p] = 0.9 * prev + 0.1 * r
                        rollout_reward_sum += sum(rewards)
                        rollout_reward_count += len(rewards)
                        congested_port_count_sum += len(rewards)
                        congested_step_count += 1

                        # Top-30% mean: robust to structurally-stuck ports.
                        rewards_sorted = sorted(rewards, reverse=True)
                        k = max(1, int(len(rewards_sorted) * 0.30))
                        top30 = rewards_sorted[:k]
                        rollout_top30_sum += sum(top30) / len(top30)
                        rollout_top30_count += 1

                        # Median reward: robust to both tails.
                        mid = len(rewards_sorted) // 2
                        median = rewards_sorted[mid] if len(rewards_sorted) % 2 == 1 \
                                 else 0.5 * (rewards_sorted[mid - 1] + rewards_sorted[mid])
                        rollout_median_sum += median
                        rollout_median_count += 1
                    else:
                        # No congested ports this step — take the top-K by
                        # congestion score so the metric is still concentrated
                        # on the ports carrying the strongest DCQCN signal.
                        topk = network_helper.get_topk_congested_ports(k=8)
                        if len(topk) > 0:
                            rewards = []
                            for p in topk:
                                components = network_helper.get_port_current_reward_components(p)
                                rewards.append(float(components["reward"]))
                                for key in reward_component_keys:
                                    reward_component_sums[key] += float(components[key])
                                reward_component_count += 1
                            rollout_reward_sum += sum(rewards)
                            rollout_reward_count += len(rewards)
                            rollout_top30_sum += max(rewards)
                            rollout_top30_count += 1
                            rollout_median_sum += sorted(rewards)[len(rewards)//2]
                            rollout_median_count += 1
                except Exception as _exc:
                    logger.warning(f"rollout reward accumulation failed at step {current_step}: {_exc}")
            # 步骤1：先执行monitor，确保获取端口标识
            # done = network_helper.monitor(current_step)

            # # 步骤2：仅在标识获取完成后（current_step≥1）执行决策
            # if current_step >= 1 and network_helper.port_identifier_map:
            #     # 获取端口状态
            #     port_states = [
            #         network_helper.get_port_current_state_list(port_idx)
            #         for port_idx in range(network_helper.get_n_port())
            #     ]
            #     # 调用decide（此时会触发fmap延迟加载）
            #     paras, actions = agent_helper.decide(port_states, epsi=args.epsilon)
                
            #     # 配置动作
            #     for port_idx, parameter in enumerate(paras):
            #         network_helper.configurator(current_step, port_idx, parameter)

                # 只有在线模式才记录经验（如果offline模式不需要记录，可添加此判断）
                if args.online and not args.eval_greedy:

                    # Record the states and actions in the agent's replay buffer
                    for port_idx in range(network_helper.get_n_port()):
                        # Get the current (new) states, (last) states, actions (in dice format), and rewards
                        current_state = network_helper.get_port_current_state_list(port_idx)
                        last_state = network_helper.get_port_last_state_list(port_idx)
                        action = actions[port_idx]
                        # reward = network_helper.get_port_current_reward(port_idx)
                        # Add the experience to the agent's replay buffer
                        # agent_helper.record(port_idx, last_state, action, reward, current_state)
                        agent_helper.record(port_idx, last_state, action, current_state)

                # Train the agent every `train_intervals` steps
                if args.online and not args.eval_greedy and current_step % args.train_intervals == 0:
                    agent_helper.sync()
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

            # Optional clean exit at max_steps for a single epoch (set by launcher).
            if args.max_steps > 0 and current_step >= args.max_steps:
                logger.info(f"Reached max_steps={args.max_steps}; exiting this epoch cleanly.")
                break

    except KeyboardInterrupt:
        print("Ctrl-C -> Exit")

    except Exception as e:
        logger.exception(f"Unexpected error in main loop: {e}")
        raise

    finally:
        # Always persist agent state so cross-run training is continuous.
        try:
            if not args.eval_greedy:
                agent_helper.save()
            agent_helper.append_epoch_metrics({
                "steps_this_epoch": current_step,
                "eval_greedy": bool(args.eval_greedy),
                "eval_tag": args.eval_tag,
                "reward_profile": args.reward_profile,
                "reward_queue_lambda": args.reward_queue_lambda,
                "reward_ecn_lambda": args.reward_ecn_lambda,
                "shared_replay_enabled": args.shared_replay == "true",
                # PRIMARY metric: top-30% congested-port reward. Robust to
                # structurally-stuck ports (permanent bottlenecks) that would
                # otherwise clamp a naive mean. Reflects policy's actual
                # ceiling on the responsive subset of the network.
                "rollout_mean_reward": (rollout_top30_sum / rollout_top30_count) if rollout_top30_count > 0 else None,
                # Legacy full-congested-mean, kept for backward compatibility.
                "rollout_all_congested_mean": (rollout_reward_sum / rollout_reward_count) if rollout_reward_count > 0 else None,
                # Median reward — robust to both tails.
                "rollout_median_reward": (rollout_median_sum / rollout_median_count) if rollout_median_count > 0 else None,
                "congested_step_ratio": (congested_step_count / current_step) if current_step > 0 else 0.0,
                "avg_congested_ports_per_step": (congested_port_count_sum / congested_step_count) if congested_step_count > 0 else 0.0,
                "forced_action": list(forced_action_idx) if forced_action_idx is not None else None,
                "action_space": args.action_space,
                "forced_port_action": (
                    list(forced_port_action)
                    if forced_port_action is not None else None
                ),
                "action_histogram": action_histogram,
                **{
                    f"reward_{key}_mean": (
                        reward_component_sums[key] / reward_component_count
                        if reward_component_count > 0 else None
                    )
                    for key in reward_component_keys
                },
                "per_port_reward_top5": (
                    [{"port": p, "reward": round(r, 4)} for p, r in
                     sorted(per_port_reward_ema.items(), key=lambda kv: -kv[1])[:5]]
                    if per_port_reward_ema else []
                ),
                "per_port_reward_bottom5": (
                    [{"port": p, "reward": round(r, 4)} for p, r in
                     sorted(per_port_reward_ema.items(), key=lambda kv: kv[1])[:5]]
                    if per_port_reward_ema else []
                ),
                # Fixed watch-list ports: their rollout reward EMA this epoch.
                # Enables plotting a specific port's reward trajectory across
                # epochs (None if the port was never congested this epoch).
                "watch_ports_reward": {
                    port: (round(value, 6) if value is not None else None)
                    for port, value in port_metric_tracker.legacy_rewards().items()
                } if watch_ports else {},
                "watch_ports_metrics": port_metric_tracker.summary(),
            })
            if args.eval_greedy:
                logger.info(
                    f"Evaluation metrics recorded without modifying training state: "
                    f"epoch={agent_helper.epoch}, global_step={agent_helper.global_train_step}"
                )
            else:
                logger.info(
                    f"Saved training state at exit: epoch={agent_helper.epoch}, "
                    f"global_step={agent_helper.global_train_step}, epsilon={agent_helper.epsilon:.4f}"
                )
            # Fixed watch-port reward -> TensorBoard under a dedicated namespace
            # so each port has its own continuous curve across epochs.
            if tb_writer is not None and watch_ports:
                step = int(agent_helper.global_train_step)
                for p in watch_ports:
                    rv = per_port_reward_ema.get(p)
                    if rv is not None:
                        tb_writer.add_scalar(f"rollout/reward_port{p}", float(rv), step)
        except Exception as e:
            logger.exception(f"Failed to save agent state on exit: {e}")
        try:
            port_metric_tracker.write_trace()
        except Exception as e:
            logger.exception(f"Failed to write watch-port trace: {e}")
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
