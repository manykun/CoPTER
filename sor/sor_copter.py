import argparse
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
from structures import AgentHelperParameters, NetworkHelperParameters
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
    parser.add_argument("-b", "--switch_buffer", type=int, default=10000)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=50000)
    parser.add_argument("--state_save_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_greedy", action="store_true", help="Pure greedy evaluation: epsilon=0, no recording/training/saving.")
    parser.add_argument("--eval_tag", type=str, default="", help="Optional tag recorded into metrics (e.g. phase/task name).")
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
    set_random_seed(args.seed)
    logger.info(f"Parsed arguments: {args}")
    logger.info(f"Random seed fixed to {args.seed}")
    logger.add(args.exp_name + "_log/sor_copter_{time}.log", level="INFO", rotation="5 MB")

    network_helper_params = NetworkHelperParameters(port_states=6, port_actions=3, state_observations=3, switch_buffer_size=args.switch_buffer)
    network_helper = NetworkHelper(ns3_socket=args.ns3_socket, nhp=network_helper_params)
    agent_helper_params = AgentHelperParameters(
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        state_save_interval=args.state_save_interval,
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
    )
    agent_helper.load(args.override_name)

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
    try:
        while True:
            if current_step == 0:
                done = network_helper.monitor(current_step)
            elif current_step < max(4, args.static_steps):
                for port_idx in range(network_helper.get_n_port()):
                    paras = network_helper.get_port_current_parameters(port_idx)
                    network_helper.configurator(current_step, port_idx, paras)
                done = network_helper.monitor(current_step)
            else:
                port_states = [network_helper.get_port_current_state_list(port_idx) for port_idx in range(network_helper.get_n_port())]
                current_epsilon = 0.0 if args.eval_greedy else agent_helper.get_current_epsilon()
                paras, actions = agent_helper.decide(port_states, epsi=current_epsilon)
                for port_idx, parameter in enumerate(paras):
                    network_helper.configurator(current_step, port_idx, parameter)
                done = network_helper.monitor(current_step)
                try:
                    n_port = network_helper.get_n_port()
                    step_sum = 0.0
                    step_active = 0
                    for port_idx in range(n_port):
                        r = float(network_helper.get_port_current_reward(port_idx))
                        if network_helper.is_port_active(port_idx):
                            step_sum += r
                            step_active += 1
                    if step_active > 0:
                        rollout_reward_sum += step_sum
                        rollout_reward_count += step_active
                    else:
                        all_sum = sum(
                            float(network_helper.get_port_current_reward(p))
                            for p in range(n_port)
                        )
                        rollout_reward_sum += all_sum / n_port
                        rollout_reward_count += 1
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
                "rollout_mean_reward": (rollout_reward_sum / rollout_reward_count) if rollout_reward_count > 0 else None,
            })
        except Exception as exc:
            logger.exception(f"Failed to save SOR agent state on exit: {exc}")
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
