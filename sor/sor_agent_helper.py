import json
import os
import pickle
import random
import resource
import time
from collections import deque
from dataclasses import replace

import numpy as np
import torch

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

COPTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "copter"))
if COPTER_DIR not in os.sys.path:
    os.sys.path.insert(0, COPTER_DIR)

from structures import (
    AgentHelperParameters,
    AgentParameters,
    acc_action_dimensions,
    scheduled_epsilon,
)
from sor_agent import SORACC
from sor_replay import SORReplayConfig, StructuredSORReplayBuffer


DEFAULT_SOR_ACC_PARAMETER = AgentParameters(state_dim=18, kmin_dim=6, kmax_dim=4, pmax_dim=10, learning_rate=1e-3, gamma=0.95)


class SORAgentHelper:
    def __init__(
        self,
        node_number: int,
        ahp: AgentHelperParameters,
        model_dir: str,
        exp_name: str = "sor_acc_experiment",
        online: bool = True,
        network_helper=None,
        sor_config: SORReplayConfig = None,
        lambda_cons: float = 0.01,
        lambda_reg: float = 0.001,
        drift_reg_threshold: float = 0.5,
        ref_update_interval: int = 256,
        sync_interval: int = 8,
        save_buffer_every: int = 5,
        acc_hidden_dims=None,
        acc_action_space="legacy",
        run_id: str = None,
        phase: str = None,
        config_hash: str = None,
    ):
        self.node_number = node_number
        self.p = ahp
        self.model_dir = model_dir if model_dir is not None else "sor_models"
        self.exp_name = exp_name
        self.online = online
        self.network_helper = network_helper
        self.run_id = run_id
        self.phase = phase
        self.config_hash = config_hash
        self.acc_action_space = acc_action_space
        self.sor_config = sor_config or SORReplayConfig(rb_size=ahp.rb_size, rb_size_global=ahp.rb_size_global)
        action_dims = acc_action_dimensions(acc_action_space)
        self.acc_parameters = replace(
            DEFAULT_SOR_ACC_PARAMETER,
            kmin_dim=action_dims[0],
            kmax_dim=action_dims[1],
            pmax_dim=(action_dims[2] if len(action_dims) == 3 else 1),
            hidden_dims=(
                tuple(acc_hidden_dims)
                if acc_hidden_dims
                else DEFAULT_SOR_ACC_PARAMETER.hidden_dims
            ),
        )
        self.agent_pool = [
            SORACC(
                f"{self.exp_name}_SORACC_{port_idx}",
                self.acc_parameters,
                lambda_cons=lambda_cons,
                lambda_reg=lambda_reg,
                drift_reg_threshold=drift_reg_threshold,
                ref_update_interval=ref_update_interval,
                action_space=acc_action_space,
            )
            for port_idx in range(node_number)
        ]
        # Every port receives the same total transition budget as ACC's FIFO
        # replay.  Without the total cap, rb_size was applied per cluster and
        # SOR could retain max_clusters times more experience.
        local_cfg = replace(
            self.sor_config,
            global_total_cap=self.p.rb_size,
            recent_size=min(self.sor_config.recent_size, self.p.rb_size),
            boundary_size=min(self.sor_config.boundary_size, self.p.rb_size),
        )
        self.rb_pool = [StructuredSORReplayBuffer(local_cfg) for _ in range(node_number)]
        global_cfg = replace(
            self.sor_config,
            rb_size=self.sor_config.rb_size_global,
            boundary_size=max(self.sor_config.boundary_size, self.sor_config.rb_size_global // 5),
            recent_size=max(self.sor_config.recent_size, self.sor_config.rb_size_global // 50),
            global_total_cap=self.sor_config.rb_size_global,
        )
        self.global_memory = StructuredSORReplayBuffer(global_cfg)
        self.global_train_step = 0
        self.global_env_step = 0
        self.phase_env_step = 0
        self.epsilon = self.p.epsilon_start
        self._train_call_count_since_save = 0
        self._recent_rewards = deque(maxlen=self.p.reward_window)
        self._recent_losses = deque(maxlen=self.p.reward_window)
        self._epoch_port_losses = {}
        self._epoch_port_rewards = {}
        self._epoch_port_q_diagnostics = {}
        self._epoch_loss_components = {}
        self._q_inflation_seen_ports = set()
        self.epoch = 0
        self._tb = None
        self.sync_interval = max(1, int(sync_interval))
        self.save_buffer_every = max(1, int(save_buffer_every))
        self._train_call_count_since_sync = 0
        self._target_update_count = 0
        self._last_target_update_step = 0
        self._started_at = time.time()

    def attach_tb(self, tb_writer):
        self._tb = tb_writer

    def _train_state_path(self):
        return os.path.join(self.model_dir, f"{self.exp_name}_train_state.json")

    def _rb_path(self, port_idx):
        return os.path.join(self.model_dir, f"{self.exp_name}_sor_rb_port{port_idx}.pkl")

    def _global_rb_path(self):
        return os.path.join(self.model_dir, f"{self.exp_name}_sor_global_rb.pkl")

    def _rng_state_path(self):
        return os.path.join(self.model_dir, f"{self.exp_name}_sor_rng.pkl")

    def _metrics_path(self):
        return os.path.join(self.model_dir, f"{self.exp_name}_metrics.jsonl")

    @staticmethod
    def _atomic_write_bytes(path, data: bytes):
        tmp = path + ".tmp"
        with open(tmp, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)

    @staticmethod
    def _atomic_write_text(path, text: str):
        tmp = path + ".tmp"
        with open(tmp, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)

    def get_current_epsilon(self) -> float:
        self.global_env_step += 1
        self.phase_env_step += 1
        self.epsilon = scheduled_epsilon(
            self.p.epsilon_start,
            self.p.epsilon_end,
            self.p.epsilon_decay_steps,
            self.global_env_step,
            self.phase_env_step,
            self.p.epsilon_schedule,
        )
        return self.epsilon

    def decide(self, port_states, epsi=0.1):
        paras = []
        actions = []
        for port_idx, agent in enumerate(self.agent_pool):
            para, action = agent.select_action(port_states[port_idx], epsilon=epsi)
            paras.append(para)
            actions.append(action)
        return paras, actions

    def record(self, port_idx, state, action, next_state):
        if not self.online:
            logger.warning(f"SORAgentHelper is offline. Experience recording skipped for port {port_idx}.")
            return
        reward = self.network_helper.get_port_current_reward(port_idx=port_idx)
        embedding = self.agent_pool[port_idx].encode_state(state)
        transition = self.rb_pool[port_idx].push(
            state, action, reward, next_state, embedding, stream_id=port_idx,
            task_label=self.phase or "unknown",
        )
        self.global_memory.push(
            state,
            action,
            reward,
            next_state,
            embedding,
            td_error=transition.td_error,
            stream_id=port_idx,
            task_label=self.phase or "unknown",
        )
        logger.info(
            f"SOR recorded port={port_idx} reward={reward:.3f} cluster={transition.cluster_id} boundary={transition.boundary}"
        )

    def sync(self):
        # Sample ONCE from global memory and broadcast to all ports.
        # The previous per-port sampling cost O(n_ports * |global|) per train call.
        if len(self.global_memory) == 0:
            return
        sampled = self.global_memory.sample(min(self.p.sync_down_size, len(self.global_memory)))
        if not sampled:
            return
        # Use push_existing to skip prototype/drift recomputation per-port; the
        # transition already carries cluster_id / embedding from the source
        # buffer, so we only need to enforce capacity.
        for rb in self.rb_pool:
            for transition in sampled:
                rb.push_existing(transition)
        logger.info(f"SOR synced {len(sampled)} global structured experiences to {len(self.rb_pool)} ports")

    def maybe_sync(self) -> bool:
        """Throttled sync: only broadcasts global -> per-port every
        sync_interval calls. Returns True if a sync was actually performed.
        """
        self._train_call_count_since_sync += 1
        if self._train_call_count_since_sync < self.sync_interval:
            return False
        self._train_call_count_since_sync = 0
        self.sync()
        return True

    def train(self, current_step: int):
        sample_size = self.p.train_set_size
        any_trained = False
        per_port_loss = {}
        per_port_reward = {}
        for port_idx, agent in enumerate(self.agent_pool):
            replay = self.rb_pool[port_idx]
            if len(replay) <= sample_size:
                logger.warning(f"SOR agent {agent.name} has insufficient experiences for training.")
                continue
            transitions = replay.sample(sample_size)
            states, actions, rewards, next_states, _, cluster_ids, _ = replay.to_arrays(transitions)
            result = agent.train_model(
                states,
                actions,
                rewards,
                next_states,
                cluster_ids=cluster_ids,
                prototypes=replay.prototype_manager.prototypes,
                drift_scores=replay.drift_tracker.drift_scores,
            )
            replay.update_td_errors(transitions, result["td_errors"])
            self._recent_losses.append(result["loss"])
            per_port_loss[port_idx] = result["loss"]
            self._epoch_port_losses.setdefault(port_idx, []).append(
                float(result["loss"])
            )
            self._epoch_loss_components.setdefault(port_idx, []).append({
                "td": float(result["loss_td"]),
                "cons": float(result["loss_cons"]),
                "reg": float(result["loss_reg"]),
            })
            batch_mean_reward = float(np.mean(rewards))
            self._recent_rewards.append(batch_mean_reward)
            per_port_reward[port_idx] = batch_mean_reward
            self._epoch_port_rewards.setdefault(port_idx, []).append(
                batch_mean_reward
            )
            inflation_limit = (
                result["expected_q_bound"] * self.p.q_inflation_factor
            )
            diagnostic = {
                key: result[key]
                for key in ("q_prediction", "q_target", "td_error")
            }
            diagnostic.update({
                "expected_q_bound": float(result["expected_q_bound"]),
                "inflation_limit": float(inflation_limit),
                "inflated": bool(
                    result["q_prediction"]["abs_p95"] > inflation_limit
                    or result["q_target"]["abs_p95"] > inflation_limit
                    or result["q_prediction"]["abs_max"] > inflation_limit
                    or result["q_target"]["abs_max"] > inflation_limit
                ),
            })
            self._epoch_port_q_diagnostics.setdefault(port_idx, []).append(
                diagnostic
            )
            any_trained = True
            logger.info(
                f"SOR trained {agent.name}: loss={result['loss']:.6f}, td={result['loss_td']:.6f}, cons={result['loss_cons']:.6f}, reg={result['loss_reg']:.6f}"
            )
        if any_trained:
            self.global_train_step += 1
            target_updated = (
                self.global_train_step % self.p.target_update_interval == 0
            )
            if target_updated:
                for agent in self.agent_pool:
                    agent.update_target_network()
                self._target_update_count += 1
                self._last_target_update_step = self.global_train_step
                logger.info(
                    "SOR target networks synchronized at global_train_step={} "
                    "(interval={}).",
                    self.global_train_step,
                    self.p.target_update_interval,
                )
            if self._epoch_port_q_diagnostics:
                current = {
                    port: samples[-1]
                    for port, samples in self._epoch_port_q_diagnostics.items()
                    if samples
                }
                inflated = [
                    port for port, detail in current.items()
                    if detail["inflated"]
                ]
                should_log = (
                    self.global_train_step == 1
                    or self.global_train_step % self.p.q_log_interval == 0
                    or target_updated
                )
                newly_inflated = [
                    port for port in inflated
                    if port not in self._q_inflation_seen_ports
                ]
                self._q_inflation_seen_ports.update(inflated)
                if should_log or newly_inflated:
                    log_method = logger.warning if inflated else logger.info
                    log_method(
                        "SOR Q diagnostics global_train_step={}: ports={}, "
                        "q_pred_abs_max={:.4f}, q_target_abs_max={:.4f}, "
                        "td_abs_p95_max={:.4f}, inflated_ports={}",
                        self.global_train_step,
                        len(current),
                        max(v["q_prediction"]["abs_max"] for v in current.values()),
                        max(v["q_target"]["abs_max"] for v in current.values()),
                        max(v["td_error"]["abs_p95"] for v in current.values()),
                        len(inflated),
                    )
            self._train_call_count_since_save += 1
            self._log_train_metrics(current_step, per_port_loss, per_port_reward)
            if self._train_call_count_since_save >= self.p.state_save_interval:
                # Only persist the lightweight train_state during training;
                # full buffer pickles are written once per epoch in save().
                self._save_train_state()
                self._train_call_count_since_save = 0

    def load(self, override_name=None):
        for agent in self.agent_pool:
            agent.load_model(self.model_dir, override_name)
        missing_local = []
        for port_idx, replay in enumerate(self.rb_pool):
            path = self._rb_path(port_idx)
            if os.path.exists(path):
                try:
                    replay.load(path)
                    logger.info(
                        f"Loaded SOR local replay port={port_idx} "
                        f"size={len(replay)} from {path}"
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to restore required SOR local replay {path}"
                    ) from exc
            else:
                missing_local.append(path)
        global_path = self._global_rb_path()
        if os.path.exists(global_path):
            try:
                self.global_memory.load(global_path)
                logger.info(f"Loaded SOR global replay from {global_path}")
            except Exception as exc:
                logger.warning(f"Failed to load SOR global replay {global_path}: {exc}")
        self._load_train_state()
        if self.global_train_step > 0:
            if missing_local:
                raise RuntimeError(
                    "Missing local SOR replay snapshots; refusing an inexact "
                    f"restart ({len(missing_local)} files missing)"
                )
            if not os.path.exists(global_path):
                raise RuntimeError(
                    "Missing global SOR replay snapshot; refusing an inexact restart"
                )
        self._load_rng_state()
        # ref_update_interval is defined over continual training, not one
        # short Python invocation/episode.  Restore its clock from the shared
        # persistent counter so reference updates do not reset every epoch.
        for agent in self.agent_pool:
            agent.train_steps = self.global_train_step
        self.epoch += 1

    def save(self):
        for agent in self.agent_pool:
            agent.save_model(self.model_dir)
        # Formal continual experiments use cadence=1 so every episode is an
        # exact resumable checkpoint.  Larger values remain available only for
        # exploratory runs where approximate restart is acceptable.
        write_buffer = (self.epoch % self.save_buffer_every == 0)
        self._save_train_state_and_buffers(write_buffer=write_buffer)

    def append_epoch_metrics(self, extra: dict = None):
        mean_reward = float(np.mean(self._recent_rewards)) if self._recent_rewards else None
        mean_loss = float(np.mean(self._recent_losses)) if self._recent_losses else None
        record = {
            "epoch": int(self.epoch),
            "global_step": int(self.global_train_step),
            "global_train_step": int(self.global_train_step),
            "epsilon": float(self.epsilon),
            "mean_reward": mean_reward,
            "mean_loss": mean_loss,
            "global_buffer_size": len(self.global_memory),
            "global_env_step": int(self.global_env_step),
            "phase_env_step": int(self.phase_env_step),
            "run_id": self.run_id,
            "phase": self.phase,
            "config_hash": self.config_hash,
            "action_space": self.acc_action_space,
            "epsilon_schedule": self.p.epsilon_schedule,
            "target_update_interval": int(self.p.target_update_interval),
            "target_update_count": int(self._target_update_count),
            "last_target_update_step": int(self._last_target_update_step),
            "wall_time_seconds": float(time.time() - self._started_at),
            "process_max_rss_mb": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
            "replay": {
                "local_capacity": int(self.p.rb_size),
                "global_capacity": int(self.p.rb_size_global),
                "local_size_total": sum(len(rb) for rb in self.rb_pool),
                "global": self.global_memory.diagnostics(),
            },
            "train_port_metrics": {
                str(port): self._summarize_port_training(port, losses)
                for port, losses in sorted(self._epoch_port_losses.items())
                if losses
            },
        }
        if extra:
            record.update(extra)
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            with open(self._metrics_path(), "a") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to append SOR metrics: {exc}")
        if self._tb is not None:
            try:
                step = int(self.global_train_step)
                for key, value in record.items():
                    if isinstance(value, (int, float)) and value is not None:
                        self._tb.add_scalar(f"epoch/{key}", value, step)
                self._tb.flush()
            except Exception as exc:
                logger.warning(f"tensorboard SOR epoch log failed: {exc}")
        return record

    def _save_train_state(self):
        os.makedirs(self.model_dir, exist_ok=True)
        state = {
            "global_train_step": int(self.global_train_step),
            "global_env_step": int(self.global_env_step),
            "phase_env_step": int(self.phase_env_step),
            "epsilon": float(self.epsilon),
            "epsilon_schedule": self.p.epsilon_schedule,
            "epoch": int(self.epoch),
            "run_id": self.run_id,
            "phase": self.phase,
            "config_hash": self.config_hash,
            "action_space": self.acc_action_space,
            "node_number": int(self.node_number),
            "replay_size_per_port": [len(rb) for rb in self.rb_pool],
            "global_replay_size": len(self.global_memory),
            "local_replay_capacity": int(self.p.rb_size),
            "global_replay_capacity": int(self.p.rb_size_global),
            "target_update_interval": int(self.p.target_update_interval),
            "target_update_count": int(self._target_update_count),
            "last_target_update_step": int(self._last_target_update_step),
        }
        try:
            self._atomic_write_text(self._train_state_path(), json.dumps(state))
        except Exception as exc:
            logger.warning(f"Failed to save SOR train_state: {exc}")

    def _save_train_state_and_buffers(self, write_buffer: bool = True):
        os.makedirs(self.model_dir, exist_ok=True)
        if write_buffer:
            try:
                t0 = time.time()
                for port_idx, replay in enumerate(self.rb_pool):
                    self._atomic_write_bytes(
                        self._rb_path(port_idx),
                        pickle.dumps(
                            replay.state_dict(), protocol=pickle.HIGHEST_PROTOCOL
                        ),
                    )
                self._atomic_write_bytes(
                    self._global_rb_path(),
                    pickle.dumps(
                        self.global_memory.state_dict(),
                        protocol=pickle.HIGHEST_PROTOCOL,
                    ),
                )
                self._save_rng_state()
                logger.info(
                    f"SOR full replay checkpoint written in "
                    f"{time.time() - t0:.2f}s "
                    f"(local_total={sum(len(rb) for rb in self.rb_pool)}, "
                    f"global={len(self.global_memory)})"
                )
            except Exception as exc:
                raise RuntimeError("Failed to save complete SOR replay state") from exc
        else:
            logger.info(f"SOR epoch={self.epoch} skipping global replay pickle (cadence={self.save_buffer_every})")
        self._save_train_state()

    def _load_train_state(self):
        path = self._train_state_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as handle:
                state = json.load(handle)
            saved_space = state.get("action_space", "legacy")
            if saved_space != self.acc_action_space:
                raise ValueError(
                    f"SOR action-space mismatch: checkpoint={saved_space}, "
                    f"requested={self.acc_action_space}"
                )
            saved_interval = int(
                state.get("target_update_interval", self.p.target_update_interval)
            )
            if saved_interval != self.p.target_update_interval:
                raise ValueError(
                    f"SOR target interval mismatch: checkpoint={saved_interval}, "
                    f"requested={self.p.target_update_interval}"
                )
            self.global_train_step = int(state.get("global_train_step", 0))
            self.global_env_step = int(state.get("global_env_step", 0))
            saved_phase = state.get("phase")
            if self.phase is not None and saved_phase != self.phase:
                self.phase_env_step = 0
                if self.p.epsilon_schedule == "phase":
                    self.epsilon = self.p.epsilon_start
                    logger.info(
                        f"Continual phase changed {saved_phase!r} -> {self.phase!r}; "
                        "resetting the phase-local epsilon schedule."
                    )
                else:
                    self.epsilon = float(state.get("epsilon", self.p.epsilon_start))
                    logger.info(
                        f"Continual phase changed {saved_phase!r} -> {self.phase!r}; "
                        f"continuing global epsilon schedule at {self.epsilon:.4f}."
                    )
            else:
                self.phase_env_step = int(
                    state.get("phase_env_step", state.get("global_env_step", 0))
                )
                self.epsilon = float(state.get("epsilon", self.p.epsilon_start))
            self.epoch = int(state.get("epoch", 0))
            self._target_update_count = int(
                state.get("target_update_count", 0)
            )
            self._last_target_update_step = int(
                state.get("last_target_update_step", 0)
            )
            if self.run_id is None:
                self.run_id = state.get("run_id")
            if self.phase is None:
                self.phase = state.get("phase")
        except Exception as exc:
            raise RuntimeError(f"Failed to load SOR train_state {path}") from exc

    def _save_rng_state(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None
            ),
        }
        self._atomic_write_bytes(
            self._rng_state_path(),
            pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL),
        )

    def _load_rng_state(self):
        path = self._rng_state_path()
        if not os.path.exists(path):
            return
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def _summarize_port_training(self, port, losses):
        q_samples = self._epoch_port_q_diagnostics.get(port, [])
        components = self._epoch_loss_components.get(port, [])
        result = {
            "updates": len(losses),
            "loss_mean": float(np.mean(losses)),
            "loss_median": float(np.median(losses)),
            "loss_max": float(np.max(losses)),
            "batch_reward_mean": (
                float(np.mean(self._epoch_port_rewards.get(port, [])))
                if self._epoch_port_rewards.get(port) else None
            ),
            "loss_td_mean": float(np.mean([v["td"] for v in components])),
            "loss_cons_mean": float(np.mean([v["cons"] for v in components])),
            "loss_reg_mean": float(np.mean([v["reg"] for v in components])),
            "replay": self.rb_pool[port].diagnostics(),
        }
        if not q_samples:
            return result

        def values(group, metric):
            return [sample[group][metric] for sample in q_samples]

        result.update({
            "q_prediction_mean": float(np.mean(values("q_prediction", "mean"))),
            "q_prediction_abs_p95_max": float(np.max(values("q_prediction", "abs_p95"))),
            "q_prediction_abs_max": float(np.max(values("q_prediction", "abs_max"))),
            "q_target_mean": float(np.mean(values("q_target", "mean"))),
            "q_target_abs_p95_max": float(np.max(values("q_target", "abs_p95"))),
            "q_target_abs_max": float(np.max(values("q_target", "abs_max"))),
            "td_error_abs_mean": float(np.mean(values("td_error", "abs_mean"))),
            "td_error_abs_p95_max": float(np.max(values("td_error", "abs_p95"))),
            "td_error_abs_max": float(np.max(values("td_error", "abs_max"))),
            "expected_q_bound": float(q_samples[-1]["expected_q_bound"]),
            "q_inflation_limit": float(q_samples[-1]["inflation_limit"]),
            "q_inflation_events": sum(bool(v["inflated"]) for v in q_samples),
            "q_inflated": any(bool(v["inflated"]) for v in q_samples),
        })
        return result

    def _log_train_metrics(self, current_step, per_port_loss, per_port_reward):
        if self._tb is None:
            return
        try:
            step = int(self.global_train_step)
            self._tb.add_scalar("train/epsilon", float(self.epsilon), step)
            self._tb.add_scalar("train/env_step", int(current_step), step)
            self._tb.add_scalar("train/global_buffer_size", len(self.global_memory), step)
            if per_port_loss:
                self._tb.add_scalar("train/loss", float(np.mean(list(per_port_loss.values()))), step)
                for port_idx, value in per_port_loss.items():
                    self._tb.add_scalar(f"train/loss_port{port_idx}", value, step)
            if per_port_reward:
                self._tb.add_scalar("train/reward", float(np.mean(list(per_port_reward.values()))), step)
                for port_idx, value in per_port_reward.items():
                    self._tb.add_scalar(f"train/reward_port{port_idx}", value, step)
            for port_idx, replay in enumerate(self.rb_pool):
                self._tb.add_scalar(f"train/buffer_size_port{port_idx}", len(replay), step)
        except Exception as exc:
            logger.warning(f"tensorboard SOR train log failed: {exc}")
