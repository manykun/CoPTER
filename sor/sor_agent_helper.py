import json
import os
import pickle
from collections import deque
from dataclasses import replace

import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

COPTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "copter"))
if COPTER_DIR not in os.sys.path:
    os.sys.path.insert(0, COPTER_DIR)

from structures import AgentHelperParameters, AgentParameters
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
    ):
        self.node_number = node_number
        self.p = ahp
        self.model_dir = model_dir if model_dir is not None else "sor_models"
        self.exp_name = exp_name
        self.online = online
        self.network_helper = network_helper
        self.sor_config = sor_config or SORReplayConfig(rb_size=ahp.rb_size, rb_size_global=ahp.rb_size_global)
        self.agent_pool = [
            SORACC(
                f"{self.exp_name}_SORACC_{port_idx}",
                DEFAULT_SOR_ACC_PARAMETER,
                lambda_cons=lambda_cons,
                lambda_reg=lambda_reg,
                drift_reg_threshold=drift_reg_threshold,
                ref_update_interval=ref_update_interval,
            )
            for port_idx in range(node_number)
        ]
        self.rb_pool = [StructuredSORReplayBuffer(self.sor_config) for _ in range(node_number)]
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
        self.epsilon = self.p.epsilon_start
        self._train_call_count_since_save = 0
        self._recent_rewards = deque(maxlen=self.p.reward_window)
        self._recent_losses = deque(maxlen=self.p.reward_window)
        self.epoch = 0
        self._tb = None
        self.sync_interval = max(1, int(sync_interval))
        self.save_buffer_every = max(1, int(save_buffer_every))
        self._train_call_count_since_sync = 0

    def attach_tb(self, tb_writer):
        self._tb = tb_writer

    def _train_state_path(self):
        return os.path.join(self.model_dir, f"{self.exp_name}_train_state.json")

    def _rb_path(self, port_idx):
        return os.path.join(self.model_dir, f"{self.exp_name}_sor_rb_port{port_idx}.pkl")

    def _global_rb_path(self):
        return os.path.join(self.model_dir, f"{self.exp_name}_sor_global_rb.pkl")

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
        # Decay by ENVIRONMENT steps to match the ACC baseline schedule.
        self.global_env_step += 1
        decay = self.p.epsilon_decay_steps
        if decay <= 0:
            self.epsilon = self.p.epsilon_end
            return self.epsilon
        frac = min(1.0, self.global_env_step / decay)
        self.epsilon = self.p.epsilon_start + (self.p.epsilon_end - self.p.epsilon_start) * frac
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
        transition = self.rb_pool[port_idx].push(state, action, reward, next_state, embedding)
        self.global_memory.push(state, action, reward, next_state, embedding, td_error=transition.td_error)
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
            batch_mean_reward = float(np.mean(rewards))
            self._recent_rewards.append(batch_mean_reward)
            per_port_reward[port_idx] = batch_mean_reward
            any_trained = True
            logger.info(
                f"SOR trained {agent.name}: loss={result['loss']:.6f}, td={result['loss_td']:.6f}, cons={result['loss_cons']:.6f}, reg={result['loss_reg']:.6f}"
            )
            if current_step % self.p.target_update_interval == 0:
                agent.update_target_network()

        if any_trained:
            self.global_train_step += 1
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
        global_path = self._global_rb_path()
        if os.path.exists(global_path):
            try:
                self.global_memory.load(global_path)
                logger.info(f"Loaded SOR global replay from {global_path}")
            except Exception as exc:
                logger.warning(f"Failed to load SOR global replay {global_path}: {exc}")
            # Per-port buffers cold-start from empty; sync() is now throttled
            # via maybe_sync() and triggered by train() rather than load(),
            # so startup latency stays bounded regardless of global buffer size.
        self._load_train_state()
        self.epoch += 1

    def save(self):
        for agent in self.agent_pool:
            agent.save_model(self.model_dir)
        # Only persist the heavy global buffer pickle every N epochs to avoid
        # epoch-end stalls; train_state.json is always written.
        write_buffer = (self.epoch % self.save_buffer_every == 0)
        self._save_train_state_and_buffers(write_buffer=write_buffer)

    def append_epoch_metrics(self, extra: dict = None):
        mean_reward = float(np.mean(self._recent_rewards)) if self._recent_rewards else None
        mean_loss = float(np.mean(self._recent_losses)) if self._recent_losses else None
        record = {
            "epoch": int(self.epoch),
            "global_step": int(self.global_train_step),
            "epsilon": float(self.epsilon),
            "mean_reward": mean_reward,
            "mean_loss": mean_loss,
            "global_buffer_size": len(self.global_memory),
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
        state = {"global_train_step": int(self.global_train_step), "global_env_step": int(self.global_env_step), "epsilon": float(self.epsilon), "epoch": int(self.epoch)}
        try:
            self._atomic_write_text(self._train_state_path(), json.dumps(state))
        except Exception as exc:
            logger.warning(f"Failed to save SOR train_state: {exc}")

    def _save_train_state_and_buffers(self, write_buffer: bool = True):
        os.makedirs(self.model_dir, exist_ok=True)
        # Persist only the global structured buffer; per-port buffers are
        # rebuilt from the global buffer via sync() after restart. This cuts
        # per-save IO from O(n_ports * 17MB) to a single file.
        if write_buffer:
            try:
                import time as _time
                t0 = _time.time()
                self.global_memory.save(self._global_rb_path())
                logger.info(f"SOR global replay pickled in {_time.time() - t0:.2f}s (size={len(self.global_memory)})")
            except Exception as exc:
                logger.warning(f"Failed to save SOR global replay: {exc}")
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
            self.global_train_step = int(state.get("global_train_step", 0))
            self.global_env_step = int(state.get("global_env_step", 0))
            self.epsilon = float(state.get("epsilon", self.p.epsilon_start))
            self.epoch = int(state.get("epoch", 0))
        except Exception as exc:
            logger.warning(f"Failed to load SOR train_state {path}: {exc}")

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
