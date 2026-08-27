# Author: Tianyu Zuo (@amefumi)
# Email: ty.zuo@outlook.com

import numpy as np
import os
import json
import pickle
import random
import time
from dataclasses import replace
from collections import deque
from loguru import logger

from structures import NetworkHelperParameters, DCQCNParameters, PortObservation, AgentParameters, AgentHelperParameters
from agent import Agent, ACC, CoPTER


DEFAULT_ACC_PARAMETER = AgentParameters(state_dim=18, kmin_dim=6, kmax_dim=4, pmax_dim=10, learning_rate=1e-3, gamma=0.95)
# copter修改为三头
DEFAULT_COPTER_PARAMETER = AgentParameters(state_dim=18, kmin_dim=5, kmax_dim=10, pmax_dim=10, learning_rate=1e-3, gamma=0.95, kmin_res=40, kmax_res=60)

# 经验回放机制，存储Agent经验
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)


    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))


    def push_batch(self, batch):
        for state, action, reward, next_state in batch:
            self.push(state, action, reward, next_state)


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
        )


    def __len__(self):
        return len(self.buffer)
    


class AgentHelper:
    def __init__(
            self, 
            node_number: int, 
            ahp: AgentHelperParameters,
            model_dir: str,
            mode: str = "ACC",
            exp_name: str = "Default",
            online: bool = True,
            # fmap_dir: str = None
            network_helper: "NetworkHelper" = None,  # 传入NetworkHelper获取标识和fmap路径
            fmap_dir: str = None,  # 确保保留fmap_dir参数
            run_id: str = None,
            phase: str = None,
            config_hash: str = None,
            acc_hidden_dims=None,
            acc_action_space="legacy",
            ):
        # Assert fmap_dir is provided if mode is CoPTER
        if mode == "CoPTER" and fmap_dir is None:
            raise ValueError("fmap_dir must be provided when mode is CoPTER.")

        # Initialize the AgentHelper with the given parameters
        logger.info(f"Initializing AgentHelper with mode: {mode}, node_number: {node_number}, online: {online}")
        
        self.node_number = node_number
        self.p = ahp
        self.model_dir = model_dir if model_dir is not None else "models"

        self.mode = mode
        self.exp_name = exp_name
        self.online = online
        self.run_id = run_id
        self.phase = phase
        self.config_hash = config_hash
        self.acc_action_space = acc_action_space
        self.acc_parameters = replace(
            DEFAULT_ACC_PARAMETER,
            hidden_dims=tuple(acc_hidden_dims) if acc_hidden_dims else DEFAULT_ACC_PARAMETER.hidden_dims,
        )

        # Load fmap if mode is CoPTER
        # self.fmap = self._load_fmap(fmap_dir, 0) if mode == "CoPTER" else None

        # Initialize all agents and their replay buffers (if available)
        self.agent_pool: list[Agent] = []
        self.rb_pool = []
        # for i in range(node_number):
        #     # Initialize agent by mode
        #     self.agent_pool.append(
        #         ACC(f"{self.exp_name}_ACC_{i}", DEFAULT_ACC_PARAMETER) if mode == "ACC" else 
        #         CoPTER(f"{self.exp_name}_CoPTER_{i}", DEFAULT_COPTER_PARAMETER, self.fmap, self.online)
        #         )
        #     self.rb_pool.append(ReplayBuffer(capacity=self.p.rb_size))

        # NOTE: Shared experience means agent periodically syncs its partial experience with a global replay buffer,
        # and samples partial experience from it to its own replay buffer.
        # shared_pool存放global经验，deque数据结构，自动淘汰最早经验
        # self.shared_rb = deque(maxlen=self.p.rb_size_global)，放在后面的循环里面了

        self.network_helper = network_helper
        self.fmap_dir = fmap_dir
        self.port_fmaps = {}  # port_idx → 对应fmap矩阵
        
        for port_idx in range(node_number):
            # 根据模式初始化智能体
            if mode == "ACC":
                agent = ACC(
                    f"{self.exp_name}_ACC_{port_idx}",
                    self.acc_parameters,
                    action_space=self.acc_action_space,
                )
            else:  # CoPTER模式
                agent = CoPTER(
                    f"{self.exp_name}_CoPTER_{port_idx}", 
                    DEFAULT_COPTER_PARAMETER, 
                    f_matrix=np.ones((5, 10)),  # 临时默认矩阵
                    online=self.online
                )
            
            self.agent_pool.append(agent)
            self.rb_pool.append(ReplayBuffer(capacity=self.p.rb_size))

        self.shared_rb = deque(maxlen=self.p.rb_size_global)
        logger.info(
            "ACC shared replay is {} (local={}, global={}, sync_up={}, "
            "sync_down={})",
            "enabled" if self.p.shared_replay_enabled else "disabled",
            self.p.rb_size,
            self.p.rb_size_global,
            self.p.sync_up_size,
            self.p.sync_down_size,
        )

        # ---- Continuous-training state (cross-run persistence) ----
        self.global_train_step = 0           # number of agent_helper.train() calls completed (across runs)
        self.global_env_step = 0             # number of environment steps observed (across runs); drives epsilon decay
        self.phase_env_step = 0              # environment steps in the current continual-learning phase
        self.epsilon = self.p.epsilon_start
        self.train_call_count = 0
        self._train_call_count_since_save = 0
        self._recent_rewards = deque(maxlen=self.p.reward_window)
        self._recent_losses = deque(maxlen=self.p.reward_window)
        # Per-process (one epoch) port-level training diagnostics.  These are
        # deliberately not persisted in train_state: append_epoch_metrics()
        # serializes them at the end of the one-epoch process.
        self._epoch_port_losses = {}
        self._epoch_port_rewards = {}
        # Current "epoch" id (i.e. how many copter.py invocations) - read from train_state if exists
        self.epoch = 0

        # Optional TensorBoard SummaryWriter (set by attach_tb after init in copter.py)
        self._tb = None

    def attach_tb(self, tb_writer):
        """Attach an already-initialized torch.utils.tensorboard.SummaryWriter."""
        self._tb = tb_writer

    # ---------- helpers for persistence paths ----------
    def _train_state_path(self):
        return os.path.join(self.model_dir, f"{self.exp_name}_train_state.json")

    def _rb_path(self, port_idx):
        return os.path.join(self.model_dir, f"{self.exp_name}_rb_port{port_idx}.pkl")

    def _shared_rb_path(self):
        return os.path.join(self.model_dir, f"{self.exp_name}_shared_rb.pkl")

    def _metrics_path(self):
        return os.path.join(self.model_dir, f"{self.exp_name}_metrics.jsonl")

    @staticmethod
    def _atomic_write_bytes(path, data: bytes):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)

    @staticmethod
    def _atomic_write_text(path, text: str):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)

    # ---------- epsilon schedule ----------
    def get_current_epsilon(self) -> float:
        # Exploration restarts at each continual-learning task boundary.  The
        # global counter remains available for audit, while the phase-local
        # counter makes task A and task B receive the same epsilon schedule.
        self.global_env_step += 1
        self.phase_env_step += 1
        decay = self.p.epsilon_decay_steps
        if decay <= 0:
            return self.p.epsilon_end
        frac = min(1.0, self.phase_env_step / decay)
        eps = self.p.epsilon_start + (self.p.epsilon_end - self.p.epsilon_start) * frac
        self.epsilon = eps
        return eps

    # 为所有端口生成决策：遍历每个端口对应的智能体，调用智能体的选择动作方法，收集并返回所有决策
    def decide(self, port_states: list[list[float]], epsi=0.1) -> tuple[list[DCQCNParameters], list[tuple[int, int, int]]]:
        paras: list[DCQCNParameters] = []
        actions: list[tuple[int, int, int]] = []

        for port_idx, agent in enumerate(self.agent_pool):
                        # 延迟加载fmap：首次调用decide且未加载时执行
            if self.mode == "CoPTER" and port_idx not in self.port_fmaps:
                # 先验证端口标识映射表是否存在
                if not self.network_helper.port_identifier_map:
                    logger.error(f"❌ 端口{port_idx}加载fmap失败：未获取到端口标识映射表")
                    self.port_fmaps[port_idx] = np.ones((5, 10))  # 默认矩阵避免崩溃
                    agent.f_matrix = self.port_fmaps[port_idx]
                    continue
                # 获取真实端口标识（此时已从NS3获取）
                port_ident = self.network_helper.port_identifier_map.get(
                    port_idx, 
                    f"default_port_{port_idx}"
                )
                # 加载fmap并缓存
                self.port_fmaps[port_idx] = self._load_fmap(port_ident)
                # 更新CoPTER agent的f_matrix
                agent.f_matrix = self.port_fmaps[port_idx]
                logger.info(f"端口{port_idx}延迟加载fmap完成，标识: {port_ident}")
            para, action = agent.select_action(
                port_states[port_idx], 
                epsilon=epsi, 
            )
            paras.append(para)
            actions.append(action)
        
        return paras, actions
    

    def train(self, current_step: int):
        self.train_call_count += 1
        sample_size = self.p.train_set_size
        any_trained = False
        # Per-train-call metric collectors
        per_port_loss = {}
        per_port_reward = {}
        for port_idx, agent in enumerate(self.agent_pool):
            # 判断当前端口的经验回放缓冲区是否有足够多的样本
            if len(self.rb_pool[port_idx]) > sample_size:
                # 从经验回放缓冲区采样训练数据
                states, actions, rewards, next_states = self.rb_pool[port_idx].sample(sample_size)

                logger.info(f"Training agent {agent.name} with {len(states)} samples.")
                loss = agent.train_model(states, actions, rewards, next_states)
                if loss is not None:
                    self._recent_losses.append(float(loss))
                    per_port_loss[port_idx] = float(loss)
                    self._epoch_port_losses.setdefault(port_idx, []).append(
                        float(loss)
                    )
                # collect mean reward of the sampled batch as a smoothed proxy
                batch_mean_reward = float(np.mean(rewards))
                self._recent_rewards.append(batch_mean_reward)
                per_port_reward[port_idx] = batch_mean_reward
                self._epoch_port_rewards.setdefault(port_idx, []).append(
                    batch_mean_reward
                )
                any_trained = True
                # 定期更新目标网络
                if current_step % self.p.target_update_interval == 0:
                    logger.info(f"Updating target network for agent {agent.name} at step {current_step}.")
                    agent.update_target_network()
            else:
                # 经验不足需要提前调用sync()
                logger.warning(f"Agent {agent.name} has insufficient experiences for training. Make sure to call sync() before training.")

        if any_trained:
            self.global_train_step += 1
            self._train_call_count_since_save += 1
            # ---- TensorBoard logging (per train() call) ----
            if self._tb is not None:
                try:
                    step = int(self.global_train_step)
                    self._tb.add_scalar("train/epsilon", float(self.epsilon), step)
                    self._tb.add_scalar("train/env_step", int(current_step), step)
                    self._tb.add_scalar("train/epoch", int(self.epoch), step)
                    self._tb.add_scalar("train/shared_buffer_size", len(self.shared_rb), step)
                    if per_port_loss:
                        self._tb.add_scalar("train/loss", float(np.mean(list(per_port_loss.values()))), step)
                        for pi, lv in per_port_loss.items():
                            self._tb.add_scalar(f"train/loss_port{pi}", lv, step)
                    if per_port_reward:
                        self._tb.add_scalar("train/reward", float(np.mean(list(per_port_reward.values()))), step)
                        for pi, rv in per_port_reward.items():
                            self._tb.add_scalar(f"train/reward_port{pi}", rv, step)
                    for pi, rb in enumerate(self.rb_pool):
                        self._tb.add_scalar(f"train/buffer_size_port{pi}", len(rb), step)
                except Exception as e:
                    logger.warning(f"tensorboard train log failed: {e}")
            # 按 state_save_interval 落盘 buffer + train_state（防崩溃丢进度）
            if self._train_call_count_since_save >= self.p.state_save_interval:
                self._save_train_state_and_buffers()
                self._train_call_count_since_save = 0


    def load(self, override_name=None):
        # Load all agents' models from the model directory
        for agent in self.agent_pool:
            agent.load_model(self.model_dir, override_name)

        # ---- Load replay buffers (per port) ----
        for i, rb in enumerate(self.rb_pool):
            rb_path = self._rb_path(i)
            if os.path.exists(rb_path):
                try:
                    with open(rb_path, "rb") as f:
                        rb_data = pickle.load(f)
                    rb.push_batch(rb_data)
                    logger.info(f"Loaded replay buffer for port {i} ({len(rb_data)} entries) from {rb_path}")
                except Exception as e:
                    logger.warning(f"Failed to load replay buffer at {rb_path}: {e}")
            else:
                logger.warning(f"No replay buffer found for port {i} at {rb_path} (cold start).")

        # ---- Load shared replay buffer ----
        shared_rb_path = self._shared_rb_path()
        if not self.p.shared_replay_enabled:
            logger.info("Shared replay disabled; not loading {}", shared_rb_path)
        elif os.path.exists(shared_rb_path):
            try:
                with open(shared_rb_path, "rb") as f:
                    shared_data = pickle.load(f)
                for entry in shared_data:
                    self.shared_rb.append(entry)
                logger.info(f"Loaded shared replay buffer ({len(shared_data)} entries) from {shared_rb_path}")
            except Exception as e:
                logger.warning(f"Failed to load shared replay buffer: {e}")
        else:
            logger.warning(f"No shared replay buffer found at {shared_rb_path} (cold start).")

        # ---- Load training state (global_train_step / epsilon / epoch) ----
        ts_path = self._train_state_path()
        if os.path.exists(ts_path):
            try:
                with open(ts_path, "r") as f:
                    ts = json.load(f)
                self.global_train_step = int(ts.get("global_train_step", 0))
                self.global_env_step = int(ts.get("global_env_step", 0))
                saved_phase = ts.get("phase")
                if self.phase is not None and saved_phase != self.phase:
                    self.phase_env_step = 0
                    self.epsilon = self.p.epsilon_start
                    logger.info(
                        f"Continual phase changed {saved_phase!r} -> {self.phase!r}; "
                        "resetting the phase-local epsilon schedule."
                    )
                else:
                    self.phase_env_step = int(
                        ts.get("phase_env_step", ts.get("global_env_step", 0))
                    )
                self.train_call_count = int(ts.get("train_call_count", self.global_train_step))
                if self.phase is None or saved_phase == self.phase:
                    self.epsilon = float(ts.get("epsilon", self.p.epsilon_start))
                self.epoch = int(ts.get("epoch", 0))
                if self.run_id is None:
                    self.run_id = ts.get("run_id")
                if self.phase is None:
                    self.phase = ts.get("phase")
                if self.config_hash is None:
                    self.config_hash = ts.get("config_hash")
                logger.info(
                    f"Resumed training state: global_step={self.global_train_step}, "
                    f"epsilon={self.epsilon:.4f}, epoch={self.epoch}"
                )
            except Exception as e:
                logger.warning(f"Failed to load train_state at {ts_path}: {e}; using defaults.")
        else:
            logger.info(f"No train_state found at {ts_path}; starting fresh from epsilon={self.p.epsilon_start}.")

        # advance epoch counter for THIS run
        self.epoch += 1

    def _save_train_state_and_buffers(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # Replay buffers
        for i, rb in enumerate(self.rb_pool):
            try:
                data = pickle.dumps(list(rb.buffer), protocol=pickle.HIGHEST_PROTOCOL)
                self._atomic_write_bytes(self._rb_path(i), data)
            except Exception as e:
                logger.warning(f"Failed to save replay buffer for port {i}: {e}")
        # Shared replay buffer. Local-only ablations deliberately never read or
        # write this file, preventing stale global experience from leaking in.
        if self.p.shared_replay_enabled:
            try:
                data = pickle.dumps(list(self.shared_rb), protocol=pickle.HIGHEST_PROTOCOL)
                self._atomic_write_bytes(self._shared_rb_path(), data)
            except Exception as e:
                logger.warning(f"Failed to save shared replay buffer: {e}")
        # Train state
        ts = {
            "global_train_step": int(self.global_train_step),
            "global_env_step": int(self.global_env_step),
            "phase_env_step": int(self.phase_env_step),
            "train_call_count": int(self.train_call_count),
            "epsilon": float(self.epsilon),
            "epoch": int(self.epoch),
            "time": time.time(),
            "exp_name": self.exp_name,
            "mode": self.mode,
            "node_number": self.node_number,
            "replay_size_per_port": [len(rb) for rb in self.rb_pool],
            "shared_replay_size": len(self.shared_rb),
            "shared_replay_enabled": bool(self.p.shared_replay_enabled),
        }
        ts.update({
            key: value for key, value in {
                "run_id": self.run_id,
                "phase": self.phase,
                "config_hash": self.config_hash,
            }.items() if value is not None
        })

        try:
            self._atomic_write_text(self._train_state_path(), json.dumps(ts))
        except Exception as e:
            logger.warning(f"Failed to save train_state: {e}")

    def save(self):
        # Save all agents' models
        for agent in self.agent_pool:
            agent.save_model(self.model_dir)
        # Save buffers + train state
        self._save_train_state_and_buffers()

    def append_epoch_metrics(self, extra: dict = None):
        """Called once per epoch (per copter.py invocation) before exit."""
        mean_reward = float(np.mean(self._recent_rewards)) if len(self._recent_rewards) > 0 else None
        mean_loss = float(np.mean(self._recent_losses)) if len(self._recent_losses) > 0 else None
        record = {
            "epoch": int(self.epoch),
            "global_train_step": int(self.global_train_step),
            "global_env_step": int(self.global_env_step),
            "phase_env_step": int(self.phase_env_step),
            "train_call_count": int(self.train_call_count),
            "epsilon": float(self.epsilon),
            "mean_reward": mean_reward,
            "mean_loss": mean_loss,
            "train_port_metrics": {
                str(port): {
                    "updates": len(losses),
                    "loss_mean": float(np.mean(losses)),
                    "loss_median": float(np.median(losses)),
                    "loss_max": float(np.max(losses)),
                    "batch_reward_mean": (
                        float(np.mean(self._epoch_port_rewards.get(port, [])))
                        if self._epoch_port_rewards.get(port) else None
                    ),
                }
                for port, losses in sorted(self._epoch_port_losses.items())
                if losses
            },
            "time": time.time(),
        }
        record.update({
            key: value for key, value in {
                "run_id": self.run_id,
                "phase": self.phase,
                "config_hash": self.config_hash,
            }.items() if value is not None
        })

        if extra:
            record.update(extra)
        try:
            with open(self._metrics_path(), "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.info(f"Epoch metrics appended: {record}")
        except Exception as e:
            logger.warning(f"Failed to append metrics: {e}")
        # ---- TensorBoard epoch-level logging ----
        if self._tb is not None:
            try:
                step = int(self.global_train_step)
                self._tb.add_scalar("epoch/index", int(self.epoch), step)
                self._tb.add_scalar("epoch/epsilon", float(self.epsilon), step)
                self._tb.add_scalar("epoch/shared_buffer_size", len(self.shared_rb), step)
                if mean_reward is not None:
                    self._tb.add_scalar("epoch/mean_reward", mean_reward, step)
                if mean_loss is not None:
                    self._tb.add_scalar("epoch/mean_loss", mean_loss, step)
                if extra:
                    for k, v in extra.items():
                        if isinstance(v, (int, float)):
                            self._tb.add_scalar(f"epoch/{k}", v, step)
                for pi, rb in enumerate(self.rb_pool):
                    self._tb.add_scalar(f"epoch/buffer_size_port{pi}", len(rb), step)
                self._tb.flush()
            except Exception as e:
                logger.warning(f"tensorboard epoch log failed: {e}")
        return record


     # 修改：记录经验时传入fmap和action以计算融合奖励
    def record(self, port_idx, state, action, next_state):
        if self.online:
            # 获取当前端口的fmap（CoPTER模式用）
            fmap = self.port_fmaps.get(port_idx, None) if self.mode == "CoPTER" else None
            
            # 计算奖励（根据模式选择计算方式）
            reward = self.network_helper.get_port_current_reward(
                port_idx=port_idx,
                # mode=self.mode,  # 传递当前模式
                # fmap=fmap,
                # action=action if self.mode == "CoPTER" else None
            )
            
            self.rb_pool[port_idx].push(state, action, reward, next_state)
            logger.info(f"Recorded experience for port {port_idx} (mode: {self.mode}) with reward {reward:.3f}")
        else:
            logger.warning(f"AgentHelper is in offline mode. Experience recording skipped for port {port_idx}.")
       
    # 共享经验池和本地经验池的上传和下载
    def sync(self):
        if not self.p.shared_replay_enabled:
            # Keep the normal record -> train cadence, but train each port only
            # from its own FIFO replay. This isolates the effect of cross-port
            # global experience sharing without clearing local task-A memory.
            return
        # Sync up: Sample local replay buffer to shared replay buffer
        sync_up_size = min(len(self.rb_pool[0]), self.p.sync_up_size)
        for i, agent in enumerate(self.agent_pool):
            states, actions, rewards, next_states = self.rb_pool[i].sample(sync_up_size)
            self.shared_rb.extend(zip(states, actions, rewards, next_states))
            logger.info(f"Agent {agent.name} synced up {sync_up_size} experiences to shared replay buffer. Total shared experiences: {len(self.shared_rb)}")
        
        # Sync down: Sample shared replay buffer to local replay buffer
        sync_down_size = min(len(self.shared_rb), self.p.sync_down_size)
        for i, agent in enumerate(self.agent_pool):
            sampled_experiences = random.sample(self.shared_rb, sync_down_size)
            self.rb_pool[i].push_batch(sampled_experiences)
            logger.info(f"Agent {agent.name} synced down {sync_down_size} experiences from shared replay buffer. Local replay buffer size: {len(self.rb_pool[i])}")


    # @staticmethod
    def _load_fmap(self, port_ident: str) -> np.ndarray:
        # fmap_row = DEFAULT_COPTER_PARAMETER.kmin_dim
        # fmap_col = DEFAULT_COPTER_PARAMETER.kmax_dim

        # # 手动导入文件名
        # fmap_path = os.path.join(fmap_dir, f"mix_webserver_websearch_hadoop_short.fmap") if fmap_dir is not None else None
        # # fmap_path = os.path.join(fmap_dir, f"node_{node_idx}.fmap") if fmap_dir is not None else None

        # if fmap_path is None or os.path.exists(fmap_path) is False:
        #     logger.warning("No existed fmap file. Using default fmaps.")
        #     # 返回全1 相当于没有引导效果
        #     return np.ones((fmap_row, fmap_col), dtype=float)
        # else:
        #     # fmap = np.loadtxt(fmap_path, dtype=float)
        #     # 情况2：读取文件（跳过第一行表头和第一列表头）
        #     fmap_with_first_col = np.loadtxt(
        #         fmap_path,
        #         dtype=float,
        #         delimiter='\t',  # 制表符分隔
        #         skiprows=1       # 跳过表头行
        #     )
        #     fmap = fmap_with_first_col[:, 1:]  # [:, 1:] 表示所有行，从第2列（索引1）开始取
        #     if fmap.shape != (fmap_row, fmap_col):
        #         # If the shape of fmap is not correct, log a warning and return a default fmap
        #         logger.warning(f"fmap file {fmap_path} has wrong shape: {fmap.shape}. Expected shape: ({fmap_row}, {fmap_col}). Using default fmaps.")
        #         return np.ones((fmap_row, fmap_col), dtype=float)
        #     return fmap
        """根据端口标识加载对应fmap文件，跳过第一行和第一列"""
        fmap_row = DEFAULT_COPTER_PARAMETER.kmin_dim  # 预期行数（如4）
        fmap_col = DEFAULT_COPTER_PARAMETER.kmax_dim  # 预期列数（如6）

        if not self.fmap_dir:
            logger.warning("未指定fmap目录，使用全1默认矩阵")
            return np.ones((fmap_row, fmap_col), dtype=float)
        
        # 打印绝对路径用于调试
        fmap_path = os.path.abspath(os.path.join(self.fmap_dir, f"{port_ident}_fmap.txt"))
        logger.info(f"尝试加载fmap: {fmap_path}")
        if not os.path.exists(fmap_path):
            logger.warning(f"端口 {port_ident} 的fmap文件不存在: {fmap_path}，使用默认矩阵")
            return np.ones((fmap_row, fmap_col), dtype=float)
        
        try:
            # 步骤1：先读取文件的第一行（跳过表头），获取总列数
            # 目的：动态确定“除第一列外的所有列”的索引，适配不同列数的文件
            with open(fmap_path, 'r') as f:
                # 跳过第一行表头（skiprows=1对应后续逻辑）
                next(f)  # 读取并丢弃表头行
                first_data_line = next(f).strip()  # 读取第一行数据
                total_cols = len(first_data_line.split())  # 按空格分割，获取总列数
                logger.info(f"fmap文件总行数: {total_cols}（第一列为索引，后续{total_cols-1}列为数据）")
            
            # 步骤2：生成“跳过第一列（索引0），读取剩余所有列”的索引序列
            # 例如：总列数=7 → usecols=[1,2,3,4,5,6]（对应6列数据）
            usecols = list(range(1, total_cols))  # 从索引1到最后一列
            if len(usecols) != fmap_col:
                logger.warning(f"fmap文件数据列数（{len(usecols)}）与预期（{fmap_col}）不匹配，可能读取异常")
            
            # 步骤3：正确调用np.loadtxt读取文件（usecols用整数序列）
            fmap = np.loadtxt(
                fmap_path,
                dtype=float,
                skiprows=1,          # 跳过第一行表头
                usecols=usecols,     # 读取除第一列外的所有列（整数序列）
                delimiter=None,      # 自动识别分隔符（空格/制表符均可）
                encoding='utf-8'     # 避免编码问题
            )

            # 步骤4：验证fmap矩阵形状是否符合预期（4行6列）
            if fmap.shape != (fmap_row, fmap_col):
                logger.error(
                    f"fmap矩阵形状错误！预期 ({fmap_row}, {fmap_col})，"
                    f"实际 ({fmap.shape[0]}, {fmap.shape[1]})，使用默认矩阵"
                )
                return np.ones((fmap_row, fmap_col), dtype=float)
            
            # 步骤5：验证数据有效性（避免全0/全NaN）
            if np.all(fmap == 0) or np.any(np.isnan(fmap)):
                logger.warning(f"fmap数据异常（全0或含NaN），使用默认矩阵")
                return np.ones((fmap_row, fmap_col), dtype=float)
            
            logger.success(f"✅ 成功加载fmap（端口{port_ident}），形状: {fmap.shape}，数据预览:\n{fmap[:2, :2]}")
            return fmap

        except Exception as e:
            logger.error(f"加载fmap失败（端口{port_ident}）: {str(e)}", exc_info=True)
            return np.ones((fmap_row, fmap_col), dtype=float)
    
