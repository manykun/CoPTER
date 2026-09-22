# Author: Tianyu Zuo (@amefumi)
# Email: ty.zuo@outlook.com

from ns3gym import ns3env
import numpy as np
import json
from collections import deque
from loguru import logger
from structures import (
    NetworkHelperParameters,
    DCQCNParameters,
    PortObservation,
    calculate_tail_safe_reward,
)


class NetworkHelper:
    def __init__(self, ns3_socket, nhp: NetworkHelperParameters):
        # Store the NetworkHelperParameters
        self.nhp = nhp
        logger.info(f"NetworkHelper initialized with parameters: {self.nhp}")
        
        # Initialize ns3-gym environment
        self.env = ns3env.Ns3Env(port=ns3_socket, startSim=False)
        logger.info("Waiting for ns3 simulator to start...")
        self.env.reset()

        # Observation and action space
        self.ob_space = self.env.observation_space
        self.ac_space = self.env.action_space
        logger.info(f"Observation space: {self.ob_space} -> shape = {self.ac_space.shape}")
        logger.info(f"Action space: {self.ac_space} -> shape = {self.ac_space.shape}")

        # Switch ports (equals to numbers of Agents)
        self.n_port = self.ob_space.shape[0] // nhp.port_states
        logger.info(f"Number of switch ports: {self.n_port}")

        # Observation history as `State`
        self.obs_history = [deque(maxlen=nhp.state_observations + 1) for _ in range(self.n_port)]
        logger.info(f"Observation history sizes: {len(self.obs_history)}")

        # Windowed statistics for queue length and ECN rate
        # These capture the temporal dynamics that instantaneous sampling misses
        self.window_size = 32  # Number of steps to average over
        self.qlen_window = [deque(maxlen=self.window_size) for _ in range(self.n_port)]
        self.ecn_window = [deque(maxlen=self.window_size) for _ in range(self.n_port)]
        self.txrate_window = [deque(maxlen=self.window_size) for _ in range(self.n_port)]

        # Initialize an empty action for next step
        self.action = [0.0] * (self.n_port * self.nhp.port_actions)   
        self.action_port_bitmap = [0] * self.n_port  # Track the port index for each action

        # Port identifier map for logging and analysis
        self.port_identifier_map = {}  # port_idx → "switch_id-connected_node_id"


    def close_env(self):
        self.env.close()
        logger.info("ns3-gym environment closed.")

    # 为指定端口设置动作
    def configurator(self, curr_step, port_idx, port_action: DCQCNParameters):
        """
        Insert the action of a single port into the action list in accordance with the port index.
        Input Definition:
        - `port_idx`: The index of the port (0 to n_port-1).
        - `port_action`: A DCQCNParameters for the specified port.
        """

        if port_idx < 0 or port_idx >= self.n_port:
            raise ValueError(f"Invalid port index: {port_idx}. Must be between 0 and {self.n_port - 1}.")
        
        start_index = port_idx * self.nhp.port_actions
        end_index = start_index + self.nhp.port_actions
        self.action[start_index:end_index] = port_action.k_min_norm, port_action.k_max_norm, port_action.p_max
        # Per-port INFO logging produces hundreds of thousands of lines on the
        # 256-host topology. Keep it available only when TRACE is requested.
        logger.trace(f"Step {curr_step} - Port {port_idx} - Action set to {port_action}.")
        self.action_port_bitmap[port_idx] = 1  # Mark the port as having an action set

    # 执行动作并获取新state
    def monitor(self, curr_step):
        """
        Step actions, monitor the environments, and save the observation history.
        Input Definition:
        - `curr_step`: The current step in the simulation (0 for initial observation, >0 for action).
        """

        if curr_step == 0:
            # If current step is 0, reset the environment and get the initial observation
            obs = self.env.reset()
            obs = np.array(obs)
            done = False
            # 2. 发送空动作（需确保所有端口标记为已设置动作）
            self.action = [0.0] * (self.n_port * self.nhp.port_actions)  
            self.action_port_bitmap = [1] * self.n_port  # 标记所有端口已设置（空动作）
            
            # 3. 首次step，获取环境信息（含port_identifiers）
            obs, _, done, info = self.env.step(self.action)
            logger.info(f"NS3 返回的 info 类型: {type (info)}") # 查看是 dict 还是 str
            logger.info(f"NS3 返回的 info 完整内容: {info}") # 打印完整信息
            # 关键：将字符串info解析为JSON字典
            parsed_info = {}
            if isinstance(info, str):
                try:
                    # 尝试解析JSON字符串
                    parsed_info = json.loads(info)
                    logger.info(f"成功解析info为JSON: {parsed_info}")
                except json.JSONDecodeError:
                    # 解析失败（如纯时间字符串）
                    logger.error(f"info字符串无法解析为JSON: {info}")
                    parsed_info = {"raw_info": info}
            else:
                parsed_info = info  
            if 'port_identifiers' in parsed_info:
                self.port_identifier_map = {
                    idx: ident for idx, ident in enumerate(parsed_info['port_identifiers'])
                }
                # 关键：打印从NS3获取的原始标识列表
                logger.info(f"从NS3获取到端口标识列表: {parsed_info['port_identifiers']}")
                logger.info(f"端口标识映射表: {self.port_identifier_map}")
            else:
                logger.error("未从NS3获取到port_identifiers，无法正确加载fmap")
                logger.error(f"解析后的info中无port_identifiers，内容: {parsed_info}")
        else:
            # 验证所有端口都设置了动作->执行动作->重置动作位图
            assert sum(self.action_port_bitmap) > self.n_port - 1, "Not all ports have actions set. Please check the configurator."
            obs, _, done, info = self.env.step(self.action)
            # print(self.action)    
            self.action_port_bitmap = [0] * self.n_port  # Reset the action bitmap for the next step

        # ns3-gym normally returns observation=None together with done=True for
        # the terminal notification.  Some ns-3.33/ns3-gym runs close the
        # simulation socket first and incorrectly leave done=False in that
        # final reply.  Once at least one environment step has completed there
        # is no observation to process or action to send, so both variants are
        # terminal.  Keep step 0 strict: a missing initial observation still
        # means that the simulator failed to start correctly.
        if obs is None:
            if done or curr_step > 0:
                if done:
                    logger.info(
                        f"Step {curr_step} - Terminal notification received "
                        "without observation."
                    )
                else:
                    logger.warning(
                        f"Step {curr_step} - ns3-gym returned observation=None "
                        "with done=False; treating the socket-close reply as "
                        "terminal."
                    )
                return True
            raise RuntimeError(
                f"ns3-gym returned observation=None while done={done} at step {curr_step}"
            )

        obs = np.asarray(obs)
        expected_size = self.n_port * self.nhp.port_states
        if obs.ndim != 1 or obs.size != expected_size:
            raise RuntimeError(
                f"Invalid ns3 observation at step {curr_step}: "
                f"shape={obs.shape}, size={obs.size}, expected={expected_size}, done={done}"
            )
        # state归一化并存储
        for port_idx in range(self.n_port):
            start_index = port_idx * self.nhp.port_states
            # ns-3 reports queue occupancy already normalized by the configured
            # switch buffer.  Scaling it again here would double-normalize the
            # state and make otherwise identical experiments depend on whether
            # BUFFER_SIZE happens to be 400 KB.
            port_obs = PortObservation(
                queue_length_norm=float(np.clip(obs[start_index], 0.0, 1.0)),
                tx_rate_norm=obs[start_index + 1],
                ecn_rate_norm=obs[start_index + 2],
                k_min_norm=obs[start_index + 3],
                k_max_norm=obs[start_index + 4],
                p_max=obs[start_index + 5]
            )
            self.obs_history[port_idx].append(port_obs)
            # Update windowed statistics for reward calculation
            self.qlen_window[port_idx].append(port_obs.queue_length_norm)
            self.ecn_window[port_idx].append(port_obs.ecn_rate_norm)
            self.txrate_window[port_idx].append(port_obs.tx_rate_norm)
            logger.trace(f"Step {curr_step} - Port {port_idx} - Observation {port_obs}.")

        logger.info(f"Step {curr_step} - Done {done}")
        return done
    

    def get_port_current_parameters(self, port_idx):
        curr_port_obs = self.obs_history[port_idx][-1]
        para = DCQCNParameters(k_min_norm=curr_port_obs.k_min_norm,
                              k_max_norm=curr_port_obs.k_max_norm,
                              p_max=curr_port_obs.p_max)
        return para


    def get_port_current_observation_list(self, port_idx):
        return self.obs_history[port_idx][-1].to_list()
    

    def get_state_dimension(self):
        return self.nhp.port_states * self.nhp.state_observations


    def get_port_last_state_list(self, port_idx):
        last_port_state_list = []
        for history_idx in [0, 1, 2]:
            last_port_state_list += self.obs_history[port_idx][history_idx].to_list()
        return last_port_state_list


    def get_port_current_state_list(self, port_idx):
        curr_port_state_list = []
        for history_idx in [1, 2, 3]:
            curr_port_state_list += self.obs_history[port_idx][history_idx].to_list()
        return curr_port_state_list
    
    def get_port_current_reward_components(self, port_idx):
        """
        Calculate reward for a single port from windowed statistics.

        Design goals:
        1. Sensitive to DCQCN parameter effects (Kmin/Kmax/Pmax).
        2. Meaningful reward gradient ONLY on congested/active ports.
        3. Values bounded in a stable numeric range.

        Legacy reward components (each in [0, 1]):
          - r_throughput: link utilization (tx_rate clamped to [0, 1]).
          - r_queue    : penalises queue build-up via exp(-k * combined_qlen).
                         Combined_qlen mixes 70% peak + 30% average so bursts
                         are visible AND sustained backlog is punished.
          - r_ecn      : penalises excessive ECN marking. Optimal marking
                         rate is around 0.02-0.05. Uses a smooth bell curve
                         so both under-marking and over-marking are penalised.
        """
        if len(self.qlen_window[port_idx]) > 0:
            avg_qlen   = float(np.mean(self.qlen_window[port_idx]))
            peak_qlen  = float(np.max(self.qlen_window[port_idx]))
            avg_ecn    = float(np.mean(self.ecn_window[port_idx]))
            peak_ecn   = float(np.max(self.ecn_window[port_idx]))
            avg_txrate = float(np.mean(self.txrate_window[port_idx]))
        else:
            curr = self.obs_history[port_idx][-1]
            avg_qlen = peak_qlen = curr.queue_length_norm
            avg_ecn  = peak_ecn  = curr.ecn_rate_norm
            avg_txrate = curr.tx_rate_norm

        # Clamp tx_rate to [0, 1] to fix upstream ns-3 rate-stat overshoot
        # (rate estimator sometimes reports >1.0 with narrow windows).
        tx = max(0.0, min(1.0, avg_txrate))

        # r_throughput: reward utilisation. Slightly nonlinear so approaching
        # 1.0 gives extra reward, encouraging the agent to keep the pipe full.
        r_throughput = tx

        # r_queue: exponential penalty. Coefficient 6.0 makes a 0.1 queue
        # occupancy drop reward from 1.0 to 0.55, giving a strong gradient.
        combined_qlen = 0.7 * peak_qlen + 0.3 * avg_qlen
        r_queue = float(np.exp(-6.0 * combined_qlen))

        # r_ecn: bell-shaped centred at 0.03 (ideal marking rate).
        # Under-marking (ecn≈0) → still ok (≈ 0.60) because queue penalty
        # will bite instead. Over-marking penalised more aggressively.
        # Using a piecewise-smooth function:
        #   ecn=0.00 → 0.60   ecn=0.03 → 1.00   ecn=0.10 → 0.50   ecn=0.30 → 0.06
        ideal_ecn = 0.03
        if avg_ecn <= ideal_ecn:
            r_ecn = 0.6 + 0.4 * (avg_ecn / ideal_ecn)
        else:
            r_ecn = float(np.exp(-7.0 * (avg_ecn - ideal_ecn)))

        # The original weighted profile remains available for reproducing
        # earlier ACC experiments.  The continual-learning experiment uses a
        # common tail-safe objective for both traffic tasks.  Squared queue
        # and ECN costs are deliberately weak at low load and increasingly
        # strong during bursts, allowing the same objective to prefer a
        # permissive action in mixed traffic and a safer balanced action in
        # incast traffic.
        tail_safe = calculate_tail_safe_reward(
            r_throughput,
            avg_qlen,
            peak_qlen,
            avg_ecn,
            peak_ecn,
            self.nhp.reward_queue_lambda,
            self.nhp.reward_ecn_lambda,
        )
        if self.nhp.reward_profile == "tail_safe":
            reward = tail_safe["reward"]
        elif self.nhp.reward_profile == "weighted":
            reward = (
                self.nhp.reward_throughput_weight * r_throughput
                + self.nhp.reward_queue_weight * r_queue
                + self.nhp.reward_ecn_weight * r_ecn
            )
        else:
            raise ValueError(
                f"unsupported reward profile: {self.nhp.reward_profile!r}"
            )

        return {
            "reward": float(reward),
            "throughput": float(r_throughput),
            "queue": float(r_queue),
            "ecn": float(r_ecn),
            "avg_tx_rate": float(avg_txrate),
            "avg_queue": float(avg_qlen),
            "peak_queue": float(peak_qlen),
            "avg_ecn": float(avg_ecn),
            "peak_ecn": float(peak_ecn),
            "combined_queue": float(tail_safe["combined_queue"]),
            "combined_ecn": float(tail_safe["combined_ecn"]),
            "queue_cost_sq": float(tail_safe["queue_cost_sq"]),
            "ecn_cost_sq": float(tail_safe["ecn_cost_sq"]),
            "tail_safe_raw": float(tail_safe["raw"]),
            "tail_safe_clipped": float(tail_safe["reward"]),
        }

    def get_port_current_reward(self, port_idx):
        """Return the scalar reward while preserving the historical API."""
        return self.get_port_current_reward_components(port_idx)["reward"]

    def get_port_congestion_score(self, port_idx):
        """Return a scalar quantifying how "interesting" (congested) a port
        is for reward purposes. Higher = more congested = more DCQCN signal.
        Used to select top-K ports for reward aggregation.
        """
        if len(self.qlen_window[port_idx]) == 0:
            return 0.0
        peak_qlen = float(np.max(self.qlen_window[port_idx]))
        peak_ecn  = float(np.max(self.ecn_window[port_idx]))
        avg_txrate = float(np.mean(self.txrate_window[port_idx]))
        # Any of these signals qualifies a port as "interesting".
        return peak_qlen * 10.0 + peak_ecn * 5.0 + min(1.0, avg_txrate)
    

    def get_n_port(self):
        return self.n_port

    def is_port_active(self, port_idx):
        """Backwards-compat: a port is active if any traffic passed through.
        Prefer `is_port_congested` for reward aggregation."""
        if len(self.txrate_window[port_idx]) > 0:
            return float(np.mean(self.txrate_window[port_idx])) > 0.01
        return self.obs_history[port_idx][-1].queue_length_norm > 0.0

    def is_port_congested(self, port_idx):
        """A port is CONGESTED (i.e. carries a meaningful DCQCN signal) if
        the peak queue length or peak ECN rate exceeded a small threshold
        inside the observation window. Only congested ports get their reward
        counted in the rollout mean — averaging over idle ports washes out
        the policy signal entirely.
        """
        if len(self.qlen_window[port_idx]) == 0:
            return False
        peak_qlen = float(np.max(self.qlen_window[port_idx]))
        peak_ecn  = float(np.max(self.ecn_window[port_idx]))
        # A port qualifies as congested when queue built up beyond a
        # trivial floor OR ECN marking actually occurred.
        return peak_qlen > 0.005 or peak_ecn > 1e-5

    def get_topk_congested_ports(self, k):
        """Return indices of the top-K most-congested ports this window."""
        scores = [(self.get_port_congestion_score(p), p) for p in range(self.n_port)]
        scores.sort(key=lambda x: -x[0])
        return [p for score, p in scores[:k] if score > 0.0]
