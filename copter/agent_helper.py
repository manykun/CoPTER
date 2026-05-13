# Author: Tianyu Zuo (@amefumi)
# Email: ty.zuo@outlook.com

import numpy as np
import os
import random
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
            fmap_dir: str = None  # 确保保留fmap_dir参数
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
                agent = ACC(f"{self.exp_name}_ACC_{port_idx}", DEFAULT_ACC_PARAMETER)
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
        sample_size = self.p.train_set_size
        for port_idx, agent in enumerate(self.agent_pool):
            # 判断当前端口的经验回放缓冲区是否有足够多的样本
            if len(self.rb_pool[port_idx]) > sample_size:
                # 从经验回放缓冲区采样训练数据
                states, actions, rewards, next_states = self.rb_pool[port_idx].sample(sample_size)

                logger.info(f"Training agent {agent.name} with {len(states)} samples.")
                agent.train_model(states, actions, rewards, next_states)
                # 定期更新目标网络
                if current_step % self.p.target_update_interval == 0:
                    logger.info(f"Updating target network for agent {agent.name} at step {current_step}.")
                    agent.update_target_network()
            else:
                # 经验不足需要提前调用sync()
                logger.warning(f"Agent {agent.name} has insufficient experiences for training. Make sure to call sync() before training.")


    def load(self, override_name=None):
        # Load all agents' models from the model directory
        for agent in self.agent_pool:
            agent.load_model(self.model_dir, override_name)
            

        # # Load replay buffers if available
        # for i, rb in enumerate(self.rb_pool):
        #     rb_path = os.path.join(self.model_dir, f"{self.exp_name}_{self.agent_pool[i].name}_rb.npy")
        #     if os.path.exists(rb_path):
        #         rb_data = np.load(rb_path, allow_pickle=True)
        #         rb.push_batch(rb_data.tolist())
        #         logger.info(f"Loaded replay buffer for agent {self.agent_pool[i].name} from {rb_path}")
        #     else:
        #         logger.warning(f"No replay buffer found for agent {self.agent_pool[i].name} at {rb_path}")

        # # Load shared experience if available
        # shared_rb_path = os.path.join(self.model_dir, f"{self.exp_name}_shared_replay_buffer.npy")
        # if os.path.exists(shared_rb_path):
        #     shared_rb_data = np.load(shared_rb_path, allow_pickle=True)
        #     self.shared_rb.push_batch(shared_rb_data.tolist())
        #     logger.info(f"Loaded shared replay buffer from {shared_rb_path}")
        # else:
        #     logger.warning(f"No shared replay buffer found at {shared_rb_path}. Using empty shared replay buffer.")


    def save(self):
        # Save all agents' models from the model directory
        for agent in self.agent_pool:
            agent.save_model(self.model_dir)

        # # Save replay buffers if available
        # for i, rb in enumerate(self.rb_pool):
        #     rb_path = os.path.join(self.model_dir, f"{self.exp_name}_{self.agent_pool[i].name}_rb")
        #     np.save(rb_path, list(rb.buffer))
        #     logger.info(f"Saved replay buffer for agent {self.agent_pool[i].name} at {rb_path}")
            
        # # Save shared experience if available
        # shared_rb_path = os.path.join(self.model_dir, f"{self.exp_name}_shared_replay_buffer")
        # np.save(shared_rb_path, list(self.shared_rb.buffer))
        # logger.info(f"Saved shared replay buffer at {shared_rb_path}")


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
    
