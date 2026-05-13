import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from environment.ns3interface import PortStatus
import random


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layers: int, nodes: list[int]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, nodes[0]),
            nn.ReLU(),
        )
        for i in range(1, layers):
            self.net.add_module(f'hidden_layer_{i}', nn.Linear(nodes[i-1], nodes[i]))
            self.net.add_module(f'relu_{i}', nn.ReLU())
        self.net.add_module('output_layer', nn.Linear(nodes[-1], output_dim))

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def  __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def __len__(self): 
        return len(self.buffer)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
class CopterAgentDQN:
    def __init__(
            self,
            global_direction,
            port_env,
            configuration,
    ):    
        # Build DDQN networks
        self.online_net = DQN(configuration["input_dim"], configuration["output_dim"], configuration["layers"], configuration["nodes"]) 
        self.target_net = DQN(configuration["input_dim"], configuration["output_dim"], configuration["layers"], configuration["nodes"])
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=configuration["learning_rate"])
        self.loss_fn = nn.MSELoss()

        self.discount_factor = configuration["discount_factor"]
        self.epsilon = configuration["initial_epsilon"]
        self.epsilon_decay = configuration["epsilon_decay"]
        self.final_epsilon = configuration["final_epsilon"]
        self.buffer = ReplayBuffer(configuration["buffer_capacity"])
        self.batch_size = configuration["batch_size"]
        self.target_update = configuration["target_update"]
        self.global_direction = global_direction
        self.env = port_env


    def get_action(self, state): 
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def update_target_net(self):
        pass

    def train(self):
        pass

    def save(self, path):
        pass

    def upload_memory(self, memory):
        pass

    def download_memory(self, memory):
        pass




configuration = {
    "input_dim": 6,
    "output_dim": 4,
    "layers": 3,
    "nodes": [16, 32, 64],
    "learning_rate": 3e-4, # 3e-4 is the best learning rate for Adam, hands down.
    "initial_epsilon": 1.0,
    "epsilon_decay": 0.999,
    "final_epsilon": 0.01,
    "discount_factor": 0.95,
    "buffer_capacity": 100_000,
    "batch_size": 64,
    "target_update": 10,
}

