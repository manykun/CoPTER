from dataclasses import dataclass


@dataclass
class NetworkHelperParameters:
    port_states: int = 6
    port_actions: int = 2
    state_observations: int = 3
    switch_buffer_size: int = 10000  # in KB


@dataclass
class AgentHelperParameters:
    train_set_size: int = 64
    sync_up_size: int = 64
    sync_down_size: int = 256
    rb_size: int = 1000
    rb_size_global: int = 100000
    target_update_interval: int = 16



@dataclass
class DCQCNParameters:
    k_min_norm: float = 1
    k_max_norm: float = 1
    p_max: float = 0.2


@dataclass
class PortObservation:
    queue_length_norm: float = 0.0
    tx_rate_norm: float = 0.0
    ecn_rate_norm: float = 0.0
    k_min_norm: float = 1.0
    k_max_norm: float = 1.0
    p_max: float = 0.2
    
    def to_list(self):
        return [self.queue_length_norm, self.tx_rate_norm, self.ecn_rate_norm, self.k_min_norm, self.k_max_norm, self.p_max]


@dataclass
class AgentParameters:
    state_dim: int = 18
    kmin_dim: int = 6
    kmax_dim: int = 4
    pmax_dim: int = 10
    learning_rate: float = 1e-3
    gamma: float = 0.95
    kmin_res: int = 40
    kmax_res: int = 60
    
    