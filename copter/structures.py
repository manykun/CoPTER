from dataclasses import dataclass


@dataclass
class NetworkHelperParameters:
    port_states: int = 6
    port_actions: int = 2
    state_observations: int = 3
    switch_buffer_size: int = 400  # in KB; kept as experiment metadata
    reward_throughput_weight: float = 0.50
    reward_queue_weight: float = 0.30
    reward_ecn_weight: float = 0.20


@dataclass
class AgentHelperParameters:
    train_set_size: int = 64
    sync_up_size: int = 64
    sync_down_size: int = 256
    rb_size: int = 1000
    rb_size_global: int = 100000
    target_update_interval: int = 16
    # ---- Continuous training: epsilon schedule + persistence ----
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000      # global train-call count over which epsilon linearly decays
    state_save_interval: int = 1          # save replay buffer / train_state every N train() calls
    reward_window: int = 200              # sliding window size for mean reward metric



@dataclass
class DCQCNParameters:
    k_min_norm: float = 1
    k_max_norm: float = 1
    p_max: float = 0.2


# ACC and SOR share the same discrete action grid.  Keep the grid in one
# place so forced-action baselines and learned policies always mean the same
# thing.
ACC_KMIN_VALUES = (0.0, 0.0949, 0.2259, 0.4066, 0.6560, 1.0)
ACC_KMAX_VALUES = (0.0, 0.25, 0.5, 1.0)
ACC_PMAX_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def validate_acc_action_indices(indices):
    """Validate and return an ACC action index triple."""
    if len(indices) != 3:
        raise ValueError("ACC action must contain kmin,kmax,pmax indices")
    action = tuple(int(index) for index in indices)
    limits = (len(ACC_KMIN_VALUES), len(ACC_KMAX_VALUES), len(ACC_PMAX_VALUES))
    names = ("kmin", "kmax", "pmax")
    for name, index, limit in zip(names, action, limits):
        if not 0 <= index < limit:
            raise ValueError(f"{name} index {index} is outside [0, {limit - 1}]")
    return action


def acc_action_from_indices(indices):
    """Map a validated ACC action index triple to DCQCN parameters."""
    kmin_index, kmax_index, pmax_index = validate_acc_action_indices(indices)
    return DCQCNParameters(
        ACC_KMIN_VALUES[kmin_index],
        ACC_KMAX_VALUES[kmax_index],
        ACC_PMAX_VALUES[pmax_index],
    )


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
    hidden_dims: tuple = (32, 64, 64, 32)
