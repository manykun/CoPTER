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
    reward_profile: str = "weighted"
    reward_queue_lambda: float = 5.0
    reward_ecn_lambda: float = 5.0


def calculate_tail_safe_reward(
    throughput,
    avg_queue,
    peak_queue,
    avg_ecn,
    peak_ecn,
    queue_lambda=5.0,
    ecn_lambda=5.0,
):
    """Return the common nonlinear reward and its auditable components."""
    combined_queue = 0.3 * avg_queue + 0.7 * peak_queue
    combined_ecn = 0.3 * avg_ecn + 0.7 * peak_ecn
    queue_cost_sq = combined_queue ** 2
    ecn_cost_sq = combined_ecn ** 2
    raw = (
        throughput
        - queue_lambda * queue_cost_sq
        - ecn_lambda * ecn_cost_sq
    )
    return {
        "reward": max(-1.0, min(1.0, raw)),
        "raw": raw,
        "combined_queue": combined_queue,
        "combined_ecn": combined_ecn,
        "queue_cost_sq": queue_cost_sq,
        "ecn_cost_sq": ecn_cost_sq,
    }


@dataclass
class AgentHelperParameters:
    train_set_size: int = 64
    sync_up_size: int = 64
    sync_down_size: int = 256
    rb_size: int = 1000
    rb_size_global: int = 100000
    shared_replay_enabled: bool = True
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


# Legacy ACC and SOR share this discrete action grid. Keep it unchanged so
# existing checkpoints and forced-action baselines retain their meaning.
ACC_KMIN_VALUES = (0.0, 0.0949, 0.2259, 0.4066, 0.6560, 1.0)
ACC_KMAX_VALUES = (0.0, 0.25, 0.5, 1.0)
ACC_PMAX_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# The multiscale action space selects one *valid threshold pair* plus Pmax.
# It is designed for configurations whose base (25 Gbps) ranges are
# Kmin=[5, 50] KB and Kmax=[15, 100] KB. ns-3 applies the existing link-rate
# scaling, so the corresponding 40 Gbps physical profiles are:
#   (8,24), (16,32), (20,40), (24,48), (32,64), (48,96),
#   (57,110), (63,120), (80,160) KB.
# Pairing the thresholds prevents the independent heads from proposing
# Kmin >= Kmax and deliberately covers the sub-32-KB queues observed in the
# forgetting study.
ACC_ACTION_SPACES = ("legacy", "multiscale")
ACC_MULTISCALE_THRESHOLD_PROFILES = (
    (0.0, 0.0),
    (1.0 / 9.0, 1.0 / 17.0),
    (1.0 / 6.0, 2.0 / 17.0),
    (2.0 / 9.0, 3.0 / 17.0),
    (1.0 / 3.0, 5.0 / 17.0),
    (5.0 / 9.0, 9.0 / 17.0),
    (49.0 / 72.0, 43.0 / 68.0),
    (55.0 / 72.0, 12.0 / 17.0),
    (1.0, 1.0),
)
ACC_MULTISCALE_PMAX_VALUES = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


def acc_action_dimensions(action_space="legacy"):
    """Return the categorical head sizes for an ACC action space."""
    if action_space == "legacy":
        return (len(ACC_KMIN_VALUES), len(ACC_KMAX_VALUES), len(ACC_PMAX_VALUES))
    if action_space == "multiscale":
        return (
            len(ACC_MULTISCALE_THRESHOLD_PROFILES),
            len(ACC_MULTISCALE_PMAX_VALUES),
        )
    raise ValueError(
        f"unknown ACC action space {action_space!r}; expected one of "
        f"{ACC_ACTION_SPACES}"
    )


def validate_acc_action_indices(indices, action_space="legacy"):
    """Validate and return categorical ACC action indices."""
    limits = acc_action_dimensions(action_space)
    expected = len(limits)
    if len(indices) != expected:
        labels = "kmin,kmax,pmax" if action_space == "legacy" else "profile,pmax"
        raise ValueError(
            f"{action_space} ACC action must contain {labels} indices"
        )
    action = tuple(int(index) for index in indices)
    names = (
        ("kmin", "kmax", "pmax")
        if action_space == "legacy"
        else ("profile", "pmax")
    )
    for name, index, limit in zip(names, action, limits):
        if not 0 <= index < limit:
            raise ValueError(f"{name} index {index} is outside [0, {limit - 1}]")
    return action


def acc_action_from_indices(indices, action_space="legacy"):
    """Map categorical ACC action indices to normalized DCQCN parameters."""
    action = validate_acc_action_indices(indices, action_space)
    if action_space == "multiscale":
        profile_index, pmax_index = action
        kmin_norm, kmax_norm = ACC_MULTISCALE_THRESHOLD_PROFILES[profile_index]
        return DCQCNParameters(
            kmin_norm,
            kmax_norm,
            ACC_MULTISCALE_PMAX_VALUES[pmax_index],
        )
    kmin_index, kmax_index, pmax_index = action
    return DCQCNParameters(
        ACC_KMIN_VALUES[kmin_index],
        ACC_KMAX_VALUES[kmax_index],
        ACC_PMAX_VALUES[pmax_index],
    )


def describe_acc_action(indices, action_space="legacy"):
    """Return a concise, auditable description of a categorical action."""
    action = validate_acc_action_indices(indices, action_space)
    parameters = acc_action_from_indices(action, action_space)
    return (
        f"indices={action}, Kmin={parameters.k_min_norm:.6f}, "
        f"Kmax={parameters.k_max_norm:.6f}, Pmax={parameters.p_max:.3f}"
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
