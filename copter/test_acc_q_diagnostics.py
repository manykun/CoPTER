#!/usr/bin/env python3
"""Focused regression tests for ACC Q diagnostics and target scheduling."""

import os
import sys
import types
from collections import deque

import numpy as np
import torch

try:
    import loguru  # noqa: F401
except ImportError:
    class SilentLogger:
        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    sys.modules["loguru"] = types.SimpleNamespace(logger=SilentLogger())

try:
    import scipy  # noqa: F401
except ImportError:
    scipy_module = types.ModuleType("scipy")
    interpolate_module = types.ModuleType("scipy.interpolate")
    interpolate_module.RegularGridInterpolator = object
    scipy_module.interpolate = interpolate_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.interpolate"] = interpolate_module

sys.path.insert(0, os.path.dirname(__file__))

from agent import ACC
from agent_helper import AgentHelper, ReplayBuffer
from structures import AgentHelperParameters, AgentParameters


def test_fresh_target_matches_policy():
    agent = ACC(
        "target_init",
        AgentParameters(state_dim=18, hidden_dims=(8,)),
        action_space="multiscale",
    )
    policy = agent.policy_net.state_dict()
    target = agent.target_net.state_dict()
    assert policy.keys() == target.keys()
    assert all(torch.equal(policy[key], target[key]) for key in policy)


def test_acc_exposes_bounded_q_diagnostics():
    agent = ACC(
        "q_metrics",
        AgentParameters(state_dim=18, hidden_dims=(8,), gamma=0.95),
        action_space="multiscale",
    )
    batch = 4
    loss = agent.train_model(
        np.zeros((batch, 18), dtype=np.float32),
        np.zeros((batch, 2), dtype=np.int64),
        np.ones(batch, dtype=np.float32),
        np.zeros((batch, 18), dtype=np.float32),
    )
    diagnostics = agent.last_train_diagnostics
    assert loss >= 0.0
    assert abs(diagnostics["expected_q_bound"] - 20.0) < 1e-9
    for group in ("q_prediction", "q_target", "td_error"):
        assert set(diagnostics[group]) == {
            "mean", "abs_mean", "abs_p95", "abs_max"
        }
        assert all(np.isfinite(value) for value in diagnostics[group].values())


class FakeAgent:
    def __init__(self):
        self.name = "fake"
        self.target_updates = 0
        self.last_train_diagnostics = None

    def train_model(self, states, actions, rewards, next_states):
        stats = {"mean": 1.0, "abs_mean": 1.0, "abs_p95": 1.0, "abs_max": 1.0}
        self.last_train_diagnostics = {
            "q_prediction": dict(stats),
            "q_target": dict(stats),
            "td_error": dict(stats),
            "reward_mean": 0.5,
            "reward_abs_max": 0.5,
            "expected_q_bound": 20.0,
        }
        return 0.25

    def update_target_network(self):
        self.target_updates += 1


def test_target_update_uses_global_optimizer_steps():
    helper = AgentHelper.__new__(AgentHelper)
    helper.p = AgentHelperParameters(
        train_set_size=2,
        target_update_interval=3,
        q_log_interval=1000,
        state_save_interval=1000,
    )
    helper.agent_pool = [FakeAgent()]
    replay = ReplayBuffer(10)
    for _ in range(3):
        replay.push([0.0], [0, 0], 0.5, [0.0])
    helper.rb_pool = [replay]
    helper.shared_rb = deque()
    helper.train_call_count = 0
    helper.global_train_step = 0
    helper._train_call_count_since_save = 0
    helper._recent_losses = deque(maxlen=10)
    helper._recent_rewards = deque(maxlen=10)
    helper._epoch_port_losses = {}
    helper._epoch_port_rewards = {}
    helper._epoch_port_q_diagnostics = {}
    helper._q_inflation_seen_ports = set()
    helper._tb = None

    # These episode-local values include old synchronization boundaries.  They
    # must not matter; only the third completed optimizer update is due.
    helper.train(current_step=16)
    helper.train(current_step=32)
    assert helper.agent_pool[0].target_updates == 0
    helper.train(current_step=7)
    assert helper.global_train_step == 3
    assert helper.agent_pool[0].target_updates == 1


if __name__ == "__main__":
    test_fresh_target_matches_policy()
    test_acc_exposes_bounded_q_diagnostics()
    test_target_update_uses_global_optimizer_steps()
    print("ACC Q diagnostics tests passed (3/3)")
