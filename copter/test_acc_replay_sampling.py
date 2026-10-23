#!/usr/bin/env python3
"""Deterministic checks for ACC congestion-stratified replay."""

import os
import random
import sys
import types

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:
    sys.modules["torch"] = types.ModuleType("torch")
    sys.modules["agent"] = types.SimpleNamespace(
        Agent=object, ACC=object, CoPTER=object
    )

try:
    import loguru  # noqa: F401
except ImportError:
    class SilentLogger:
        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    sys.modules["loguru"] = types.SimpleNamespace(logger=SilentLogger())

sys.path.insert(0, os.path.dirname(__file__))

from agent_helper import ReplayBuffer


def transition(congested, value):
    state = np.zeros(18, dtype=float)
    next_state = np.zeros(18, dtype=float)
    if congested:
        next_state[12] = 0.02
        next_state[14] = 0.001
    return state, (0, 0), float(value), next_state


def test_stratified_sampling_hits_requested_fraction():
    random.seed(7)
    replay = ReplayBuffer(100)
    replay.push_batch(
        [transition(True, index) for index in range(20)]
        + [transition(False, index) for index in range(20, 100)]
    )
    replay.sample(20, "congestion_stratified", 0.5)
    detail = replay.last_sample_diagnostics
    assert detail["congested_available"] == 20
    assert detail["congested_sampled"] == 10
    assert detail["sample_size"] == 20
    assert detail["buffer_size"] == 100


def test_stratified_sampling_falls_back_without_duplicates():
    random.seed(8)
    replay = ReplayBuffer(20)
    replay.push_batch(
        [transition(True, index) for index in range(2)]
        + [transition(False, index) for index in range(2, 20)]
    )
    states, actions, rewards, next_states = replay.sample(
        10, "congestion_stratified", 0.5
    )
    assert len(states) == len(actions) == len(rewards) == len(next_states) == 10
    assert len(set(rewards.tolist())) == 10
    assert replay.last_sample_diagnostics["congested_sampled"] == 2


def test_uniform_sampling_remains_available():
    random.seed(9)
    replay = ReplayBuffer(20)
    replay.push_batch([transition(False, index) for index in range(20)])
    replay.sample(8)
    assert replay.last_sample_diagnostics == {
        "sampling": "uniform",
        "congested_available": 0,
        "congested_sampled": 0,
        "sample_size": 8,
        "buffer_size": 20,
    }


if __name__ == "__main__":
    test_stratified_sampling_hits_requested_fraction()
    test_stratified_sampling_falls_back_without_duplicates()
    test_uniform_sampling_remains_available()
    print("ACC replay sampling tests passed")
