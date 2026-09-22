#!/usr/bin/env python3
"""Regression tests for the ACC local-only replay ablation."""

from collections import deque
from types import SimpleNamespace
import sys

import numpy as np

# Keep this small replay unit test independent of PyTorch/loguru. The test
# exercises AgentHelper.sync() on an uninitialized helper, so model classes and
# logging are intentionally replaced before importing the production module.
class _Logger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class _Agent:
    pass


sys.modules.setdefault("loguru", SimpleNamespace(logger=_Logger()))
sys.modules.setdefault(
    "agent",
    SimpleNamespace(Agent=_Agent, ACC=_Agent, CoPTER=_Agent),
)

from agent_helper import AgentHelper, ReplayBuffer


def transition(value):
    state = np.asarray([value], dtype=np.float32)
    return state, (0, 0, 0), float(value), state + 1


def helper(shared_replay_enabled):
    instance = AgentHelper.__new__(AgentHelper)
    instance.p = SimpleNamespace(
        shared_replay_enabled=shared_replay_enabled,
        sync_up_size=1,
        sync_down_size=1,
    )
    instance.rb_pool = [ReplayBuffer(8), ReplayBuffer(8)]
    instance.agent_pool = [SimpleNamespace(name="p0"), SimpleNamespace(name="p1")]
    instance.shared_rb = deque(maxlen=16)
    for index, replay in enumerate(instance.rb_pool):
        replay.push(*transition(index))
    return instance


def test_local_only_sync_is_a_noop():
    instance = helper(False)
    before = [len(replay) for replay in instance.rb_pool]
    instance.sync()
    assert len(instance.shared_rb) == 0
    assert [len(replay) for replay in instance.rb_pool] == before


def test_enabled_sync_exchanges_experience():
    instance = helper(True)
    instance.sync()
    assert len(instance.shared_rb) == 2
    assert [len(replay) for replay in instance.rb_pool] == [2, 2]


if __name__ == "__main__":
    test_local_only_sync_is_a_noop()
    test_enabled_sync_exchanges_experience()
    print("ACC shared replay tests passed (2/2)")
