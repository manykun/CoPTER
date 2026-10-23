"""Regression tests for ns3-gym's inconsistent terminal socket reply."""

import sys
import types
from types import SimpleNamespace


# The local development environment does not ship ns3-gym.  NetworkHelper only
# needs the module at import time; these tests inject a deterministic fake env.
ns3gym_package = types.ModuleType("ns3gym")
ns3env_module = types.ModuleType("ns3gym.ns3env")
ns3env_module.Ns3Env = object
ns3gym_package.ns3env = ns3env_module
sys.modules.setdefault("ns3gym", ns3gym_package)
sys.modules.setdefault("ns3gym.ns3env", ns3env_module)

try:
    import loguru  # noqa: F401
except ImportError:
    loguru_module = types.ModuleType("loguru")

    class NullLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    loguru_module.logger = NullLogger()
    sys.modules["loguru"] = loguru_module

from network_helper import NetworkHelper


class FakeEnv:
    def __init__(self, done):
        self.done = done

    def reset(self):
        return [0.0] * 6

    def step(self, _action):
        return None, 0.0, self.done, {}


def helper(done):
    instance = NetworkHelper.__new__(NetworkHelper)
    instance.env = FakeEnv(done)
    instance.n_port = 1
    instance.nhp = SimpleNamespace(port_actions=3, port_states=6)
    instance.action = [0.0, 0.0, 0.0]
    instance.action_port_bitmap = [1]
    instance.port_identifier_map = {}
    return instance


def test_none_done_false_after_progress_is_terminal():
    assert helper(done=False).monitor(51) is True


def test_none_done_false_at_initial_step_is_error():
    try:
        helper(done=False).monitor(0)
    except RuntimeError as exc:
        assert "observation=None" in str(exc)
    else:
        raise AssertionError("missing initial observation must remain an error")


def test_none_done_true_is_terminal():
    assert helper(done=True).monitor(51) is True


if __name__ == "__main__":
    test_none_done_false_after_progress_is_terminal()
    test_none_done_false_at_initial_step_is_error()
    test_none_done_true_is_terminal()
    print("NetworkHelper terminal tests passed (3/3)")
