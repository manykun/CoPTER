import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "copter"))

# Keep this focused unit test independent of the server-only logging/scipy
# packages; ACC itself does not use interpolation.
if "loguru" not in sys.modules:
    loguru = types.ModuleType("loguru")
    loguru.logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    sys.modules["loguru"] = loguru
try:
    import scipy.interpolate  # noqa: F401
except ImportError:
    scipy = types.ModuleType("scipy")
    interpolate = types.ModuleType("scipy.interpolate")
    interpolate.RegularGridInterpolator = object
    scipy.interpolate = interpolate
    sys.modules["scipy"] = scipy
    sys.modules["scipy.interpolate"] = interpolate

from agent import ACC
from structures import AgentParameters


class MultiscaleAgentTests(unittest.TestCase):
    def test_select_and_train_two_head_action(self):
        params = AgentParameters(state_dim=18, hidden_dims=(8, 8))
        agent = ACC("test", params, action_space="multiscale")
        physical, action = agent.select_action([0.0] * 18, epsilon=0.0)
        self.assertEqual(len(action), 2)
        self.assertLess(physical.k_min_norm, 1.01)
        states = np.zeros((4, 18), dtype=np.float32)
        actions = np.asarray([action] * 4, dtype=np.int64)
        rewards = np.zeros(4, dtype=np.float32)
        loss = agent.train_model(states, actions, rewards, states)
        self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
