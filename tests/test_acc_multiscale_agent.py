import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch


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

    def test_factorized_actions_are_masked_during_selection_and_training(self):
        params = AgentParameters(state_dim=18, hidden_dims=(8, 8))
        agent = ACC("factorized", params, action_space="factorized_interp")
        q_kmin = torch.zeros((2, 17))
        q_kmax = torch.zeros((2, 17))
        q_pmax = torch.zeros((2, 7))
        # Independent argmax would choose the invalid highest-Kmin / lowest-
        # Kmax pair.  The joint mask must select a physically valid pair.
        q_kmin[:, 16] = 10.0
        q_kmax[:, 0] = 10.0
        actions_t = agent._factorized_greedy_actions(
            (q_kmin, q_kmax, q_pmax)
        )
        for row in zip(*(head.tolist() for head in actions_t)):
            physical = agent.select_action([0.0] * 18, epsilon=1.0)[0]
            self.assertLess(
                5.0 + 45.0 * physical.k_min_norm,
                15.0 + 85.0 * physical.k_max_norm,
            )
            from structures import validate_acc_action_indices
            validate_acc_action_indices(row, "factorized_interp")
        states = np.zeros((2, 18), dtype=np.float32)
        actions = np.asarray(
            [tuple(int(head[index]) for head in actions_t) for index in range(2)],
            dtype=np.int64,
        )
        loss = agent.train_model(
            states, actions, np.zeros(2, dtype=np.float32), states
        )
        self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
