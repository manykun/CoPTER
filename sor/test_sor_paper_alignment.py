#!/usr/bin/env python3
"""Fast invariants required by the SOR paper protocol (no ns-3 needed)."""

import hashlib
import logging
import random
import types
import tempfile
from pathlib import Path

import numpy as np
import torch

COPTER = Path(__file__).resolve().parents[1] / "copter"
import sys

sys.path.insert(0, str(COPTER))
try:
    import loguru  # noqa: F401
except ImportError:
    class BraceLogger:
        def _write(self, level, message, *args):
            logging.getLogger("sor-paper-test").log(
                level, message.format(*args) if args else message
            )
        def info(self, message, *args): self._write(logging.INFO, message, *args)
        def warning(self, message, *args): self._write(logging.WARNING, message, *args)
        def error(self, message, *args): self._write(logging.ERROR, message, *args)
    fallback = BraceLogger()
    sys.modules["loguru"] = types.SimpleNamespace(logger=fallback)
try:
    import scipy  # noqa: F401
except ImportError:
    scipy = types.ModuleType("scipy")
    interpolate = types.ModuleType("scipy.interpolate")
    interpolate.RegularGridInterpolator = object
    scipy.interpolate = interpolate
    sys.modules["scipy"] = scipy
    sys.modules["scipy.interpolate"] = interpolate
from sor_agent_helper import SORAgentHelper
from agent import ACC
from structures import AgentHelperParameters, AgentParameters, acc_action_dimensions


def digest_parameters(module):
    digest = hashlib.sha256()
    for parameter in module.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def transition(helper, port, index):
    state = np.full(18, index / 10.0, dtype=np.float32)
    next_state = state + 0.01
    action = (index % 9, index % 7)
    embedding = helper.agent_pool[port].encode_state(state)
    local = helper.rb_pool[port].push(
        state, action, 0.2 + index / 100.0, next_state, embedding,
        stream_id=port,
    )
    helper.global_memory.push_existing(local)


def main():
    seed = 17
    params = AgentParameters(
        state_dim=18,
        kmin_dim=9,
        kmax_dim=7,
        pmax_dim=1,
        hidden_dims=(32, 64, 64, 32),
    )
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    acc = ACC("alignment_ACC_0", params, action_space="multiscale")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    with tempfile.TemporaryDirectory() as tmp:
        ahp = AgentHelperParameters(
            train_set_size=2,
            rb_size=5,
            rb_size_global=9,
            target_update_interval=2,
            epsilon_schedule="global",
            state_save_interval=1,
        )
        sor = SORAgentHelper(
            node_number=2,
            ahp=ahp,
            model_dir=tmp,
            exp_name="alignment",
            acc_action_space="multiscale",
            save_buffer_every=1,
        )
        assert acc_action_dimensions("multiscale") == (9, 7)
        assert digest_parameters(acc.policy_net) == digest_parameters(
            sor.agent_pool[0].policy_net
        ), "ACC/SOR initial policy tensors differ"

        for index in range(8):
            for port in range(2):
                transition(sor, port, index)
        assert all(len(replay) <= 5 for replay in sor.rb_pool)
        assert len(sor.global_memory) <= 9

        initial_target = digest_parameters(sor.agent_pool[0].target_net)
        sor.train(8)
        assert sor.global_train_step == 1
        assert digest_parameters(sor.agent_pool[0].target_net) == initial_target
        sor.train(16)
        assert sor.global_train_step == 2
        assert digest_parameters(sor.agent_pool[0].target_net) == digest_parameters(
            sor.agent_pool[0].policy_net
        )
        assert sor._last_target_update_step == 2

        sor.epoch = 1
        sizes = [len(replay) for replay in sor.rb_pool]
        global_size = len(sor.global_memory)
        prototypes = [len(replay.prototype_manager.prototypes) for replay in sor.rb_pool]
        policy_hash = digest_parameters(sor.agent_pool[0].policy_net)
        target_hash = digest_parameters(sor.agent_pool[0].target_net)
        reference_hash = digest_parameters(sor.agent_pool[0].reference_net)
        optimizer_entries = len(sor.agent_pool[0].optimizer.state)
        sor.save()
        expected_rng = (random.random(), float(np.random.random()), float(torch.rand(1)))

        restored = SORAgentHelper(
            node_number=2,
            ahp=ahp,
            model_dir=tmp,
            exp_name="alignment",
            acc_action_space="multiscale",
            save_buffer_every=1,
        )
        restored.load()
        assert restored.global_train_step == 2
        assert restored._last_target_update_step == 2
        assert [len(replay) for replay in restored.rb_pool] == sizes
        assert len(restored.global_memory) == global_size
        assert [len(replay.prototype_manager.prototypes) for replay in restored.rb_pool] == prototypes
        assert digest_parameters(restored.agent_pool[0].policy_net) == policy_hash
        assert digest_parameters(restored.agent_pool[0].target_net) == target_hash
        assert digest_parameters(restored.agent_pool[0].reference_net) == reference_hash
        assert len(restored.agent_pool[0].optimizer.state) == optimizer_entries
        actual_rng = (random.random(), float(np.random.random()), float(torch.rand(1)))
        assert actual_rng == expected_rng
        assert digest_parameters(restored.agent_pool[0].policy_net) == digest_parameters(
            sor.agent_pool[0].policy_net
        )
        assert digest_parameters(restored.agent_pool[0].target_net) == digest_parameters(
            sor.agent_pool[0].target_net
        )
        assert digest_parameters(restored.agent_pool[0].reference_net) == digest_parameters(
            sor.agent_pool[0].reference_net
        )

    print("SOR paper alignment tests passed")


if __name__ == "__main__":
    main()
