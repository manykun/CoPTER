import os
import random
import copy
from typing import Iterable, Optional

import numpy as np
import torch

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

from backbone_sor import SORMultiHeadACC, SORTripleHeadACC

try:
    from structures import (
        acc_action_dimensions,
        acc_action_from_indices,
        describe_acc_action,
    )
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "copter")))
    from structures import (
        acc_action_dimensions,
        acc_action_from_indices,
        describe_acc_action,
    )


class SORACC:
    def __init__(self, name: str, agent_params, lambda_cons: float = 0.01,
                 lambda_reg: float = 0.001, drift_reg_threshold: float = 0.5,
                 ref_update_interval: int = 256, action_space: str = "legacy"):
        self.p = agent_params
        self.name = name
        self.lambda_cons = lambda_cons
        self.lambda_reg = lambda_reg
        self.drift_reg_threshold = drift_reg_threshold
        self.ref_update_interval = ref_update_interval
        self.action_space = action_space
        self.train_steps = 0
        self.device = torch.device("cpu")
        self.action_dims = acc_action_dimensions(action_space)
        if action_space == "legacy":
            network_factory = lambda: SORTripleHeadACC(
                self.p.state_dim, *self.action_dims,
                hidden_dims=self.p.hidden_dims,
            )
        else:
            network_factory = lambda: SORMultiHeadACC(
                self.p.state_dim, self.action_dims,
                hidden_dims=self.p.hidden_dims,
            )
        self.policy_net = network_factory().to(self.device)
        self.target_net = network_factory().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Do not consume a third random initialization: this keeps each SOR
        # policy initialization byte-identical to ACC for the same seed while
        # still maintaining an independent reference network.
        self.reference_net = copy.deepcopy(self.policy_net).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.p.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, self.name)
        checkpoint = {
            "format_version": 2,
            "action_space": self.action_space,
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "reference": self.reference_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": int(self.train_steps),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None
                ),
            },
        }
        tmp_path = f"{path}.tmp.{os.getpid()}"
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        logger.info(f"SORACC Agent {self.name} - checkpoint saved to {path}")

    def load_model(self, save_path, exp_name_override: Optional[str] = None):
        parts = self.name.split("_SORACC_", 1)
        name = f"{exp_name_override}_SORACC_{parts[1]}" if exp_name_override is not None and len(parts) == 2 else self.name
        checkpoint_path = os.path.join(save_path, name)
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            saved_space = checkpoint.get("action_space", "legacy")
            if saved_space != self.action_space:
                raise ValueError(
                    f"SOR action-space mismatch: checkpoint={saved_space}, "
                    f"requested={self.action_space}"
                )
            self.policy_net.load_state_dict(checkpoint["policy"])
            self.target_net.load_state_dict(
                checkpoint.get("target", checkpoint["policy"])
            )
            self.reference_net.load_state_dict(
                checkpoint.get("reference", checkpoint["policy"])
            )
            if checkpoint.get("optimizer") is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.train_steps = int(checkpoint.get("train_steps", 0))
            rng = checkpoint.get("rng", {})
            if rng.get("python") is not None:
                random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            logger.info(
                f"SORACC Agent {self.name} - checkpoint loaded from {checkpoint_path}"
            )
            return

        # Backward compatibility with pre-v2 SOR checkpoints.
        policy_path = os.path.join(save_path, f"{name}_policy.pt")
        target_path = os.path.join(save_path, f"{name}_target.pt")
        reference_path = os.path.join(save_path, f"{name}_reference.pt")
        if os.path.isfile(policy_path) and os.path.isfile(target_path):
            self.policy_net.load_state_dict(torch.load(policy_path, map_location=self.device))
            self.target_net.load_state_dict(torch.load(target_path, map_location=self.device))
            if os.path.isfile(reference_path):
                self.reference_net.load_state_dict(torch.load(reference_path, map_location=self.device))
            else:
                self.reference_net.load_state_dict(self.policy_net.state_dict())
            logger.info(f"SORACC Agent {self.name} - Model loaded from {save_path}")
        else:
            logger.info(f"SORACC Agent {self.name} - Model files not found in {save_path}, reinitializing models.")
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.reference_net.load_state_dict(self.policy_net.state_dict())

    def encode_state(self, state) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.policy_net.encode(state_tensor)
        return embedding.squeeze(0).cpu().numpy()

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            action = tuple(
                random.randint(0, size - 1) for size in self.action_dims
            )
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_heads = self.policy_net(state_tensor)
            action = tuple(int(head.argmax().item()) for head in q_heads)
        logger.info(
            f"SORACC Agent {self.name} - Action ({self.action_space}): "
            f"{describe_acc_action(action, self.action_space)}"
        )
        return acc_action_from_indices(action, self.action_space), action

    def train_model(self, states, actions, rewards, next_states, cluster_ids=None, prototypes=None, drift_scores=None):
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)

        q_outputs, embeddings = self.policy_net(states_t, return_embedding=True)
        if actions_t.ndim != 2 or actions_t.shape[1] != len(q_outputs):
            raise ValueError(
                f"{self.action_space} replay action width is "
                f"{tuple(actions_t.shape)}; expected (*, {len(q_outputs)})"
            )
        q_prediction = self._gather_q(q_outputs, actions_t)

        with torch.no_grad():
            next_policy_outputs = self.policy_net(next_states_t)
            next_actions = torch.stack([head.argmax(dim=1) for head in next_policy_outputs], dim=1)
            next_target_outputs = self.target_net(next_states_t)
            q_target = self._gather_q(next_target_outputs, next_actions)
            q_estimation = rewards_t.unsqueeze(1) + self.p.gamma * q_target
            td_errors = (q_estimation - q_prediction).detach().squeeze(1)

        loss_td = self.loss_fn(q_prediction, q_estimation)
        loss = loss_td
        loss_cons = torch.tensor(0.0, device=self.device)
        if prototypes is not None and cluster_ids is not None:
            prototype_rows = []
            valid_rows = []
            for row_index, cluster_id in enumerate(cluster_ids):
                proto = prototypes.get(int(cluster_id)) if hasattr(prototypes, "get") else None
                if proto is not None:
                    prototype_rows.append(proto)
                    valid_rows.append(row_index)
            # A missing prototype is an absence of supervision, not a target
            # at the origin.  Pulling embeddings toward zero silently damages
            # the representation after replay synchronization.
            if valid_rows:
                prototype_t = torch.FloatTensor(
                    np.asarray(prototype_rows, dtype=np.float32)
                ).to(self.device)
                row_t = torch.LongTensor(valid_rows).to(self.device)
                loss_cons = torch.mean((embeddings.index_select(0, row_t) - prototype_t) ** 2)
                loss = loss + self.lambda_cons * loss_cons

        loss_reg = torch.tensor(0.0, device=self.device)
        if self.lambda_reg > 0 and drift_scores is not None and cluster_ids is not None:
            high_drift = torch.FloatTensor([float(drift_scores.get(int(cid), 0.0) > self.drift_reg_threshold) for cid in cluster_ids]).to(self.device)
            if float(high_drift.sum().item()) > 0:
                with torch.no_grad():
                    ref_outputs = self.reference_net(states_t)
                reg_terms = [(cur - ref).pow(2).mean(dim=1) for cur, ref in zip(q_outputs, ref_outputs)]
                per_sample_reg = sum(reg_terms) / len(reg_terms)
                loss_reg = (per_sample_reg * high_drift).sum() / high_drift.sum().clamp_min(1.0)
                loss = loss + self.lambda_reg * loss_reg

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.train_steps += 1
        if self.ref_update_interval > 0 and self.train_steps % self.ref_update_interval == 0:
            self.reference_net.load_state_dict(self.policy_net.state_dict())

        def tensor_stats(value):
            flat = value.detach().float().reshape(-1).cpu()
            absolute = flat.abs()
            return {
                "mean": float(flat.mean().item()),
                "abs_mean": float(absolute.mean().item()),
                "abs_p95": float(torch.quantile(absolute, 0.95).item()),
                "abs_max": float(absolute.max().item()),
            }

        expected_q_bound = (
            1.0 / (1.0 - self.p.gamma)
            if 0.0 <= self.p.gamma < 1.0 else float("inf")
        )
        return {
            "loss": float(loss.item()),
            "loss_td": float(loss_td.item()),
            "loss_cons": float(loss_cons.item()),
            "loss_reg": float(loss_reg.item()),
            "td_errors": td_errors.cpu().numpy(),
            "q_prediction": tensor_stats(q_prediction),
            "q_target": tensor_stats(q_estimation),
            "td_error": tensor_stats(q_prediction - q_estimation),
            "reward_mean": float(rewards_t.mean().item()),
            "reward_abs_max": float(rewards_t.abs().max().item()),
            "expected_q_bound": float(expected_q_bound),
        }

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    @staticmethod
    def _gather_q(outputs, actions):
        return sum(
            head.gather(1, actions[:, index].unsqueeze(1))
            for index, head in enumerate(outputs)
        )
