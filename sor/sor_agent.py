import os
import random
from typing import Iterable, Optional

import numpy as np
import torch

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

from backbone_sor import SORTripleHeadACC

try:
    from structures import (
        ACC_KMAX_VALUES,
        ACC_KMIN_VALUES,
        ACC_PMAX_VALUES,
        acc_action_from_indices,
    )
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "copter")))
    from structures import (
        ACC_KMAX_VALUES,
        ACC_KMIN_VALUES,
        ACC_PMAX_VALUES,
        acc_action_from_indices,
    )


class SORACC:
    def __init__(self, name: str, agent_params, lambda_cons: float = 0.01, lambda_reg: float = 0.001, drift_reg_threshold: float = 0.5, ref_update_interval: int = 256):
        self.p = agent_params
        self.name = name
        self.lambda_cons = lambda_cons
        self.lambda_reg = lambda_reg
        self.drift_reg_threshold = drift_reg_threshold
        self.ref_update_interval = ref_update_interval
        self.train_steps = 0
        self.device = torch.device("cpu")
        self.policy_net = SORTripleHeadACC(self.p.state_dim, self.p.kmin_dim, self.p.kmax_dim, self.p.pmax_dim).to(self.device)
        self.target_net = SORTripleHeadACC(self.p.state_dim, self.p.kmin_dim, self.p.kmax_dim, self.p.pmax_dim).to(self.device)
        self.reference_net = SORTripleHeadACC(self.p.state_dim, self.p.kmin_dim, self.p.kmax_dim, self.p.pmax_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.reference_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.p.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()

    @staticmethod
    def action_values():
        return ACC_KMIN_VALUES, ACC_KMAX_VALUES, ACC_PMAX_VALUES

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(save_path, f"{self.name}_policy.pt"))
        torch.save(self.target_net.state_dict(), os.path.join(save_path, f"{self.name}_target.pt"))
        torch.save(self.reference_net.state_dict(), os.path.join(save_path, f"{self.name}_reference.pt"))
        logger.info(f"SORACC Agent {self.name} - Model saved to {save_path}")

    def load_model(self, save_path, exp_name_override: Optional[str] = None):
        parts = self.name.split("_SORACC_", 1)
        name = f"{exp_name_override}_SORACC_{parts[1]}" if exp_name_override is not None and len(parts) == 2 else self.name
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
            action = (
                random.randint(0, self.p.kmin_dim - 1),
                random.randint(0, self.p.kmax_dim - 1),
                random.randint(0, self.p.pmax_dim - 1),
            )
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_kmin, q_kmax, q_pmax = self.policy_net(state_tensor)
            action = (int(q_kmin.argmax().item()), int(q_kmax.argmax().item()), int(q_pmax.argmax().item()))
        return acc_action_from_indices(action), action

    def train_model(self, states, actions, rewards, next_states, cluster_ids=None, prototypes=None, drift_scores=None):
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)

        q_outputs, embeddings = self.policy_net(states_t, return_embedding=True)
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
            for cluster_id in cluster_ids:
                proto = prototypes.get(int(cluster_id)) if hasattr(prototypes, "get") else None
                if proto is None:
                    proto = np.zeros(embeddings.shape[1], dtype=np.float32)
                prototype_rows.append(proto)
            prototype_t = torch.FloatTensor(np.asarray(prototype_rows, dtype=np.float32)).to(self.device)
            loss_cons = torch.mean((embeddings - prototype_t) ** 2)
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

        return {
            "loss": float(loss.item()),
            "loss_td": float(loss_td.item()),
            "loss_cons": float(loss_cons.item()),
            "loss_reg": float(loss_reg.item()),
            "td_errors": td_errors.cpu().numpy(),
        }

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    @staticmethod
    def _gather_q(outputs, actions):
        q_kmin, q_kmax, q_pmax = outputs
        return (
            q_kmin.gather(1, actions[:, 0].unsqueeze(1))
            + q_kmax.gather(1, actions[:, 1].unsqueeze(1))
            + q_pmax.gather(1, actions[:, 2].unsqueeze(1))
        )
