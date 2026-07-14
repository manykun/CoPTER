import torch
import torch.nn as nn


class SORTripleHeadACC(nn.Module):
    def __init__(self, state_dim, k_min_dim, k_max_dim, p_max_dim, embedding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
        )
        self.k_min_head = nn.Linear(embedding_dim, k_min_dim)
        self.k_max_head = nn.Linear(embedding_dim, k_max_dim)
        self.p_max_head = nn.Linear(embedding_dim, p_max_dim)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x, return_embedding=False):
        embedding = self.encode(x)
        outputs = (
            self.k_min_head(embedding),
            self.k_max_head(embedding),
            self.p_max_head(embedding),
        )
        if return_embedding:
            return outputs, embedding
        return outputs
