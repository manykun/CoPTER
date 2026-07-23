import torch
import torch.nn as nn


class SORTripleHeadACC(nn.Module):
    def __init__(
        self,
        state_dim,
        k_min_dim,
        k_max_dim,
        p_max_dim,
        hidden_dims=(32, 64, 64, 32),
    ):
        super().__init__()
        hidden_dims = tuple(int(width) for width in hidden_dims)
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths")
        layers = []
        input_dim = state_dim
        for output_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
            input_dim = output_dim
        self.encoder = nn.Sequential(*layers)
        self.k_min_head = nn.Linear(input_dim, k_min_dim)
        self.k_max_head = nn.Linear(input_dim, k_max_dim)
        self.p_max_head = nn.Linear(input_dim, p_max_dim)

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
