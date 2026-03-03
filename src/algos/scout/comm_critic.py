# Value head with message context; used for counterfactual advantages on send/recv.
from __future__ import annotations
import torch
import torch.nn as nn


class CommCritic(nn.Module):
    """V(s, agent_feat) and Q(s, send_feat, recv_feat) for comm advantage."""

    def __init__(self, state_dim: int, agent_feat_dim: int, hidden: int):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(state_dim + agent_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q = nn.Sequential(
            nn.Linear(state_dim + 2 * agent_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def value_withmsg(self, state_tile: torch.Tensor, agent_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state_tile, agent_feat], dim=-1)
        return self.v(x).squeeze(-1)

    def q_comm_pair(
        self, state_tile: torch.Tensor, send_feat: torch.Tensor, recv_feat: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([state_tile, send_feat, recv_feat], dim=-1)
        return self.q(x).squeeze(-1)
