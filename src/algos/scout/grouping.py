# src/algos/grouping.py
# MIT License
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------- utils --------------------------
def column_balance_loss(P_tau: torch.Tensor) -> torch.Tensor:
    """
    P_tau: [N, M] row-stochastic soft assignments (same τ used as sampling).
    Penalize deviation of column mass from uniform N/M.
    """
    N, M = P_tau.shape
    col = P_tau.sum(0) / max(N, 1)  # [M]
    target = P_tau.new_full((M,), 1.0 / M)
    return F.mse_loss(col, target, reduction="mean")


def row_entropy(P_tau: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = P_tau.clamp_min(eps)
    return (-p * p.log()).sum(dim=-1).mean()


# -------------------------- modules --------------------------
@dataclass
class GroupingConfig:
    d_in: int  # descriptor's grouping projection dim (z_grp)
    m_groups: int = 4
    gumbel_tau: float = 1.0
    logit_scale: float = 1.0
    use_prototypes: bool = True
    dropout: float = 0.0


class GroupingPolicy(nn.Module):
    """
    Grouping module that operates on the descriptor's grouping projection z_grp.
    Provides:
      - logits(z_grp): [N, M]
      - sample(z_grp, tau): returns (y_soft, logp_grp_tau, P_tau, G)
      - loss(...): implements group-critic PG + edge-utility + balance - entropy

    You should call .sample(...) once at each K-step block start, cache outputs,
    and later call .loss(...) using those cached tensors (no re-forward).
    """

    def __init__(self, cfg: GroupingConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.use_prototypes:
            # cosine-similarity against learnable prototypes
            self.prototypes = nn.Parameter(
                torch.randn(cfg.m_groups, cfg.d_in) * (1.0 / (cfg.d_in**0.5))
            )
            self.classifier = None
        else:
            self.prototypes = None
            self.classifier = nn.Sequential(
                nn.Linear(cfg.d_in, max(64, cfg.d_in)),
                nn.ReLU(),
                nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
                nn.Linear(max(64, cfg.d_in), cfg.m_groups),
            )

    @property
    def m(self) -> int:
        return self.cfg.m_groups

    # ---- logits over groups from z_grp ----
    def logits(self, z_grp: torch.Tensor) -> torch.Tensor:
        """
        z_grp: [N, D]
        returns raw logits [N, M] (pre-temperature, pre-scale)
        """
        if self.prototypes is not None:
            z_n = F.normalize(z_grp, dim=-1)
            p_n = F.normalize(self.prototypes, dim=-1)  # [M, D]
            # cosine similarities as logits
            return (z_n @ p_n.t()) * self.cfg.logit_scale
        else:
            return self.classifier(z_grp) * self.cfg.logit_scale

    # ---- one sampling call per block start; cache all tensors you need ----
    # @torch.no_grad()
    def sample(
        self,
        z_grp: torch.Tensor,
        *,
        tau: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict to cache in the buffer at block start:
          {
            'y_soft': [N,M],
            'logp_grp_tau': [N],           # same-τ log-prob of sampled y
            'P_tau': [N,M],                # softmax(logits/τ)
            'G': [N,N],                    # Y Y^T soft affinity
          }
        """
        τ = self.cfg.gumbel_tau if tau is None else tau
        L = self.logits(z_grp)  # [N, M]
        logp_tau = F.log_softmax(L / τ, dim=-1)  # [N, M]
        y_soft = F.gumbel_softmax(L, tau=τ, hard=False, dim=-1)  # [N, M]
        # same-τ log-prob of the sampled (soft) one-hot (ST trick in loss)
        logp_grp_tau = (y_soft * logp_tau).sum(-1)  # [N]
        P_tau = (L / τ).softmax(dim=-1)  # [N, M]
        G = y_soft @ y_soft.t()  # [N, N]
        return {"y_soft": y_soft, "logp_grp_tau": logp_grp_tau, "P_tau": P_tau, "G": G}

    # ---- loss that matches our math (no re-forward; no argmax labels) ----
    def loss(
        self,
        *,
        # cached from .sample() at block start (no recompute):
        logp_grp_tau: torch.Tensor,  # [B] or [N] flattened across agents in block
        P_tau: torch.Tensor,  # [B, M] or [N, M]
        G: torch.Tensor,  # [Bn, Bn] or [N, N] (pairwise affinity)
        # advantages / utilities:
        A_grp: torch.Tensor,  # [B] group-critic advantage per block (broadcast ok)
        U_pair: Optional[
            torch.Tensor
        ] = None,  # [Bn, Bn] pairwise recipient utility (A_recv) (diag ignored)
        # weights:
        lambda_edge: float = 0.0,
        lambda_bal: float = 0.1,
        lambda_ent: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        L_grp = L_grpPG + L_edge + λ_bal * balance − λ_ent * row_entropy

        Shapes:
          - If you train per-block, flatten all agents in the block into B (same ordering for P_tau rows and logp entries).
          - G and U_pair should be square over those B agents; zero the diagonal beforehand if needed.
        """
        # (A) group-critic policy surrogate (use cached same-τ log-prob)
        # detach A_grp or keep it if you want critic grads to flow; typically detach
        L_grpPG = -(A_grp.detach() * logp_grp_tau).mean()

        # (B) edge-utility alignment term
        if (U_pair is not None) and (lambda_edge > 0.0):
            # mask diagonal (self-edges) if not already
            Bn = G.shape[0]
            eye = torch.eye(Bn, dtype=torch.bool, device=G.device)
            U = U_pair.clone().masked_fill(eye, 0.0)
            Gm = G.clone().masked_fill(eye, 0.0)

            # ---- row-standardize U (it is already row-centered upstream; std, not sum)
            u_std = U.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
            U_hat = (U / u_std).clamp_(-5.0, 5.0)  # clamp avoids rare spikes

            # ---- center & L2-norm G to comparable scale
            Gc = Gm - Gm.mean(dim=1, keepdim=True)
            g_std = Gc.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
            G_hat = Gc / g_std

            # ---- cosine-like alignment rowwise
            edge_util = (U_hat * G_hat).sum(dim=1)  # [Bn]
            L_edge = -lambda_edge * edge_util.mean()
        else:
            L_edge = torch.zeros((), device=P_tau.device)

        # balance & entropy on P_tau (computed with same τ as sampling)
        L_bal = lambda_bal * column_balance_loss(P_tau)
        L_ent = -lambda_ent * row_entropy(P_tau)

        L_total = L_grpPG + L_edge + L_bal + L_ent

        logs = {
            "grp/pg": float(L_grpPG.detach().item()),
            "grp/edge": float(L_edge.detach().item()),
            "grp/balance": float(L_bal.detach().item()),
            "grp/row_entropy": float((-L_ent).detach().item()),
            "grp/total": float(L_total.detach().item()),
        }
        return L_total, logs
