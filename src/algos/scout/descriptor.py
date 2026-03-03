# src/algos/descriptor.py
# MIT License
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- utils --------------------
class RunningNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("count", torch.tensor(0.0))
        self.register_buffer("mean", torch.tensor([]))
        self.register_buffer("var", torch.tensor([]))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.detach().float()
        if x.numel() == 0:
            return
        b = x.shape[0]
        m = x.mean(0)
        v = x.var(0, unbiased=False)
        if self.mean.numel() == 0:
            self.mean = m
            self.var = v
            self.count = torch.tensor(float(b), device=x.device)
            return
        delta = m - self.mean
        tot = self.count + b
        new_mean = self.mean + delta * (b / tot)
        m_a = self.var * self.count
        m_b = v * b
        M2 = m_a + m_b + delta.pow(2) * (self.count * b / tot)
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean.numel() == 0:
            return x
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


# -------------------- config --------------------
@dataclass
class DescriptorConfig:
    # Core continuous per-agent observation vector
    obs_dim: int
    obs_proj_dim: int = 64

    # Optional features (all are generic, env-agnostic)
    include_hidden: bool = True
    hidden_dim: int = 64
    hidden_proj_dim: int = 32

    include_time_frac: bool = True  # scalar in [0,1]
    include_budget_frac: bool = True  # scalar in [0,1] per agent
    include_progress: bool = False  # scalar in [0,1] (optional)

    # Optional pooled message context (already-received msgs this macro step)
    include_msg_pool: bool = True
    msg_dim: int = 32
    msg_pool: str = "mean"  # mean | max

    # Normalization
    normalize_obs: bool = True
    normalize_hidden: bool = False
    normalize_msgs: bool = False

    # New: projection heads for grouping and messaging
    grp_proj_dim: int = 64
    msg_proj_dim: int = 64
    # Orthogonality penalty weight (0 to disable)
    ortho_coef: float = 0.0

    # InfoNCE temperature (used by descriptor.info_nce)
    nce_temperature: float = 0.1


# -------------------- encoders --------------------
class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        hid = max(64, d_out)
        self.net = nn.Sequential(
            nn.Linear(d_in, hid),
            nn.ReLU(),
            nn.Linear(hid, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x):
        return self.net(x)


class DescriptorBuilder(nn.Module):
    """
    Environment-agnostic plan descriptor builder.

    Expects a dict of *generic* per-agent features (all optional except 'obs'):
      features = {
        'obs':          [N, obs_dim]              (required)
        'hidden':       [N, hidden_dim]           (optional)
        'time_frac':    [N, 1]                    (optional, in [0,1])
        'budget_frac':  [N, 1]                    (optional, in [0,1])
        'progress':     [N, 1]                    (optional, in [0,1])
        'msgs':         [N, M, msg_dim]           (optional pooled context)
        'msgs_mask':    [N, M] boolean/binary     (optional; 1=valid)
      }

    Outputs:
      xi: [N, d_xi]  (concatenation of projected components)

    New helpers:
      project_group(xi) -> z_grp : [N, Dg]
      project_msg(xi)    -> z_msg : [N, Dm]
      heads(features, ...) -> dict with xi, z_grp, z_msg
      info_nce(z_send, z_recv, negatives=None, temperature=None) -> loss
      ortho_penalty(z_grp, z_msg) -> scalar penalty
    """

    def __init__(self, cfg: DescriptorConfig):
        super().__init__()
        self.cfg = cfg

        # Normalizers (continuous)
        self.obs_norm = RunningNorm() if cfg.normalize_obs else None
        self.hid_norm = (
            RunningNorm() if (cfg.normalize_hidden and cfg.include_hidden) else None
        )
        self.msg_norm = (
            RunningNorm() if (cfg.normalize_msgs and cfg.include_msg_pool) else None
        )

        # Encoders
        self.obs_enc = MLP(cfg.obs_dim, cfg.obs_proj_dim)
        self.hid_enc = (
            MLP(cfg.hidden_dim, cfg.hidden_proj_dim) if cfg.include_hidden else None
        )
        self.msg_enc = (
            MLP(cfg.msg_dim, cfg.msg_dim) if cfg.include_msg_pool else None
        )  # keep msg_dim

        # Projection heads (shared trunk -> two projections)
        # - z_grp is used to compute grouping similarities/logits
        # - z_msg is used as the message content vector
        self.grp_proj = MLP(self.d_xi, cfg.grp_proj_dim)
        self.msg_proj = MLP(self.d_xi, cfg.msg_proj_dim)

    def _pool_msgs(
        self, msgs: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        msgs: [N, M, D]; mask: [N, M] with 1 for valid, 0 for pad.
        Returns pooled [N, D] using mean or max over valid msgs.
        """
        N, M, D = msgs.shape
        if mask is None:
            if self.cfg.msg_pool == "max":
                return msgs.max(dim=1).values
            return msgs.mean(dim=1)
        mask = mask.float().unsqueeze(-1)  # [N,M,1]
        msgs = msgs * mask
        if self.cfg.msg_pool == "max":
            very_neg = torch.finfo(msgs.dtype).min
            msgs_masked = msgs + (1.0 - mask) * very_neg
            return msgs_masked.max(dim=1).values
        denom = mask.sum(dim=1).clamp_min(1.0)  # avoid div by zero
        return msgs.sum(dim=1) / denom  # mean over valid

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        *,
        device: Optional[torch.device] = None,
        update_norm: bool = True,
    ) -> torch.Tensor:
        """
        Back-compat: returns only xi (the shared descriptor).
        Use .heads(...) if you also need z_grp and z_msg.
        """
        assert "obs" in features, "DescriptorBuilder requires features['obs']"
        x_obs = features["obs"].to(device) if device else features["obs"]

        parts = []

        # Normalize + encode obs
        if self.obs_norm is not None and self.training and update_norm:
            self.obs_norm.update(x_obs)
        x_obs_n = self.obs_norm.normalize(x_obs) if self.obs_norm is not None else x_obs
        parts.append(self.obs_enc(x_obs_n))

        # Optional: hidden
        if self.cfg.include_hidden and ("hidden" in features):
            h = features["hidden"].to(x_obs.device)
            if self.hid_norm is not None and self.training and update_norm:
                self.hid_norm.update(h)
            h_n = self.hid_norm.normalize(h) if self.hid_norm is not None else h
            parts.append(self.hid_enc(h_n))

        # Optional: time/budget/progress (already small scalars; no encoder)
        for key in ("time_frac", "budget_frac", "progress"):
            if getattr(self.cfg, f"include_{key}", False) and (key in features):
                parts.append(features[key].to(x_obs.device))

        # Optional: pooled message context
        if self.cfg.include_msg_pool and ("msgs" in features):
            msgs = features["msgs"].to(x_obs.device)  # [N,M,Dm]
            mask = features.get("msgs_mask", None)
            if mask is not None:
                mask = mask.to(x_obs.device)
            if self.msg_norm is not None and self.training and update_norm:
                # Flatten valid msgs for running stats
                if mask is not None:
                    flat = msgs[mask.bool()] if msgs.numel() and mask.any() else None
                else:
                    flat = msgs.reshape(-1, msgs.shape[-1])
                if flat is not None and flat.numel():
                    self.msg_norm.update(flat)
            if self.msg_norm is not None:
                msgs = (msgs - self.msg_norm.mean.to(msgs.device)) / torch.sqrt(
                    self.msg_norm.var.to(msgs.device) + self.msg_norm.eps
                )
            pooled = self._pool_msgs(msgs, mask)  # [N,Dm]
            parts.append(self.msg_enc(pooled))

        xi = torch.cat(parts, dim=-1)
        return xi

    # -------- New public helpers --------
    def heads(
        self,
        features: Dict[str, torch.Tensor],
        *,
        device: Optional[torch.device] = None,
        update_norm: bool = True,
        detach_grp: bool = False,
        detach_msg: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience: build xi then project to z_grp and z_msg.
        detach_* lets you stop gradients selectively if needed.
        """
        xi = self.forward(features, device=device, update_norm=update_norm)
        z_in_grp = xi.detach() if detach_grp else xi
        z_in_msg = xi.detach() if detach_msg else xi
        z_grp = self.grp_proj(z_in_grp)
        z_msg = self.msg_proj(z_in_msg)
        return {"xi": xi, "z_grp": z_grp, "z_msg": z_msg}

    def project_group(self, xi: torch.Tensor) -> torch.Tensor:
        return self.grp_proj(xi)

    def project_msg(self, xi: torch.Tensor) -> torch.Tensor:
        return self.msg_proj(xi)

    def info_nce(
        self,
        z_send: torch.Tensor,  # [B, Dm]
        z_recv: torch.Tensor,  # [B, Dm] (positives, aligned with z_send)
        negatives: Optional[
            torch.Tensor
        ] = None,  # [B, K, Dm] or None -> use batch as negatives
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Communication-aware contrastive loss:
          - positives: (sender i, chosen recipient j) when send==1
          - negatives: other agents' z_msg; if None, uses in-batch negatives
        """
        T = temperature or self.cfg.nce_temperature
        z_s = F.normalize(z_send, dim=-1)
        z_r = F.normalize(z_recv, dim=-1)

        if negatives is None:
            # in-batch: use other recipients as negatives
            logits = (z_s @ z_r.t()) / T  # [B, B]
            labels = torch.arange(z_s.size(0), device=z_s.device)
            loss = F.cross_entropy(logits, labels)
            return loss

        # explicit negatives
        B, K, D = negatives.shape
        z_neg = F.normalize(negatives, dim=-1)  # [B,K,D]
        pos = (z_s * z_r).sum(-1, keepdim=True) / T  # [B,1]
        neg = (z_s.unsqueeze(1) * z_neg).sum(-1) / T  # [B,K]
        logits = torch.cat([pos, neg], dim=1)  # [B,1+K]
        labels = torch.zeros(B, dtype=torch.long, device=z_s.device)  # 0 = positive
        loss = F.cross_entropy(logits, labels)
        return loss

    def ortho_penalty(self, z_grp: torch.Tensor, z_msg: torch.Tensor) -> torch.Tensor:
        """
        Encourage the two projections to carry complementary info.
        Uses cosine similarity squared averaged over batch.
        """
        if self.cfg.ortho_coef <= 0.0:
            return torch.zeros((), device=z_grp.device)
        zg = F.normalize(z_grp, dim=-1)
        zm = F.normalize(z_msg, dim=-1)
        cos2 = (zg * zm).sum(-1).pow(2).mean()
        return self.cfg.ortho_coef * cos2

    @property
    def d_xi(self) -> int:
        d = self.cfg.obs_proj_dim
        if self.cfg.include_hidden:
            d += self.cfg.hidden_proj_dim
        if self.cfg.include_time_frac:
            d += 1
        if self.cfg.include_budget_frac:
            d += 1
        if self.cfg.include_progress:
            d += 1
        if self.cfg.include_msg_pool:
            d += self.cfg.msg_dim
        return d
