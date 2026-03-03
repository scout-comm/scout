from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# --------------------------- Utils ---------------------------
def masked_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return logits
    return logits.masked_fill(~mask, float("-inf"))


def cat_logprob(
    logits: torch.Tensor, actions: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    mlogits = masked_logits(logits, mask)
    dist = Categorical(logits=mlogits)
    logp = dist.log_prob(actions)
    ent = dist.entropy()
    return logp, ent


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
    last_value: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T, A = rewards.shape
    if last_value is None:
        last_value = torch.zeros(A, device=values.device, dtype=values.dtype)
    adv = torch.zeros(T, A, device=values.device, dtype=values.dtype)
    lastgaelam = torch.zeros(A, device=values.device, dtype=values.dtype)
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - (
            dones[t].float()
            if dones.ndim == 2
            else dones[t].float().expand_as(last_value)
        )
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return returns, adv


def ppo_clip_surrogate(
    new_logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor, clip_coef: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * adv
    loss = -torch.mean(torch.min(unclipped, clipped))
    with torch.no_grad():
        approx_kl = torch.mean(old_logp - new_logp)
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_coef).float())
    return loss, approx_kl, clipfrac


# --------------------------- Models ---------------------------
class SharedActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int, n_actions: int, n_agents: int):
        super().__init__()
        self.n_agents = n_agents
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.env_head = nn.Linear(hidden, n_actions)
        self.send_head = nn.Linear(hidden, 2)
        self.recv_head = nn.Linear(hidden, n_agents)

    def forward(self, obs_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(obs_flat)
        return {
            "env_logits": self.env_head(h),
            "send_logits": self.send_head(h),
            "recv_logits": self.recv_head(h),
        }


class RecurrentActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int, n_actions: int, n_agents: int):
        super().__init__()
        self.n_agents = n_agents
        self.hidden = hidden
        self.obs_embed = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.env_head = nn.Linear(hidden, n_actions)
        self.send_head = nn.Linear(hidden, 2)
        self.recv_head = nn.Linear(hidden, n_agents)

    def init_hidden(self, n_agents: int, device: torch.device):
        return torch.zeros(n_agents, self.hidden, device=device)

    def forward(self, obs_flat: torch.Tensor, h_in: torch.Tensor):
        x = self.obs_embed(obs_flat)
        h_out = self.gru(x, h_in)
        return {
            "env_logits": self.env_head(h_out),
            "send_logits": self.send_head(h_out),
            "recv_logits": self.recv_head(h_out),
            "h_out": h_out,
        }


# -------- NEW: Group-aware critic (per-group values -> per-agent by Y) --------
class GroupCritic(nn.Module):
    """
    Predict per-group values v_g(s) (shape [T, M]) and map to per-agent values
    by soft assignment: V_i(s, Y) = (Y @ v)(i).
    Complexity O(M) per step; scalable for large N.
    """

    def __init__(self, state_dim: int, hidden: int, n_groups: int):
        super().__init__()
        self.n_groups = n_groups
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_groups),
        )

    def forward(
        self,
        state: torch.Tensor,  # (T, state_dim)
        P_tau: torch.Tensor,  # (T, N, M) soft assignments (same τ as sampling)
    ) -> torch.Tensor:
        """
        Returns per-agent values: (T, N).
        Compute per-group values v(s_t) -> shape (T, M)
        Then V_i = sum_g P_tau[i,g] * v_g.
        """
        v_groups = self.net(state)  # (T, M)
        # if v_groups.dim() == 3 and v_groups.shape[1] == 1:
        #     v_groups = v_groups.squeeze(1)
        # broadcast v across agents and do soft gather by P_tau
        V = torch.einsum("tnm,tm->tn", P_tau, v_groups)  # (T, N)
        return V


# --------------------------- Config ---------------------------
@dataclass
class PPOCfg:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    entropy_coef_action: float = 0.02
    entropy_coef_send: float = 0.01
    entropy_coef_recv: float = 0.01
    lr: float = 3e-4
    loss_weight_action: float = 1.0
    loss_weight_send: float = 1.0
    loss_weight_recv: float = 1.0
    max_grad_norm: float = 1.0
    update_epochs: int = 8
    minibatch_size: int = 4096
    target_kl: float = 0.15
    max_logp_diff: float = 20.0
    comm_rate_target: Optional[float] = None
    comm_rate_coef: float = 0.005


# --------------------------- Centralized PPO ---------------------------
class CentralizedPPO(nn.Module):
    """
    CTDE PPO with shared per-agent actor and a group-aware critic.
    NOTE: pass soft group affinity G=Y Y^T to act/evaluate to apply the soft log-mask.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        n_agents: int,
        n_groups: int,  # NEW: number of groups M
        hidden: int,
        cfg: PPOCfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_agents = n_agents
        self.n_groups = n_groups
        self.hidden_dim = hidden

        self.policy = RecurrentActor(obs_dim, hidden, n_actions, n_agents)
        self.group_critic = GroupCritic(state_dim, hidden, n_groups)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,  # (A, obs_dim)
        h: torch.Tensor,  # (A, hidden)
        G_soft: Optional[
            torch.Tensor
        ] = None,  # (A, A) soft affinity Y Y^T (row i, col j)
        recv_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ):
        """
        If G_soft is provided, we add log(G_soft+eps) to recv logits (soft log-mask).
        """
        A = obs.shape[0]
        device = obs.device
        if recv_mask is None:
            recv_mask = ~torch.eye(A, dtype=torch.bool, device=device)

        out = self.policy(obs, h)

        # ---- sample env & send
        env_dist = Categorical(logits=out["env_logits"])
        env_a = env_dist.sample()
        logp_env = env_dist.log_prob(env_a)

        send_dist = Categorical(logits=out["send_logits"])
        send_a = send_dist.sample()
        logp_send = send_dist.log_prob(send_a)

        # ---- recipient logits + soft log-mask from groups
        recv_logits = out["recv_logits"]
        if G_soft is not None:
            recv_logits = recv_logits + torch.log(G_soft + eps)
        recv_logits_masked = masked_logits(recv_logits, recv_mask)

        row_has_valid = recv_mask.any(dim=-1)
        recv_a = torch.empty(A, dtype=torch.long, device=device)
        logp_recv = torch.empty(A, dtype=recv_logits_masked.dtype, device=device)

        idx_valid = row_has_valid.nonzero(as_tuple=False).squeeze(-1)
        if idx_valid.numel() > 0:
            dist_valid = Categorical(logits=recv_logits_masked[idx_valid])
            ra_valid = dist_valid.sample()
            recv_a[idx_valid] = ra_valid
            logp_recv[idx_valid] = dist_valid.log_prob(ra_valid)

        idx_invalid = (~row_has_valid).nonzero(as_tuple=False).squeeze(-1)
        if idx_invalid.numel() > 0:
            recv_a[idx_invalid] = idx_invalid
            logp_recv[idx_invalid] = 0.0

        return {
            "env_action": env_a,
            "send_action": send_a,
            "recv_action": recv_a,
            "logp_env": logp_env,
            "logp_send": logp_send,
            "logp_recv": logp_recv,
            "h_out": out["h_out"],
            "send_logits": out["send_logits"],  # for ablation_no_comm
        }

    @torch.no_grad()
    def evaluate(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        G_soft: Optional[torch.Tensor] = None,
        recv_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ):
        A = obs.shape[0]
        device = obs.device
        if recv_mask is None:
            recv_mask = ~torch.eye(A, dtype=torch.bool, device=device)
        out = self.policy(obs, h)
        recv_logits = out["recv_logits"]
        if G_soft is not None:
            recv_logits = recv_logits + torch.log(G_soft + eps)
        logits = {
            "env_logits": out["env_logits"],
            "send_logits": out["send_logits"],
            "recv_logits": masked_logits(recv_logits, recv_mask),
            "h_out": out["h_out"],
        }
        return logits

    def value(self, state: torch.Tensor, P_tau: torch.Tensor) -> torch.Tensor:
        """
        Group-aware critic value: state (T, Sd), P_tau (T, N, M) -> values (T, N)
        """
        return self.group_critic(state, P_tau)

    def update(self, batch: Dict[str, torch.Tensor], ablation_no_comm: bool = False) -> Dict[str, float]:
        """
        Expected batch keys (T = horizon, A = n_agents, M = n_groups):
          obs: (T, A, obs_dim)
          hidden: (T, A, hidden_dim)
          state: (T, state_dim)
          actions: (T, A) int64
          send: (T, A) int64 {0,1}
          recv: (T, A) int64
          old_logp_env/send/recv: (T, A) float
          rewards: (T, A) float
          dones: (T,) or (T, A) bool
          # NEW for group-aware critic and soft log-mask:
          grp_P_tau: (T, A, M)        # cached P_tau at each t (same τ as sampling)
          grp_G: (T, A, A)            # cached G = Y Y^T at each t  (soft)
          # Optional (preferred):
          A_env:  (T, A)   # precomputed GAE advantages
          A_send: (T, A)   # comm counterfactual advantage
          A_recv: (T, A)   # recipient advantage (masked by send==1)
          returns: (T, A)  # target returns for critic
        """
        cfg = self.cfg
        device = next(self.parameters()).device

        # ----------- unpack -----------
        obs = batch["obs"].to(device).detach()
        hidden = batch["hidden"].to(device).detach()
        state = batch["state"].to(device).detach()
        actions = batch["actions"].to(device).detach()
        send = batch["send"].to(device).detach()
        recv = batch["recv"].to(device).detach()
        old_logp_env = batch["old_logp_env"].to(device).detach()
        old_logp_send = batch["old_logp_send"].to(device).detach()
        old_logp_recv = batch["old_logp_recv"].to(device).detach()
        rewards = batch["rewards"].to(device).detach()
        dones = batch["dones"].to(device).detach()

        P_tau_t = batch["grp_P_tau"].to(device).detach()  # (T, A, M)
        G_t = batch["grp_G"].to(device).detach()  # (T, A, A)

        maybe_returns = batch.get("returns", None)
        maybe_A_env = batch.get("A_env", None)
        maybe_A_send = batch.get("A_send", None)
        maybe_A_recv = batch.get("A_recv", None)

        T, A, _ = obs.shape
        assert A == self.n_agents

        # ----------- critic values & advantages -----------
        # group-aware values at rollout time (we expect you stored them; but recompute is fine)
        with torch.no_grad():
            values_old = self.value(state, P_tau_t)  # (T, A)
            last_value = torch.zeros(A, device=device)

            if maybe_returns is None or maybe_A_env is None:
                returns, adv_env = compute_gae(
                    rewards, values_old, dones, cfg.gamma, cfg.gae_lambda, last_value
                )
            else:
                returns = maybe_returns.to(device)
                adv_env = maybe_A_env.to(device)

            # normalize env adv
            adv_env = (adv_env - adv_env.mean()) / (adv_env.std().clamp_min(1e-6))

            # fallback simple proxies for send/recv if not provided
            if maybe_A_send is None:
                # proxy with TD-error normed; better to pass true counterfactuals
                next_values = torch.zeros_like(values_old)
                next_values[:-1] = values_old[1:]
                dones_ = (
                    dones
                    if dones.ndim == 2
                    else dones.unsqueeze(-1).expand_as(values_old)
                )
                nonterm = 1.0 - dones_.float()
                td_error = rewards + cfg.gamma * next_values * nonterm - values_old
                A_send = (td_error - td_error.mean()) / (td_error.std().clamp_min(1e-6))
            else:
                A_send = maybe_A_send.to(device)

            if maybe_A_recv is None:
                A_recv = A_send.clone()  # crude fallback; prefer pairwise in trainer
            else:
                A_recv = maybe_A_recv.to(device)

        # ----------- flatten -----------
        def flat(x):
            return x.reshape(T * A, -1) if x.ndim == 3 else x.reshape(T * A)

        obs_f, h_f = flat(obs), flat(hidden)
        actions_f, send_f, recv_f = flat(actions), flat(send), flat(recv)
        old_env_f, old_send_f, old_recv_f = (
            flat(old_logp_env),
            flat(old_logp_send),
            flat(old_logp_recv),
        )
        adv_env_f, A_send_f, A_recv_f = flat(adv_env), flat(A_send), flat(A_recv)
        returns_f = flat(returns)
        # state_flat = state.reshape(T * A, -1)                     # (T*A, Sd)
        state_flat = state.repeat_interleave(A, dim=0)  # (T*A, Sd)
        P_tau_t = P_tau_t.reshape(T * A, self.n_groups)  # (T*A, M)

        # build soft log-mask per row from G_t
        G_f = G_t.reshape(T * A, A)
        # Ensure each row has at least one valid recipient (allow self as fallback)
        row_has_valid = (G_f > 0).any(dim=-1)
        if not row_has_valid.all():
            idx_bad = (~row_has_valid).nonzero(as_tuple=False).squeeze(-1)
            G_f[idx_bad, idx_bad % A] = 1.0
        recv_mask_f = G_f > 0  # boolean where prob > 0

        # ----------- PPO epochs -----------
        N = T * A
        idx = torch.arange(N, device=device)
        metrics = {
            k: 0.0
            for k in [
                "loss_pi",
                "loss_v",
                "entropy",
                "entropy_env",
                "entropy_send",
                "entropy_recv",
                "approx_kl",
                "approx_kl_env",
                "approx_kl_send",
                "approx_kl_recv",
                "clipfrac",
                "clipfrac_env",
                "clipfrac_send",
                "clipfrac_recv",
                "comm_rate",
                "send_mask_frac",
            ]
        }
        total_mb = 0

        for _ in range(cfg.update_epochs):
            perm = idx[torch.randperm(N, device=device)]
            for s in range(0, N, cfg.minibatch_size):
                mb = perm[s : min(s + cfg.minibatch_size, N)]

                out = self.policy(obs_f[mb], h_f[mb])

                G_f_mb = G_f[mb].detach()
                recv_mask_mb = recv_mask_f[mb].detach()
                state_mb = state_flat[mb].detach()
                P_mb = P_tau_t[mb].detach()
                returns_mb = returns_f[mb].detach()
                old_env_mb = old_env_f[mb].detach()
                old_send_mb = old_send_f[mb].detach()
                old_recv_mb = old_recv_f[mb].detach()
                adv_env_mb = adv_env_f[mb].detach()
                A_send_mb = A_send_f[mb].detach()
                A_recv_mb = A_recv_f[mb].detach()
                actions_mb = actions_f[mb].detach()
                send_mb = send_f[mb].detach()
                recv_mb = recv_f[mb].detach()

                # ---- apply soft log-mask to recv logits using G_f
                recv_logits_masked = masked_logits(
                    out["recv_logits"] + torch.log(G_f_mb + 1e-6), recv_mask_mb
                )

                logp_env, ent_env = cat_logprob(out["env_logits"], actions_mb)
                logp_send, ent_send = cat_logprob(out["send_logits"], send_mb)

                send_mask_b = send_mb.bool()
                send_mask_frac = send_mask_b.float().mean()
                if send_mask_b.any():
                    logp_recv_all, ent_recv_all = cat_logprob(
                        recv_logits_masked, recv_mb
                    )
                    logp_recv = logp_recv_all[send_mask_b]
                    old_recv_sel = old_recv_mb[send_mask_b]
                    adv_recv_sel = A_recv_mb[send_mask_b]
                    if adv_recv_sel.numel() > 1:
                        adv_recv_sel = (adv_recv_sel - adv_recv_sel.mean()) / (
                            adv_recv_sel.std().clamp_min(1e-6)
                        )
                    loss_pi_recv, kl_recv, clipfrac_recv = ppo_clip_surrogate(
                        logp_recv, old_recv_sel, adv_recv_sel, cfg.clip_coef
                    )
                    ent_recv = ent_recv_all[send_mask_b].mean()
                else:
                    loss_pi_recv = kl_recv = clipfrac_recv = ent_recv = (
                        out["env_logits"].sum() * 0.0
                    )  # zero scalar on correct device

                loss_pi_env, kl_env, clipfrac_env = ppo_clip_surrogate(
                    logp_env, old_env_mb, adv_env_mb, cfg.clip_coef
                )
                
                # Skip send policy loss if ablation_no_comm (we force send=0 anyway)
                if ablation_no_comm:
                    zero = out["env_logits"].sum() * 0.0  # zero scalar on correct device
                    loss_pi_send = kl_send = clipfrac_send = zero
                else:
                    loss_pi_send, kl_send, clipfrac_send = ppo_clip_surrogate(
                        logp_send, old_send_mb, A_send_mb, cfg.clip_coef
                    )

                # --- critic on MB-only graph
                v_groups_mb = self.group_critic.net(state_mb)  # needs grad
                v_pred = (P_mb * v_groups_mb).sum(dim=-1)  # P_mb is detached
                loss_v = F.mse_loss(v_pred, returns_mb)

                # Skip comm_reg if ablation_no_comm
                if ablation_no_comm:
                    comm_reg = out["env_logits"].sum() * 0.0
                else:
                    send_probs = F.softmax(out["send_logits"], dim=-1)[:, 1]
                    comm_reg = (
                        cfg.comm_rate_coef
                        * F.mse_loss(
                            send_probs.mean(),
                            torch.tensor(cfg.comm_rate_target, device=device),
                        )
                        if cfg.comm_rate_target is not None and cfg.comm_rate_coef > 0.0
                        else send_probs.mean() * 0.0
                    )

                # Skip send/recv losses if ablation_no_comm
                if ablation_no_comm:
                    loss_pi = cfg.loss_weight_action * loss_pi_env
                    ent = cfg.entropy_coef_action * ent_env.mean()
                else:
                    loss_pi = (
                        cfg.loss_weight_action * loss_pi_env
                        + cfg.loss_weight_send * loss_pi_send
                        + cfg.loss_weight_recv * loss_pi_recv
                    )
                    ent = (
                        cfg.entropy_coef_action * ent_env.mean()
                        + cfg.entropy_coef_send * ent_send.mean()
                        + cfg.entropy_coef_recv * ent_recv
                    )

                loss = loss_pi + cfg.vf_coef * loss_v - ent + comm_reg

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                # ---- metrics
                with torch.no_grad():
                    approx_kl_overall = (kl_env + kl_send + kl_recv) / 3.0
                    clipfrac_overall = (
                        clipfrac_env + clipfrac_send + clipfrac_recv
                    ) / 3.0
                    comm_rate = send_f.float().mean().item()

                    metrics["loss_pi"] += loss_pi.item()
                    metrics["loss_v"] += loss_v.item()
                    # Use same ent calculation as loss
                    metrics["entropy"] += ent.item()
                    metrics["entropy_env"] += ent_env.mean().item()
                    metrics["entropy_send"] += ent_send.mean().item()
                    metrics["entropy_recv"] += (
                        ent_recv.item()
                        if ent_recv.numel() == 1
                        else float(ent_recv.mean().item())
                    )

                    metrics["approx_kl"] += approx_kl_overall.item()
                    metrics["approx_kl_env"] += kl_env.item()
                    metrics["approx_kl_send"] += kl_send.item()
                    metrics["approx_kl_recv"] += kl_recv.item()

                    metrics["clipfrac"] += clipfrac_overall.item()
                    metrics["clipfrac_env"] += clipfrac_env.item()
                    metrics["clipfrac_send"] += clipfrac_send.item()
                    metrics["clipfrac_recv"] += clipfrac_recv.item()

                    metrics["comm_rate"] += comm_rate
                    metrics["send_mask_frac"] += send_mask_frac.item()
                    total_mb += 1

        for k in metrics:
            metrics[k] /= max(total_mb, 1)
        return metrics
