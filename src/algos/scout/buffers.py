# src/algos/buffers.py
# MIT License
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class BufferSpec:
    T: int  # primitive horizon per iteration
    A: int  # number of agents
    obs_dim: int
    state_dim: int
    device: torch.device
    hidden_dim: int
    # NEW: dims for comm + grouping
    msg_dim: int  # D_MSG (mailbox width)
    z_msg_dim: int  # descriptor's message vector width (ideally == msg_dim)
    m_groups: int  # number of groups M


class RolloutBuffer:
    """
    Stores a single iteration of rollouts (length T) for A agents.

    Provides a PPO batch:
      obs(T,A,obs_dim), state(T,state_dim), actions(T,A), send(T,A), recv(T,A),
      old_logp_env(T,A), old_logp_send(T,A), old_logp_recv(T,A),
      rewards(T,A), dones(T,A), values(T,A), last_value(A),
      recv_mask(T,A,A), hidden(T,A,H), had_valid_recv(T,A)

    Plus per-step features used by comm-critic and counterfactuals:
      msg_pool_visible(T,A,msg_dim), z_msg(T,A,z_msg_dim),
      grp_P_tau(T,A,M), grp_G(T,A,A)

    And per-block (macro-step) caches for grouping loss:
      block_logp_tau: list of (A,), block_P_tau: list of (A,M),
      block_G: list of (A,A), block_tstarts: list[int]
    """

    def __init__(self, spec: BufferSpec):
        self.spec = spec
        T, A, D_obs, D_state, dev, H = (
            spec.T,
            spec.A,
            spec.obs_dim,
            spec.state_dim,
            spec.device,
            spec.hidden_dim,
        )
        D_MSG, Z_MSG, M = spec.msg_dim, spec.z_msg_dim, spec.m_groups

        # --- per-step tensors (PPO core) ---
        self.obs = torch.empty(T, A, D_obs, device=dev, dtype=torch.float32)
        self.state = torch.empty(T, D_state, device=dev, dtype=torch.float32)
        self.actions = torch.empty(T, A, device=dev, dtype=torch.long)
        self.send = torch.empty(T, A, device=dev, dtype=torch.long)  # {0,1}
        self.recv = torch.empty(T, A, device=dev, dtype=torch.long)  # [0..A-1]
        self.logp_env = torch.empty(T, A, device=dev, dtype=torch.float32)
        self.logp_send = torch.empty(T, A, device=dev, dtype=torch.float32)
        self.logp_recv = torch.empty(T, A, device=dev, dtype=torch.float32)
        self.rewards = torch.empty(T, A, device=dev, dtype=torch.float32)
        self.dones = torch.zeros(T, A, device=dev, dtype=torch.bool)  # per-agent dones
        self.values = torch.empty(T, A, device=dev, dtype=torch.float32)
        self.recv_mask = torch.empty(T, A, A, device=dev, dtype=torch.bool)
        self.hidden = torch.empty(T, A, H, device=dev, dtype=torch.float32)
        self.had_valid_recv = torch.zeros(T, A, device=dev, dtype=torch.bool)

        # --- NEW per-step tensors (comm + grouping features) ---
        self.msg_pool_visible = torch.empty(
            T, A, D_MSG, device=dev, dtype=torch.float32
        )
        self.z_msg = torch.empty(T, A, Z_MSG, device=dev, dtype=torch.float32)
        self.grp_P_tau = torch.empty(T, A, M, device=dev, dtype=torch.float32)
        self.grp_G = torch.empty(T, A, A, device=dev, dtype=torch.float32)

        # --- bootstrap value for the final state ---
        self.last_value = torch.zeros(A, device=dev, dtype=torch.float32)

        # --- per-block (macro) caches for grouping update ---
        self.block_logp_tau: List[torch.Tensor] = []  # each (A,)
        self.block_P_tau: List[torch.Tensor] = []  # each (A,M)
        self.block_G: List[torch.Tensor] = []  # each (A,A)
        self.block_tstarts: List[int] = []

        # --- cursor ---
        self.t_ptr: int = 0

        # default recv mask (disallow self, allow others)
        self._default_recv_mask = ~torch.eye(A, dtype=torch.bool, device=dev)  # (A,A)

    # -------------------- step interface --------------------

    @torch.no_grad()
    def add_step(
        self,
        *,
        obs: torch.Tensor,  # (A, obs_dim)
        state: torch.Tensor,  # (state_dim,)
        actions: torch.Tensor,  # (A,)
        send: torch.Tensor,  # (A,) in {0,1}
        recv: torch.Tensor,  # (A,) in [0..A-1]
        logp_env: torch.Tensor,  # (A,)
        logp_send: torch.Tensor,  # (A,)
        logp_recv: torch.Tensor,  # (A,)
        reward: torch.Tensor,  # (A,)
        done: torch.Tensor,  # (,) or (A,)
        value: torch.Tensor,  # (A,)
        recv_mask: Optional[torch.Tensor] = None,  # (A,A) bool
        hidden: torch.Tensor,  # (A,H)
        had_valid_recv: Optional[torch.Tensor] = None,  # (A,)
        # NEW:
        msg_pool_visible: Optional[torch.Tensor] = None,  # (A, msg_dim)
        z_msg: Optional[torch.Tensor] = None,  # (A, z_msg_dim)
        grp_P_tau: Optional[torch.Tensor] = None,  # (A, M)
        grp_G: Optional[torch.Tensor] = None,  # (A, A)
    ) -> None:
        """Append one primitive step."""
        t = self.t_ptr
        assert t < self.spec.T, f"RolloutBuffer overflow: t={t} >= T={self.spec.T}"
        A = self.spec.A

        # basic shapes
        assert obs.shape == (A, self.spec.obs_dim)
        assert state.shape == (self.spec.state_dim,)
        assert actions.shape == (A,)
        assert send.shape == (A,)
        assert recv.shape == (A,)
        assert logp_env.shape == (A,)
        assert logp_send.shape == (A,)
        assert logp_recv.shape == (A,)
        assert reward.shape == (A,)
        assert value.shape == (A,)
        assert hidden.shape == (A, self.spec.hidden_dim)

        # store core
        self.obs[t].copy_(obs)
        self.state[t].copy_(state)
        self.actions[t].copy_(actions)
        self.send[t].copy_(send)
        self.recv[t].copy_(recv)
        self.logp_env[t].copy_(logp_env)
        self.logp_send[t].copy_(logp_send)
        self.logp_recv[t].copy_(logp_recv)
        self.rewards[t].copy_(reward)
        self.values[t].copy_(value)
        self.hidden[t].copy_(hidden)

        # dones: scalar or per-agent
        if done.ndim == 0:
            self.dones[t].fill_(bool(done.item()))
        else:
            assert done.shape == (A,)
            self.dones[t].copy_(done.to(torch.bool))

        # recv mask exactly as used at act-time
        mask = (
            self._default_recv_mask if recv_mask is None else recv_mask.to(torch.bool)
        )
        assert mask.shape == (A, A)
        self.recv_mask[t].copy_(mask)

        # had_valid_recv
        if had_valid_recv is None:
            had = mask.any(dim=-1)
        else:
            assert had_valid_recv.shape == (A,)
            had = had_valid_recv.to(torch.bool)
        self.had_valid_recv[t].copy_(had)

        # NEW: per-step comm/grouping tensors (all required for counterfactuals)
        if (
            msg_pool_visible is None
            or z_msg is None
            or grp_P_tau is None
            or grp_G is None
        ):
            raise ValueError(
                "add_step now requires msg_pool_visible, z_msg, grp_P_tau, grp_G"
            )
        assert msg_pool_visible.shape == (A, self.spec.msg_dim)
        assert z_msg.shape == (A, self.spec.z_msg_dim)
        assert grp_P_tau.shape == (A, self.spec.m_groups)
        assert grp_G.shape == (A, A)
        self.msg_pool_visible[t].copy_(msg_pool_visible)
        self.z_msg[t].copy_(z_msg)
        self.grp_P_tau[t].copy_(grp_P_tau)
        self.grp_G[t].copy_(grp_G)

        self.t_ptr += 1

    @torch.no_grad()
    def set_last_value(self, last_value: torch.Tensor) -> None:
        """Set bootstrap value for the last state: shape (A,)."""
        assert last_value.shape == (self.spec.A,)
        self.last_value.copy_(last_value)

    # -------------------- per-step group tensors (bulk setter) --------------------

    @torch.no_grad()
    def set_group_tensors(
        self,
        *,
        grp_P_tau: torch.Tensor,  # (T, A, M)
        grp_G: torch.Tensor,  # (T, A, A)
    ) -> None:
        """Optional bulk setter if you prefer to set after the loop."""
        assert grp_P_tau.shape == (self.spec.T, self.spec.A, self.spec.m_groups)
        assert grp_G.shape == (self.spec.T, self.spec.A, self.spec.A)
        self.grp_P_tau.copy_(grp_P_tau)
        self.grp_G.copy_(grp_G)

    # -------------------- macro (grouping) interface --------------------

    # @torch.no_grad()
    def add_macro_group(
        self,
        *,
        logp_grp_tau: torch.Tensor,  # (A,)
        P_tau: torch.Tensor,  # (A, M)
        G: torch.Tensor,  # (A, A)
        t_start: int,
    ) -> None:
        """Record one macro step (block) used later for grouping loss."""
        A, M = self.spec.A, self.spec.m_groups
        assert logp_grp_tau.shape == (A,)
        assert P_tau.shape == (A, M)
        assert G.shape == (A, A)
        self.block_logp_tau.append(logp_grp_tau.clone())
        self.block_P_tau.append(P_tau.clone())
        self.block_G.append(G.clone())
        self.block_tstarts.append(int(t_start))

    # -------------------- finalize & reset --------------------

    @torch.no_grad()
    def finalize(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
        """Return (ppo_batch_dict, aux_macro_dict)."""
        batch: Dict[str, torch.Tensor] = {
            # PPO core
            "obs": self.obs,
            "state": self.state,
            "actions": self.actions,
            "send": self.send,
            "recv": self.recv,
            "old_logp_env": self.logp_env,
            "old_logp_send": self.logp_send,
            "old_logp_recv": self.logp_recv,
            "rewards": self.rewards,
            "dones": self.dones,
            "values": self.values,
            "last_value": self.last_value,
            "recv_mask": self.recv_mask,
            "hidden": self.hidden,
            "had_valid_recv": self.had_valid_recv,
            # NEW per-step features
            "msg_pool_visible": self.msg_pool_visible,
            "z_msg": self.z_msg,
            "grp_P_tau": self.grp_P_tau,
            "grp_G": self.grp_G,
        }

        # macro aux
        if len(self.block_P_tau) == 0:
            aux = {
                "block_logp_tau": [],
                "block_P_tau": [],
                "block_G": [],
                "block_tstarts": [],
            }
        else:
            aux = {
                "block_logp_tau": self.block_logp_tau,
                "block_P_tau": self.block_P_tau,
                "block_G": self.block_G,
                "block_tstarts": self.block_tstarts,
            }
        return batch, aux

    @torch.no_grad()
    def reset(self) -> None:
        """Clear pointers and macro lists to reuse the buffer next iteration."""
        self.t_ptr = 0
        self.block_logp_tau.clear()
        self.block_P_tau.clear()
        self.block_G.clear()
        self.block_tstarts.clear()
        # (tensors are overwritten on next use; no need to zero them)
