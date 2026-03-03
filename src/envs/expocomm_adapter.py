# MIT License
# src/envs/expocomm_adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable

import numpy as np


@dataclass
class ExpoCommAdapterCfg:
    """
    Configuration for the ExpoCommAdapter.

    comm_radius: Optional float. If set, recipients farther than this Euclidean
                 distance (in env "position" units) will be masked out.
    use_alive_heuristic: If True, infer an "alive" mask per step by checking whether
                 an agent's observation vector is all zeros; mask dead agents out of recv.
                 Leave False to allow comms regardless of this heuristic.
    """

    comm_radius: Optional[float] = None
    use_alive_heuristic: bool = False


class ExpoCommAdapter:
    """
    Adapter for the ExpoComm MultiAgentEnv wrappers you shared:
      - _AdvPursuitWrapper / AdvPursuit_w_PretrainedOpp
      - _BattleWrapper / Battle_w_PretrainedOpp

    Produces fixed-shape numpy arrays compatible with your trainer/PPO:
      - obs:        (A, obs_dim) float32
      - state:      (state_dim,) float32
      - rewards:    (A,) float32
      - dones:      (A,) bool  (scalar episode done broadcast to all agents)
      - recv_mask:  (A, A) bool (baseline: no-self; optional distance/alive gating)
    """

    def __init__(self, env: Any, cfg: ExpoCommAdapterCfg = ExpoCommAdapterCfg()):
        """
        env: an instance of one of the ExpoComm wrappers (already constructed)
             e.g., _BattleWrapper(**env_config) or Battle_w_PretrainedOpp(**env_config)
        """
        self.env = env
        self.cfg = cfg

        # Discover shapes from env_info (wrappers implement get_env_info()).
        info = self.env.get_env_info()
        self.n_agents: int = int(info["n_agents"])
        self.obs_dim: int = int(info["obs_shape"])
        self.state_dim: int = int(info["state_shape"])

        # Internal caches
        self._last_obs: Optional[np.ndarray] = None
        self._last_positions: Optional[np.ndarray] = None  # (A,2)

    # -------------------- public API --------------------

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Returns:
          obs:       (A, obs_dim) float32
          state:     (state_dim,) float32
          recv_mask: (A, A) bool  (no-self ∧ optional alive ∧ optional distance)
          info:      dict with 'alive_mask' and optional extras
        """
        # The wrappers don't take seed in reset; it's passed to inner env at construction.
        obs, state = self.env.reset()
        assert (
            obs.shape[0] == self.n_agents
        ), "Adapter got obs for wrong #agents (should be RED only)."
        obs = np.asarray(obs, dtype=np.float32).reshape(self.n_agents, -1)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        # assert obs.shape[1] == self.obs_dim, f"obs_dim mismatch: {obs.shape[1]} vs {self.obs_dim}"
        # assert state.shape[0] == self.state_dim, f"state_dim mismatch: {state.shape[0]} vs {self.state_dim}"

        # Try to get positions (optional).
        positions = self._safe_get_positions()
        self._last_obs = obs
        self._last_positions = positions

        alive_mask = (
            self._infer_alive(obs)
            if self.cfg.use_alive_heuristic
            else np.ones((self.n_agents,), dtype=bool)
        )
        recv_mask = self._build_recv_mask(alive_mask, positions)

        info = {
            "alive_mask": alive_mask.copy(),
            "positions": np.asarray(positions, np.float32),
        }
        return obs, state, recv_mask, info

    def step(
        self,
        env_actions: np.ndarray,
        send: Optional[np.ndarray] = None,
        recv: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]
    ]:
        """
        Args:
          env_actions: (A,) int array of environment actions.

        Returns:
          obs:        (A, obs_dim) float32
          state:      (state_dim,) float32
          rewards:    (A,) float32
          dones:      (A,) bool           (scalar ep-done broadcast)
          recv_mask:  (A, A) bool         (no-self ∧ optional alive ∧ optional distance)
          info:       dict                (passthrough of env info and alive_mask)
        """
        # Env expects list/tensor of ints length A
        if isinstance(env_actions, np.ndarray):
            acts = env_actions.astype(np.int64).tolist()
        else:
            acts = list(map(int, env_actions))

        rewards, done, info = self.env.step(acts)

        # Pull fresh observations & state via getters (wrappers use them after step)
        obs = np.asarray(self.env.get_obs(), dtype=np.float32).reshape(
            self.n_agents, -1
        )
        state = np.asarray(self.env.get_state(), dtype=np.float32).reshape(-1)

        # Optional positions for distance-based mask
        positions = self._safe_get_positions()

        # Scalar episode done -> broadcast to all agents
        done_flag = bool(done)
        dones = np.full((self.n_agents,), done_flag, dtype=bool)

        # Rewards
        rew = np.asarray(rewards, dtype=np.float32).reshape(self.n_agents)

        # Alive mask heuristic (optional)
        alive_mask = (
            self._infer_alive(obs)
            if self.cfg.use_alive_heuristic
            else np.ones((self.n_agents,), dtype=bool)
        )
        recv_mask = self._build_recv_mask(alive_mask, positions)

        self._last_obs = obs
        self._last_positions = positions

        out_info: Dict[str, Any] = {
            "alive_mask": alive_mask.copy(),
            "raw_info": info,
        }
        return obs, state, rew, dones, recv_mask, out_info

    # -------------------- internals --------------------

    def _safe_get_positions(self) -> Optional[np.ndarray]:
        """Use wrapper.get_positions() if present: returns flat (A*2,) -> reshape (A,2)."""
        try:
            arr = self.env.get_positions()  # wrapper API
            arr = np.asarray(arr, dtype=np.float32).reshape(self.n_agents, 2)
            return arr
        except Exception:
            return None

    def _infer_alive(self, obs: np.ndarray) -> np.ndarray:
        """
        Heuristic: an agent is considered 'alive' if its observation vector is not all zeros.
        The ExpoComm wrappers set zeros for missing/dead agents in step() packing code.
        """
        # alive if any nonzero entry
        return np.abs(obs).sum(axis=1) > 0.0

    def _build_recv_mask(
        self, alive: np.ndarray, positions: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Base mask: no self; if use_alive_heuristic -> alive-to-alive only.
        If comm_radius is set and positions available, also enforce distance <= radius.
        """
        A = self.n_agents
        mask = np.ones((A, A), dtype=bool)
        np.fill_diagonal(mask, False)

        if alive is not None:
            mask &= np.outer(alive, alive)

        if (self.cfg.comm_radius is not None) and (positions is not None):
            # Pairwise distances
            dx = positions[:, None, 0] - positions[None, :, 0]
            dy = positions[:, None, 1] - positions[None, :, 1]
            dist2 = dx * dx + dy * dy
            mask &= dist2 <= (self.cfg.comm_radius**2)

        # Safety: ensure each row has at least one valid recipient (fallback allow self)
        row_has = mask.any(axis=1)
        if not row_has.all():
            idxs = np.nonzero(~row_has)[0]
            for i in idxs:
                # candidates = alive others
                candidates = (
                    np.where(alive & (np.arange(A) != i))[0]
                    if alive is not None
                    else np.array([], dtype=int)
                )
                if candidates.size > 0:
                    # pick the first (or nearest if positions available)
                    if positions is not None:
                        # nearest alive
                        diffs = positions[candidates] - positions[i]
                        j = candidates[np.argmin(np.sum(diffs * diffs, axis=1))]
                    else:
                        j = candidates[0]
                    mask[i, j] = True
                else:
                    # as a last resort, allow self to avoid all -inf logits
                    mask[i, i] = True
        return mask
