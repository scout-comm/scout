# src/envs/pettingzoo_wrappers.py
from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.mpe import simple_spread_v3
import numpy as np
from typing import Dict, List, Tuple


class BudgetedCommBus:
    def __init__(self, agents, budget: int = 2):
        self.budget = budget
        self.reset(agents)

    def reset(self, agents):
        self.agents = list(agents)
        self.remaining = {a: self.budget for a in self.agents}
        self.inbox = {a: [] for a in self.agents}
        self.log = []
        self.step_edges = []
        # optional, set by trainer for nicer IDs
        self.agent_to_idx = {a: i for i, a in enumerate(self.agents)}

    def send(self, t_cycle, sender: str, recipient: str, payload) -> bool:
        """
        Try to send one message. Logs delivery and edge (even if dropped).
        Returns True if delivered (budget available), False otherwise.
        """
        ok = False
        if self.remaining.get(sender, 0) > 0 and recipient in self.inbox:
            self.remaining[sender] -= 1
            self.inbox[recipient].append((sender, payload))
            ok = True

        dropped = int(not ok)
        s_idx = self.agent_to_idx.get(sender, -1)
        r_idx = self.agent_to_idx.get(recipient, -1)
        self.step_edges.append((t_cycle, s_idx, r_idx, dropped))
        self.log.append((t_cycle, sender, recipient, payload, ok))
        return ok

    def read(self, agent):
        msgs = self.inbox.get(agent, [])
        self.inbox[agent] = []
        return msgs


class GroupVisibilityWrapper(BaseWrapper):
    """
    Runtime-configurable visibility mask for PettingZoo MPE (simple_spread).

    Observation layout (per PettingZoo simple_spread):
      [ self_vel(2), self_pos(2),
        landmark_rel_pos (2*L),
        other_agents_rel_pos (2*(N-1))  -- in env.agents order excluding self,
        other_agents_vel (2*(N-1))      -- same order as above
      ]

    Modes:
      - "all": no masking
      - "self_only": hide everyone except the observing agent
      - "group_only": keep only agents in observer's current group (set via set_visible_groups)
    """

    def __init__(self, env, agent_order=None):
        super().__init__(env)
        self.mode = "all"
        self.visible_groups: Dict[int, set] | None = None
        self.agent_order = agent_order or list(env.agents)

        # lazily inferred counts
        self._n_agents = None
        self._n_landmarks = None

        # mapping: per observer -> per other -> (pos_slice, vel_slice)
        self.other_slices: Dict[str, Dict[str, Tuple[slice, slice]]] = {}
        self._build_other_agent_slices()

    # ---------- public API ----------
    def set_mode(self, mode: str):
        assert mode in ("all", "self_only", "group_only")
        self.mode = mode

    def set_visible_groups(self, groups: Dict[int, List[str]]):
        # normalize to sets for fast lookup
        self.visible_groups = {g: set(members) for g, members in groups.items()}

    # ---------- PettingZoo hooks ----------
    def last(self, observe=True):
        obs, rew, term, trunc, info = self.env.last(observe=observe)
        if observe and obs is not None:
            obs = self._apply_visibility(self.env.agent_selection, obs)
        return obs, rew, term, trunc, info

    def observe(self, agent):
        obs = self.env.observe(agent)
        return self._apply_visibility(agent, obs)

    # --- replace _infer_counts, _build_other_agent_slices, _mask_others as below ---

    def _infer_counts(self):
        if self._n_agents is None:
            self._n_agents = len(self.env.agents)
        if self._n_landmarks is None:
            self._n_landmarks = len(self.env.unwrapped.world.landmarks)
        # communication channel width per agent (0, 2, ...). In simple_spread_v3 it's 2 by default.
        self._dim_c = getattr(self.env.unwrapped.world, "dim_c", 0)

    def _build_other_agent_slices(self):
        """
        For each observer, precompute obs indices for each *other* agent:
        obs = [ self_vel(2), self_pos(2),
                landmarks(2*L),
                other_pos(2*(N-1)),
                comm(dim_c*(N-1)) ]
        """
        self._infer_counts()
        nA = self._n_agents
        nL = self._n_landmarks
        dim_c = self._dim_c

        base = 2 + 2 + 2 * nL  # start of other_pos
        other_pos_len = 2 * (nA - 1)
        comm_start = base + other_pos_len  # start of comm block
        per_obs = {}

        for obs_agent in self.env.agents:
            order = [a for a in self.env.agents if a != obs_agent]  # order used by env
            per_other = {}
            # other positions
            for idx, other in enumerate(order):
                pos_sl = slice(base + 2 * idx, base + 2 * (idx + 1))
                # comm slice for this 'other' (may be 0-width if dim_c == 0)
                if dim_c > 0:
                    c_sl = slice(
                        comm_start + dim_c * idx, comm_start + dim_c * (idx + 1)
                    )
                else:
                    c_sl = slice(0, 0)
                per_other[other] = (pos_sl, c_sl)
            per_obs[obs_agent] = per_other
        self.other_slices = per_obs

    def _mask_others(
        self, obs_vec: np.ndarray, observer: str, keep: set[str]
    ) -> np.ndarray:
        """
        Zero out all other-agent features (pos and comm) not in `keep`.
        `keep` should contain *other agents* to retain for this observer.
        """
        per_other = self.other_slices.get(observer, {})
        masked = obs_vec.copy()
        for other, (pos_sl, comm_sl) in per_other.items():
            if other not in keep:
                masked[pos_sl] = 0.0
                if comm_sl.stop > comm_sl.start:  # has comm dims
                    masked[comm_sl] = 0.0
        return masked

    def _apply_visibility(self, agent: str, obs: np.ndarray) -> np.ndarray:
        if self.mode == "all":
            return obs
        if self.mode == "self_only":
            # keep no others
            return self._mask_others(obs, agent, keep=set())
        if self.mode == "group_only":
            keep = set()
            if self.visible_groups is not None:
                for members in self.visible_groups.values():
                    if agent in members:
                        keep = set(members) - {agent}  # only keep other members
                        break
            return self._mask_others(obs, agent, keep=keep)
        return obs


def load_mpe(
    n_agents=4,
    max_cycles=50,
    budget=2,
    hide_others=True,
    render_mode=None,
    dynamic_rescaling=True,
):
    """
    Create simple_spread with dynamic visibility.
    NOTE: We always build the base env with full observations; masking is done by the wrapper.
    """
    env = simple_spread_v3.env(
        N=n_agents,
        local_ratio=0.5,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=render_mode,
        dynamic_rescaling=dynamic_rescaling,
    )
    env.reset(seed=0)

    # Wrap for runtime visibility control
    env = GroupVisibilityWrapper(env)

    # Initial visibility mode (eval usually passes hide_others=True)
    if hide_others:
        env.set_mode("self_only")
    else:
        env.set_mode("all")

    bus = BudgetedCommBus(env.agents, budget)
    return env, bus
