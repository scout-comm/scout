# wherever you register PettingZoo envs (same file as battle registration)
from gym.spaces import Dict as GymDict, Box
import supersuit as ss
import numpy as np
import math
from pettingzoo.sisl import pursuit_v3  # typical import path
from .multiagentenv import MultiAgentEnv
from .magent import PettingZooEnv  # your existing shim

REGISTRY = {}
REGISTRY["pursuit_v3"] = pursuit_v3.parallel_env

# raw C = 3 (map, pursuers, evaders); processed = 2 (pursuers, evaders); state = 2
processed_channel_dim_dict = {"pursuit_v3": (3, 2, 2)}
print("Registered pursuit_v3 wrapper in REGISTRY.")


class _PursuitWrapper(MultiAgentEnv):
    """
    Wrapper for PettingZoo SISL pursuit_v3 with fixed-shape obs/actions.
    Processed obs: concat of [pursuer_layer, evader_layer] in the local patch.
    State: same two channels from the first agent's patch.
    """

    def __init__(self, **env_config):
        map_key = env_config.pop("map_name", "pursuit_v3")
        self.seed = env_config.pop("seed", None)
        self.episode_limit = env_config.get("max_cycles", 500)

        # Build PZ env and seed
        base_env = REGISTRY[map_key](**env_config)
        try:
            base_env.reset(seed=self.seed)
        except TypeError:
            if hasattr(base_env, "seed"):
                base_env.seed(self.seed)

        # Pad to uniform shapes
        env = ss.pad_observations_v0(base_env)
        env = ss.pad_action_space_v0(env)
        self.env = PettingZooEnv(env)

        # Channel bookkeeping
        self.raw_channel_dim = processed_channel_dim_dict[map_key][0]  # 3
        self.processed_channel_dim = processed_channel_dim_dict[map_key][1]  # 2
        self.state_channel_dim = processed_channel_dim_dict[map_key][2]  # 2

        # Introspect obs shape (channel-first or channel-last)
        shp = tuple(int(s) for s in self.env.observation_space.shape)
        if len(shp) == 3 and shp[0] in (1, 2, 3, 4, 5, 6, 7, 8, 9) and shp[1] == shp[2]:
            # channel-first: (C,H,W)
            self._channels_first = True
            self._C, self._H, self._W = shp
        elif len(shp) == 3:
            # channel-last: (H,W,C)
            self._channels_first = False
            self._H, self._W, self._C = shp
        else:
            raise RuntimeError(
                f"Unexpected pursuit obs shape {shp}; expected 3D image-like."
            )

        # Spaces
        self.action_space = self.env.action_space
        self.observation_space = GymDict(
            {
                # We mirror the underlying space to keep dtype/range consistent
                "obs": Box(
                    low=self.env.observation_space.low,
                    high=self.env.observation_space.high,
                    dtype=self.env.observation_space.dtype,
                ),
                "state": Box(
                    low=self.env.observation_space.low,
                    high=self.env.observation_space.high,
                    dtype=self.env.observation_space.dtype,
                ),
            }
        )

        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        env_config["map_name"] = map_key
        self.env_config = env_config

        # step buffers
        self._obs = None
        self._episode_length = 0

    # ------------- core API -------------

    def reset(self):
        obss = self.env.reset()  # dict: agent -> obs (3,H,W) or (H,W,3)
        obs = []
        for a in self.agents:
            obs.append(obss[a].flatten())
        self._obs = np.array(obs, dtype=np.float32)
        self._episode_length = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        # list[int/np/tensor] -> dict
        action_dict = {}
        for agent, action in zip(self.agents, actions):
            action_dict[agent] = (
                int(action)
                if not isinstance(action, (np.ndarray,))
                else int(action.item())
            )

        obss, rews, dones, _pos_infos = self.env.step(action_dict)

        rewards = []
        obs = []
        for agent in self.agents:
            rewards.append(rews.get(agent, 0.0))
            if agent in obss:
                obs.append(obss[agent].flatten())
            else:
                obs.append(np.zeros(self._C * self._H * self._W, dtype=np.float32))
        self._obs = np.asarray(obs, dtype=np.float32)

        done = bool(dones["__all__"])
        info = {}
        if done or self._episode_length >= self.episode_limit:
            info["episode_length"] = self._episode_length
        else:
            self._episode_length += 1

        return np.asarray(rewards, dtype=np.float32), done, info

    # ------------- adapter-facing helpers -------------

    def _obs_image(self):
        """
        Returns the latest obs as (A, H, W, C) for easy channel picking.
        """
        A = self.n_agents
        if self._channels_first:
            x = self._obs.reshape(A, self._C, self._H, self._W).transpose(
                0, 2, 3, 1
            )  # -> (A,H,W,C)
        else:
            x = self._obs.reshape(A, self._H, self._W, self._C)
        return x

    def get_obs(self):
        """
        Processed obs: concat pursuer layer (ch=1) and evader layer (ch=2) -> (A, H*W*2).
        """
        x = self._obs_image()  # (A,H,W,C)
        assert self._C >= 3, "pursuit expects 3 raw channels (map, pursuers, evaders)"
        pursuers = x[:, :, :, 1:2]
        evaders = x[:, :, :, 2:3]
        proc = np.concatenate([pursuers, evaders], axis=-1)  # (A,H,W,2)
        return proc.reshape(self.n_agents, -1)

    def get_state(self):
        """
        State: same two channels from the first agent (shape: H*W*2).
        """
        x = self._obs_image()  # (A,H,W,C)
        pursuers = x[0:1, :, :, 1:2]
        evaders = x[0:1, :, :, 2:3]
        s = np.concatenate([pursuers, evaders], axis=-1).reshape(-1)
        return s

    # Optional; not implemented for SISL pursuit (adapter will fall back)
    # def get_positions(self):
    #     raise NotImplementedError

    def get_avail_actions(self):
        valid = [1] * self.action_space.n
        return [valid for _ in range(self.n_agents)]

    def close(self):
        self.env.close()

    def get_env_info(self):
        # Use H*W from resolved dims (don't rely on channel ordering in the space)
        hw = self._H * self._W
        return {
            "state_shape": hw * self.state_channel_dim,  # H*W*2
            "obs_shape": hw * self.processed_channel_dim,  # H*W*2
            "n_actions": self.action_space.n,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
