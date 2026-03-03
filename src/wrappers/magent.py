import gym


# code adapted from https://github.com/ray-project/ray/blob/releases/1.8.0/rllib/env/wrappers/pettingzoo_env.py
class PettingZooEnv(gym.Env):
    def __init__(self, env):
        self.par_env = env
        # agent idx list
        self.agents = self.par_env.possible_agents

        # Get dictionaries of obs_spaces and act_spaces
        self.observation_spaces = self.par_env.observation_spaces
        self.action_spaces = self.par_env.action_spaces

        # Get first observation space, assuming all agents have equal space
        self.observation_space = self.observation_spaces[self.agents[0]]

        # Get first action space, assuming all agents have equal space
        self.action_space = self.action_spaces[self.agents[0]]

        assert all(
            obs_space == self.observation_space
            for obs_space in self.par_env.observation_spaces.values()
        ), (
            "Observation spaces for all agents must be identical. Perhaps "
            "SuperSuit's pad_observations wrapper can help (useage: "
            "`supersuit.aec_wrappers.pad_observations(env)`"
        )

        assert all(
            act_space == self.action_space
            for act_space in self.par_env.action_spaces.values()
        ), (
            "Action spaces for all agents must be identical. Perhaps "
            "SuperSuit's pad_action_space wrapper can help (useage: "
            "`supersuit.aec_wrappers.pad_action_space(env)`"
        )

        self.reset()

    def reset(self):
        obss = self.par_env.reset()
        return obss

    def step(self, action_dict):
        obss, rews, dones, infos = self.par_env.step(action_dict)
        dones["__all__"] = all(dones.values())
        return obss, rews, dones, infos

    def close(self):
        self.par_env.close()

    def seed(self, seed=None):
        self.par_env.seed(seed)

    def render(self, mode="human"):
        return self.par_env.render(mode)
