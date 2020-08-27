import copy
import logging

from .base import Agent

logger = logging.getLogger(__name__)


class GreedyAgent(Agent):
    def __init__(
        self,
        *args,
        verbose=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.sim_env = copy.deepcopy(self.env)
        self.sim_env.reset_at_episode_end = False  # Avoids expensive re-sampling of jets every time we parse a path

    def set_env(self, env):
        self.env = env
        self.sim_env = copy.deepcopy(self.env)
        self.sim_env.reset_at_episode_end = False  # Avoids expensive re-sampling of jets every time we parse a path

    def _predict(self, state):
        best_action, best_reward = None, -float("inf")
        for action in self._find_legal_actions(state):
            reward = self._parse_action(action)
            if reward > best_reward:
                best_action, best_reward = action, reward
        assert best_action is not None
        return best_action, {}

    def update(self, state, reward, action, done, next_state, next_reward, num_episode, **kwargs):
        pass

    def _parse_action(self, action):
        self.sim_env.set_internal_state(self.env.get_internal_state())
        self.sim_env.verbose = False
        _, reward, _, _ = self.sim_env.step(action)
        return reward
