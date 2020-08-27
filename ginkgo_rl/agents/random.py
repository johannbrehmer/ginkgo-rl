import copy
import logging
import random

from .base import Agent

logger = logging.getLogger(__name__)


class RandomAgent(Agent):
    def _predict(self, state):
        actions = self._find_legal_actions(state)
        action = random.choice(actions)
        return action, {}

    def update(self, state, reward, action, done, next_state, next_reward, num_episode, **kwargs):
        pass
