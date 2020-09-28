import copy
import logging

from .base import Agent

logger = logging.getLogger(__name__)


class GreedyAgent(Agent):
    def __init__(self, *args, verbose=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.sim_env = copy.deepcopy(self.env)
        self.sim_env.reset_at_episode_end = False  # Avoids expensive re-sampling of jets every time we parse a path

        self.episode_likelihood_evaluations = 0

    def set_env(self, env):
        self.env = env
        self.sim_env = copy.deepcopy(self.env)
        self.sim_env.reset_at_episode_end = False  # Avoids expensive re-sampling of jets every time we parse a path

    def _predict(self, state):
        best_action, best_reward = None, -float("inf")
        legal_actions = self._find_legal_actions(state)
        rewards = []

        for action in legal_actions:
            reward = self._parse_action(action)
            rewards.append(reward)

            if reward > best_reward or best_action is None:
                best_action, best_reward = action, reward

        if self.verbose > 0:
            self._report_decision(legal_actions, rewards, best_action)

        assert best_action is not None
        return best_action, {"likelihood_evaluations": self.episode_likelihood_evaluations}  # TODO

    def update(self, state, reward, action, done, next_state, next_reward, num_episode, **kwargs):
        if done:
            self.episode_likelihood_evaluations = 0

    def _parse_action(self, action):
        self.sim_env.set_internal_state(self.env.get_internal_state())
        self.sim_env.verbose = False
        _, reward, _, _ = self.sim_env.step(action)
        self.episode_likelihood_evaluations += 1
        return reward

    def _report_decision(self, legal_actions, log_likelihoods, chosen_action):
        logger.debug(f"Greedy results:")
        for i, (action_, log_likelihood) in enumerate(zip(legal_actions, log_likelihoods)):
            is_chosen = "*" if action_ == chosen_action else " "
            logger.debug(f" {is_chosen} {action_:>2d}: " f"log likelihood = {log_likelihood:6.2f}")
