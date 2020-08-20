from collections import OrderedDict
import numpy as np
import torch
import random
import copy

from .base import Agent
from ..utils.normalization import AffineNormalizer


class MCTSNode:
    def __init__(self, parent, path, reward_normalizer=None, reward_min=0., reward_max=1.):
        self.parent = parent
        self.path = path
        self.children = OrderedDict()
        self.reward_normalizer = AffineNormalizer(reward_min, reward_max) if reward_normalizer is None else reward_normalizer
        self.terminal = None  # None means undetermined

        self.q = 0.0  # Total reward
        self.n = 0  # Total visit count

    def expand(self, actions):
        self.terminal = False

        for action in actions:
            if action in self.children:
                continue

            self.children[action] = MCTSNode(self, self.path + [action], self.reward_normalizer)

    def set_terminal(self, terminal):
        self.terminal = terminal

    def select_random(self):
        assert self.children and not self.terminal
        return random.choice(list(self.children.keys()))

    def select_puct(self, policy_probs=None, c_puct=1.0):
        assert self.children and not self.terminal

        pucts = self._compute_pucts(policy_probs, c_puct=c_puct)

        # Pick highest
        best_puct = - float("inf")
        choice = None
        for i, puct in enumerate(pucts):
            if puct > best_puct:
                best_puct = puct
                choice = i

        assert choice is not None
        return list(self.children.keys())[choice]

    def select_best(self):
        best_q = - float("inf")
        choice = None
        for action, child in self.children.values():
            if child.q > best_q:
                best_q = child.q
                choice = action

        assert choice is not None
        return choice

    def give_reward(self, reward, backup=True):
        self.q += reward
        self.n += 1
        self.reward_normalizer.update(reward)

        if backup and self.parent:
            self.parent.give_reward(reward, backup=True)

    def prune(self, update_q=0.):
        """ Steps into a subtree and updates all paths (and the new root's parent link) """
        self.path = self.path[1:]
        self.q += update_q

        if len(self.path) == 0:
            self.parent = None

        for child in self.children.values():
            child.prune(update_q=update_q)

    def _compute_pucts(self, policy_probs=None, c_puct=1.0):
        if policy_probs is None:  # By default assume a uniform policy
            policy_probs = 1. / len(self)

        q_children = np.asarray([self.reward_normalizer.evaluate(child.q) for child in self.children.values()])
        n_children = np.asarray([child.n for child in self.children.values()])
        mean_q_children = np.where(n_children > 0, q_children / n_children, self.reward_normalizer.evaluate(self.q) / self.n)

        pucts = mean_q_children + c_puct * policy_probs * (self.n + 1.e-9) ** 0.5 / (1. + n_children)

        return pucts

    def __len__(self):
        return len(self.children)


class BaseMCAgent(Agent):
    def __init__(
        self,
        n_mc=1000,
        c_puct=1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_mc = n_mc
        self.c_puct = c_puct
        self.mcts_head = self._init_mcts()

    def learn(self, total_timesteps):
        raise NotImplementedError

    def predict(self, state):
        # If there is only one action, don't bother with MCTS
        actions = self._find_legal_actions(state)
        if len(actions) == 1:
            return actions[0]

        # Run MCTS
        action, info = self._mcts(n=self.n)
        return action, info

    def _update(self, state, reward, value, action, done, next_state, next_reward, num_episode):
        """ Updates after environment reaction """

        # MCTS updates
        if done:
            # Reset MCTS when done with an episode
            self.mcts_head = self._init_mcts()
        else:
            # Update MCTS tree when deciding on an action
            self.mcts_head = self.mcts_head.children[action]
            self.mcts_head.prune(update_q= - reward)  # This updates the node.path and node.q fields

        # Policy training
        raise NotImplementedError

    def _parse_path(self, state, path):
        """ Given a path (list of actions), computes the resulting environment state and total reward """

        # Ensure that self.env always has current state
        assert np.isclose(state, self.env.state)
        env = copy.deepcopy(self.env)

        # Follow path
        total_reward = 0.
        for action in path:
            state, reward, done, info = env.step(action)
            total_reward += reward

        return state, total_reward

    @staticmethod
    def _init_mcts():
        """ Initializes MCTS tree """
        return MCTSNode(None, [])

    def _check_if_path_terminates(self, initial_state, path):
        """ Given an initial state and a path (list of actions), check if the final state is terminal """
        return len(path) >= len(self._legal_action_extractor(initial_state)) - 1

    def _mcts(self, state, n):
        """ Run Monte-Carl tree search from state for n trajectories"""

        # Set up MCTS graph
        if self.mcts_head.terminal is None:
            self.mcts_head.set_terminal(self._check_if_path_is_terminal(state, self.mcts_head.path))

        # Run n trajectories using PUCT based on policy
        for _ in range(n):
            node = self.mcts_head
            total_reward = 0.0

            while not node.terminal:
                this_state, total_reward = self._parse_path(state, node.path)

                # Expand
                if not node.children:
                    actions = self._legal_action_extractor(this_state)
                    node.expand(actions)

                # Select
                policy_probs = self._evaluate_policy(this_state, node.children.keys())
                node = node.select_puct(policy_probs, self.c_puct)
                if node.terminal is None:
                    node.set_terminal(self._check_if_path_is_terminal(state, node.path))

            # Backup
            node.give_reward(total_reward, backup=True)

        # Select best action
        action = self.mcts_head.select_best()
        info = {}

        return action, info

    def _evaluate_policy(self, state, legal_actions):
        """ Evaluates the policy on the state and returns the probabilities for each action """
        raise NotImplementedError
