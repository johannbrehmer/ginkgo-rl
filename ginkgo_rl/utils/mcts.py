import random
from collections import OrderedDict

import numpy as np
import torch

from ginkgo_rl.utils.normalization import AffineNormalizer


class MCTSNode:
    def __init__(self, parent, path, reward_normalizer=None, reward_min=None, reward_max=None, q_step=None):
        self.parent = parent
        self.path = path
        self.children = OrderedDict()

        self.reward_min = reward_min
        self.reward_max = reward_max
        self.reward_normalizer = (
            AffineNormalizer(hard_min=reward_min, hard_max=reward_max)
            if reward_normalizer is None
            else reward_normalizer
        )

        self.terminal = None  # None means undetermined
        self.n = 0  # Total visit count
        self.q = 0.0  # Total reward collected during playing out any trajectories that go through this state
        self.q_max = -float("inf")  # Highest reward encountered in this node

        self.q_step = q_step  # Reward received for the transition self.parent -> self
        self.n_beamsearch = 0  # Flag for beam search

    def expand(self, actions, step_rewards=None):
        self.terminal = False

        if step_rewards is None:
            step_rewards = [None for action in actions]

        for action, reward in zip(actions, step_rewards):
            if action in self.children:
                continue

            self.children[action] = MCTSNode(
                self,
                self.path + [action],
                self.reward_normalizer,
                reward_min=self.reward_min,
                reward_max=self.reward_max,
                q_step=reward,
            )

    def set_terminal(self, terminal):
        self.terminal = terminal

    def give_reward(self, reward, backup=True, beamsearch=False):
        reward = np.clip(reward, self.reward_min, self.reward_max)

        self.q_max = max(self.q_max, reward)
        self.q += reward
        self.n += 1
        self.reward_normalizer.update(reward)
        if beamsearch:
            self.n_beamsearch += 1

        if backup and self.parent:
            self.parent.give_reward(reward, backup=True, beamsearch=beamsearch)

    def get_reward(self, mode="mean"):
        """ Returns normalized mean reward (for `mode=="mean"`) or best reward (for `mode=="max"`) """
        assert mode in ["mean", "max"]

        if mode == "max":
            try:
                return self.reward_normalizer.evaluate(self.q_max)
            except TypeError:  # Happens with un-initializded normalizer
                return 0.5
        else:
            if self.n > 0:
                return self.reward_normalizer.evaluate(self.q / self.n)
            elif self.parent is not None:
                return self.parent.get_reward(mode)
            else:
                return 0.5

    def children_q_steps(self):
        return [child.q_step for child in self.children.values()]

    def select_random(self):
        assert self.children and not self.terminal
        return random.choice(list(self.children.keys()))

    def select_puct(self, policy_probs=None, mode="mean", c_puct=1.0):
        assert self.children and not self.terminal

        pucts = self._compute_pucts(policy_probs, mode=mode, c_puct=c_puct)
        assert len(pucts) > 0

        # Pick highest
        best_puct = -float("inf")
        choice = None
        for i, puct in enumerate(pucts):
            if puct > best_puct or choice is None:
                best_puct = puct
                choice = i

        assert choice is not None
        return list(self.children.keys())[choice]

    def select_best(self, mode="max"):
        best_q = -float("inf")
        choice = None

        for action, child in self.children.items():
            q = child.get_reward(mode=mode)
            if q > best_q or choice is None:
                best_q = q
                choice = action

        assert choice is not None
        return choice

    def select_beam_search(self, beam_size):
        choices = sorted(list(self.children.keys()), key=lambda x: self.children[x].q_step, reverse=True)[:beam_size]
        return choices

    def select_greedy(self):
        choices = self.select_beam_search(1)
        return 0 if not choices else choices[0]

    def prune(self):
        """ Steps into a subtree and updates all paths (and the new root's parent link) """
        self.path = self.path[1:]
        if len(self.path) == 0:
            self.parent = None

        for child in self.children.values():
            child.prune()

    def _compute_pucts(self, policy_probs=None, mode="mean", c_puct=1.0):
        if policy_probs is None:  # By default assume a uniform policy
            policy_probs = 1.0 / len(self)

        assert len(policy_probs) == len(self) > 0

        n_children = torch.tensor([child.n for child in self.children.values()], dtype=policy_probs.dtype)
        q_children = torch.tensor(
            [child.get_reward(mode=mode) for child in self.children.values()], dtype=policy_probs.dtype
        )
        pucts = q_children + c_puct * policy_probs * (self.n + 1.0e-9) ** 0.5 / (1.0 + n_children)

        assert len(n_children) == len(self) > 0
        assert len(q_children) == len(self) > 0
        assert len(pucts) == len(self) > 0

        return pucts

    def __len__(self):
        return len(self.children)
