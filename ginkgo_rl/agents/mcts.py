from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import random
import copy
import logging

from .base import Agent
from ..utils.normalization import AffineNormalizer
from ..utils.nets import MultiHeadedMLP

logger = logging.getLogger(__name__)


class MCTSNode:
    # TODO: temperature selection with probability ~ n(a)^(1/temperature)

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

        if self.n > 0:
            q_children = np.asarray([self.reward_normalizer.evaluate(child.q) for child in self.children.values()])
            n_children = np.asarray([child.n for child in self.children.values()])
            mean_q_children = np.where(n_children > 0, q_children / n_children, self.reward_normalizer.evaluate(self.q) / self.n)
        else:
            n_children = np.zeros(len(self))
            mean_q_children = 0.5 * np.ones(len(self))

        pucts = mean_q_children + c_puct * policy_probs * (self.n + 1.e-9) ** 0.5 / (1. + n_children)

        return pucts

    def __len__(self):
        return len(self.children)


class BaseMCTSAgent(Agent):
    def __init__(
        self,
        *args,
        n_mc=100,
        c_puct=1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_mc = n_mc
        self.c_puct = c_puct
        self.mcts_head = self._init_mcts()

    def _predict(self, state):
        # If there is only one action, don't bother with MCTS
        actions = self._find_legal_actions(state)
        if len(actions) == 1:
            return actions[0], {"log_prob": 1.0}

        # Run MCTS
        action, info = self._mcts(state, n=self.n_mc)
        return action, info

    def _update(self, state, reward, value, action, done, next_state, next_reward, num_episode, **kwargs):
        """ Updates after environment reaction """

        # Memorize step
        self.history.store(log_prob=kwargs["log_prob"], reward=reward)

        # MCTS updates
        if done:
            # Reset MCTS when done with an episode
            self.mcts_head = self._init_mcts()
        else:
            # Update MCTS tree when deciding on an action
            self.mcts_head = self.mcts_head.children[action]
            self.mcts_head.prune(update_q= - reward)  # This updates the node.path and node.q fields

        # No further steps after each step
        if not done or not self.training:
            return

        # Policy training (if necessary)
        self._train()

        # Reset memory for next episode
        self.history.clear()

    def _parse_path(self, state, path):
        """ Given a path (list of actions), computes the resulting environment state and total reward """

        # Store env state state
        assert np.all(np.isclose(state, self.env.state))
        internal_state = self.env.get_internal_state()
        verbose_state = self.env.verbose
        self.env.verbose = False

        # Follow path
        total_reward = 0.
        for action in path:
            state, reward, done, info = self.env.step(action)
            total_reward += reward

        # Reset env
        self.env.verbose = verbose_state
        self.env.set_internal_state(internal_state)

        return state, total_reward

    @staticmethod
    def _init_mcts():
        """ Initializes MCTS tree """
        return MCTSNode(None, [])

    def _check_if_path_terminates(self, initial_state, path):
        """ Given an initial state and a path (list of actions), check if the final state is terminal """
        return len(path) >= len(self._find_legal_actions(initial_state)) - 1

    def _mcts(self, state, n):
        """ Run Monte-Carl tree search from state for n trajectories"""

        # Set up MCTS graph
        if self.mcts_head.terminal is None:
            self.mcts_head.set_terminal(self._check_if_path_terminates(state, self.mcts_head.path))

        # Run n trajectories using PUCT based on policy
        for _ in range(n):
            node = self.mcts_head
            total_reward = 0.0

            while not node.terminal:
                this_state, total_reward = self._parse_path(state, node.path)

                # Expand
                if not node.children:
                    actions = self._find_legal_actions(this_state)
                    node.expand(actions)

                # Select
                policy_probs = self._evaluate_policy(this_state, node.children.keys())
                action = node.select_puct(policy_probs, self.c_puct)
                node = node.children[action]
                if node.terminal is None:
                    terminal = self._check_if_path_terminates(state, node.path)
                    node.set_terminal(terminal)

            # Backup
            node.give_reward(total_reward, backup=True)

        # Select best action
        action = self.mcts_head.select_best()
        info = {"log_prob": torch.log(self._evaluate_policy(state, self._find_legal_actions(state), action=action))}

        # Debug output
        probs_ =  self._evaluate_policy(state, self._find_legal_actions(state))
        logger.debug("MCTS results:")
        for i, (action_, node_) in enumerate(self.mcts_head.children.items()):
            chosen = 'x' if action_ == action else ' '
            logger.debug(
                f"  {chosen} {action:>3d}: "
                f"p = {probs_[i].detach().item():.2f}, "
                f"n = {node_.n:>3d}, "
                f"q = {node_.q / node_.n:>5.1f}, "
                f"q_norm = {node_.reward_normalizer.evaluate(node_.q) / node_.n:>5.1f}"
            )

        return action, info

    def _evaluate_policy(self, state, legal_actions, action=None):
        """ Evaluates the policy on the state and returns the probabilities for a given action or all legal actions """
        raise NotImplementedError

    def _train(self):
        """ Policy updates at end of episode """
        raise NotImplementedError


class RandomMCTSAgent(BaseMCTSAgent):
    def _evaluate_policy(self, state, legal_actions, action=None):
        """ Evaluates the policy on the state and returns the probabilities for a given action or all legal actions """
        if action is not None:
            1. / len(legal_actions)
        else:
            return 1. / len(legal_actions) * np.ones(len(legal_actions))

    def _train(self):
        pass


class MCTSAgent(BaseMCTSAgent):
    def __init__(self, *args, hidden_sizes=(100,100,), activation=nn.ReLU, **kwargs):
        super().__init__(*args, **kwargs)

        self.actor = MultiHeadedMLP(self.state_length + 1, hidden_sizes=hidden_sizes, head_sizes=(1,), activation=activation, head_activations=(None,))
        self.softmax = nn.Softmax(dim=0)

    def _evaluate_policy(self, state, legal_actions, action=None):
        batch_states = []

        for action in legal_actions:
            action_ = torch.tensor([action]).to(self.device, self.dtype)
            batch_states.append(torch.cat((action_, state), dim=0).unsqueeze(0))

        batch_states = torch.cat(batch_states, dim=0)

        (probs,) = self.actor(batch_states)
        probs = self.softmax(probs).flatten()

        if action is not None:
            assert action in legal_actions
            return probs[legal_actions.index(action)]

        return probs

    def _train(self):
        # Roll out last episode
        rollout = self.history.rollout()
        log_probs = torch.stack(rollout["log_prob"], dim=0)

        # Compute loss: train policy to get closer to (deterministic) MCS choice
        loss = -torch.sum(log_probs)

        # Gradient update
        self._gradient_step(loss)
