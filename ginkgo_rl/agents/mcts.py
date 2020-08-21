from collections import OrderedDict
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

    def __init__(self, parent, path, reward_normalizer=None, reward_min=None, reward_max=None):
        self.parent = parent
        self.path = path
        self.children = OrderedDict()
        self.reward_normalizer = AffineNormalizer(hard_min=reward_min, hard_max=reward_max) if reward_normalizer is None else reward_normalizer
        self.terminal = None  # None means undetermined

        self.q = 0.0  # Total reward
        self.n = 0  # Total visit count

    def get_nmq(self):
        """ Returns normalized mean reward """
        if self.n > 0:
            return self.reward_normalizer.evaluate(self.q / self.n)
        elif self.parent is not None:
            return self.parent.get_nmq()
        else:
            return 0.5

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
        assert len(pucts) > 0

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

        for action, child in self.children.items():
            q = child.get_nmq()
            if q > best_q:
                best_q = q
                choice = action

        assert choice is not None
        return choice

    def give_reward(self, reward, backup=True):
        self.q += reward
        self.n += 1
        self.reward_normalizer.update(reward)

        if backup and self.parent:
            self.parent.give_reward(reward, backup=True)

    def prune(self):
        """ Steps into a subtree and updates all paths (and the new root's parent link) """
        self.path = self.path[1:]

        if len(self.path) == 0:
            self.parent = None

        for child in self.children.values():
            child.prune()

    def _compute_pucts(self, policy_probs=None, c_puct=1.0):
        if policy_probs is None:  # By default assume a uniform policy
            policy_probs = 1. / len(self)

        assert len(policy_probs) == len(self) > 0

        n_children = torch.tensor([child.n for child in self.children.values()], dtype=policy_probs.dtype)
        nmq_children = torch.tensor([child.get_nmq() for child in self.children.values()], dtype=policy_probs.dtype)
        pucts = nmq_children + c_puct * policy_probs * (self.n + 1.e-9) ** 0.5 / (1. + n_children)

        assert len(n_children) == len(self) > 0
        assert len(nmq_children) == len(self) > 0
        assert len(pucts) == len(self) > 0

        return pucts

    def __len__(self):
        return len(self.children)


class BaseMCTSAgent(Agent):
    def __init__(
        self,
        *args,
        n_mc_target=200,
        n_mc_min=10,
        c_puct=1.0,
        reward_range=(-200., 0.),
        verbose=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_mc_target = n_mc_target
        self.n_mc_min = n_mc_min
        self.c_puct = c_puct
        self.reward_range = reward_range
        self.sim_env = copy.deepcopy(self.env)
        self.verbose = verbose

        self._init_episode()

    def set_env(self, env):
        self.env = env
        self.sim_env = copy.deepcopy(self.env)

    def _predict(self, state):
        # If there is only one action, don't bother with MCTS
        actions = self._find_legal_actions(state)
        assert actions

        # Run MCTS
        action, info = self._mcts(state)
        return action, info

    def update(self, state, reward, action, done, next_state, next_reward, num_episode, **kwargs):
        """ Updates after environment reaction """

        # Keep track of total reward
        self.episode_reward += next_reward

        # MCTS updates
        if done:
            # Reset MCTS when done with an episode
            self._init_episode()
        else:
            # Update MCTS tree when deciding on an action
            self.mcts_head = self.mcts_head.children[action]
            self.mcts_head.prune()  # This updates the node.path

        # Memorize step
        if self.training:
            self.history.store(log_prob=kwargs["log_prob"], reward=reward)

        if self.training and done:
            # Training
            self._train()

            # Reset memory for next episode
            self.history.clear()

    def _parse_path(self, state, path):
        """ Given a path (list of actions), computes the resulting environment state and total reward """

        # Store env state state
        self.sim_env.set_internal_state(self.env.get_internal_state())
        self.sim_env.verbose = False

        # Follow path
        total_reward = 0.0
        terminal = False

        for action in path:
            state, reward, done, info = self.sim_env.step(action)
            total_reward += reward

            if done:
                terminal = True
                break

        state = self._tensorize(state)
        return state, total_reward, terminal

    def _init_episode(self):
        """ Initializes MCTS tree and total reward so far """
        self.mcts_head = MCTSNode(None, [], reward_min=self.reward_range[0], reward_max=self.reward_range[1])
        self.episode_reward = 0.0

    def _mcts(self, state, max_steps=1000):
        """ Run Monte-Carl tree search from state for n trajectories"""

        if len(self.mcts_head.children) == 1:
            n = 1
        else:
            n = max(self.n_mc_target - self.mcts_head.n, self.n_mc_min)
        logger.debug(f"Starting MCTS with {n} trajectories")

        for i in range(n):
            if self.verbose: logger.debug(f"Initializing MCTS trajectory {i+1} / {n}")
            node = self.mcts_head
            total_reward = self.episode_reward

            for _ in range(max_steps):
                # Parse current state
                this_state, total_reward, terminal = self._parse_path(state, node.path)
                node.set_terminal(terminal)
                if self.verbose: logger.debug(f"  Node {node.path}")

                # Termination
                if terminal:
                    if self.verbose: logger.debug(f"  Node is terminal")
                    break

                # Expand
                if not node.children:
                    actions = self._find_legal_actions(this_state)
                    if not actions:
                        logger.warning("Could not find legal actions! Treating state as terminal.")
                        break

                    if self.verbose: logger.debug(f"    Expanding: {len(actions)} legal actions")
                    node.expand(actions)

                # Select
                policy_probs = self._evaluate_policy(this_state, node.children.keys())
                action = node.select_puct(policy_probs, self.c_puct)
                if self.verbose: logger.debug(f"    Selecting action {action}")
                node = node.children[action]

            # Backup
            if self.verbose: logger.debug(f"  Backing up total reward of {total_reward}")
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
                f"  {chosen} {action_:>3d}: "
                f"p = {probs_[i].detach().item():.2f}, "
                f"n = {node_.n:>3d}, "
                f"q = {node_.q / (node_.n + 1.e-9):>5.1f}, "
                f"nmq = {node_.get_nmq():>4.2f}"
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
            torch.tensor(1. / len(legal_actions), dtype=self.dtype)
        else:
            return 1. / len(legal_actions) * torch.ones(len(legal_actions), dtype=self.dtype)

    def _train(self):
        pass


class MCTSAgent(BaseMCTSAgent):
    def __init__(self, *args, hidden_sizes=(100,100,), activation=nn.ReLU(), **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = MultiHeadedMLP(1 + self.state_length, hidden_sizes=hidden_sizes, head_sizes=(1,), activation=activation, head_activations=(None,))
        self.softmax = nn.Softmax(dim=0)

    def _evaluate_policy(self, state, legal_actions, action=None):
        batch_states = self._batch_state(state, legal_actions)
        (probs,) = self.actor(batch_states)
        probs = self.softmax(probs).flatten()

        if action is not None:
            assert action in legal_actions
            return probs[legal_actions.index(action)]

        return probs

    def _batch_state(self, state, legal_actions):
        state_ = state.view(-1)
        batch_states = []
        for action in legal_actions:
            action_ = torch.tensor([action]).to(self.device, self.dtype)
            batch_states.append(torch.cat((action_, state_), dim=0).unsqueeze(0))
        batch_states = torch.cat(batch_states, dim=0)
        return batch_states

    def _train(self):
        # Roll out last episode
        rollout = self.history.rollout()
        log_probs = torch.stack(rollout["log_prob"], dim=0)

        # Compute loss: train policy to get closer to (deterministic) MCS choice
        loss = -torch.sum(log_probs)

        # Gradient update
        self._gradient_step(loss)
