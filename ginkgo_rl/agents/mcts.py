from collections import OrderedDict
import numpy as np
import torch
import random

from .base import Agent


class MCTSNode:
    def __init__(self, parent, path):
        self.parent = parent
        self.path = path
        self.children = OrderedDict()

        self.terminal = False

        self.q = 0  # Total reward
        self.n = 0  # Total visit count

    def expand(self, actions):
        self.terminal = False

        for action in actions:
            if action in self.children:
                continue

            self.children[action] = MCTSNode(self, self.path + [action])

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

    def give_reward(self, reward, backpropagate=True):
        self.q += reward
        self.n += 1

        if backpropagate and self.parent:
            self.parent.give_reward(reward, backpropagate=True)

    def _compute_pucts(self, policy_probs=None, c_puct=1.0):
        if policy_probs is None:  # By default assume a uniform policy
            policy_probs = 1. / len(self)

        q_children = np.asarray([child.q for child in self.children.values()])
        n_children = np.asarray([child.n for child in self.children.values()])
        mean_q_children = np.where(n_children > 0, q_children / n_children, self.q / self.n)

        pucts = mean_q_children + c_puct * policy_probs * (self.n + 1.e-9) ** 0.5 / (1. + n_children)

        return pucts

    def __len__(self):
        return len(self.children)


class BaseMCAgent(Agent):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)


    def learn(self, total_timesteps):
        raise NotImplementedError

    def predict(self, state):
        # If there is only one action, don't bother with MCTS
        actions = self._find_legal_actions(state)
        if len(actions) == 1:
            return actions[0]

        # Run MCTS
        action, info = self._mcts()
        return action, info

    def _update(self, state, reward, value, action, done, next_state, next_reward, num_episode):
        raise NotImplementedError

    def _initialize_game(self, state):
        self.available_cards = list(range(self.num_cards))
        self.num_players = self._num_players_from_state(state)

    def _path_to_state(self, state, path):
        """ Given an initial state and a path (list of actions), computes the resulting environment state """
        raise NotImplementedError

    def _find_legal_actions(self, state):
        """ Given a path (list of actions), finds all allowed actions """
        raise NotImplementedError

    def _compute_reward(self, path):
        """ Given a path (list of actions), compute the reward """
        raise NotImplementedError

    def _check_if_path_terminates(self, initial_state, path):
        """ Given an initial state and a path (list of actions), check if the final state is terminal """
        raise NotImplementedError

    def _mcts(self, state, n):
        """ Run Monte-Carl tree search from state for n trajectories"""

        # Set up MCTS graph
        # TODO: retain information between successive mergings
        head = MCTSNode(None, [])
        head.set_terminal(self._check_if_path_is_terminal(state, head.path))

        # Run n trajectories using PUCT based on policy
        for _ in range(n):
            node = head
            node.set_terminal(self._check_if_path_is_terminal(self.state, node.path))

            while not node.terminal:
                this_state = self._path_to_state(state, node.path)

                # Expand
                if not node.children:
                    actions = self._find_legal_actions(this_state)
                    node.expand(actions)

                # Select
                policy_probs = self._evaluate_policy(this_state, node.children.keys())
                node = node.select_puct(policy_probs, self.c_puct)
                if node.terminal is None:
                    node.set_terminal(self._check_if_path_is_terminal(state, node.path))

            # Backpropagate
            reward = self._compute_reward(node.path)
            node.give_reward(reward, backpropagate=True)

        # Select best action
        action = head.select_best()
        info = {}

        return action, info


    # def _mcts(self, legal_actions, state):
    #     n = len(legal_actions)
    #     n_mc = self._compute_n_mc(n)
    #     outcomes = {action : [] for action in legal_actions}
    #     log_probs = {action : [] for action in legal_actions}
    #
    #     for _ in range(n_mc):
    #         env = self._draw_env(legal_actions, state)
    #         action, log_prob, outcome = self._play_out(env, outcomes)
    #         outcomes[action].append(outcome)
    #         log_probs[action].append(log_prob)
    #
    #     action, info = self._choose_action_from_outcomes(outcomes, log_probs)
    #     return action, info
    #
    # def _play_out(self, env, outcomes):
    #     states, all_legal_actions = env._create_states()
    #     states = self._tensorize(states)
    #     done = False
    #     outcome = 0.
    #     initial_action = None
    #     initial_log_prob = None
    #
    #     while not done:
    #         actions, agent_infos = [], []
    #         for i, (state, legal_actions) in enumerate(zip(states, all_legal_actions)):
    #             action, log_prob = self._choose_action_mc(legal_actions, state, outcomes, first_move=(initial_action is None), opponent=(i > 0))
    #             actions.append(int(action))
    #
    #             if initial_action is None:
    #                 initial_action = action
    #                 initial_log_prob = log_prob
    #
    #         (next_states, next_all_legal_actions), next_rewards, done, _ = env.step(actions)
    #         next_states = self._tensorize(next_states)
    #
    #         outcome += next_rewards[0]
    #         states = next_states
    #         all_legal_actions = next_all_legal_actions
    #
    #     return initial_action, initial_log_prob, outcome
    #
    # def _choose_action_from_outcomes(self, outcomes, log_probs):
    #     best_action = list(outcomes.keys())[0]
    #     best_mean = - float("inf")
    #
    #     for action, outcome in outcomes.items():
    #         if np.mean(outcome) > best_mean:
    #             best_action = action
    #             best_mean = np.mean(outcome)
    #
    #     info = {"log_prob": log_probs[best_action][0]}
    #
    #     logger.debug("AlphaAlmostZero thoughts:")
    #     for action, outcome in outcomes.items():
    #         chosen = 'x' if action == best_action else ' '
    #         logger.debug(f"  {chosen} {action + 1:>3d}: p = {np.exp(log_probs[action][0].detach().numpy()):.2f}, n = {len(outcome):>3d}, E[r] = {np.mean(outcome):>5.1f}")
    #
    #     return best_action, info
    #
    # def _choose_action_mc(self, legal_actions, state, outcomes, first_move=True, opponent=False):
    #     raise NotImplementedError
