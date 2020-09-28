import torch
from torch import nn
import copy
import logging
import numpy as np

from ginkgo_rl.utils.mcts import MCTSNode
from .base import Agent
from ..utils.nets import MultiHeadedMLP
from ..utils.various import check_for_nans, NanException

logger = logging.getLogger(__name__)


class BaseMCTSAgent(Agent):
    def __init__(
        self,
        *args,
        n_mc_target=5,
        n_mc_min=5,
        n_mc_max=100,
        planning_mode="mean",
        decision_mode="max_reward",
        c_puct=1.0,
        reward_range=(-200.0, 0.0),
        initialize_with_beam_search=True,
        beam_size=10,
        verbose=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.n_mc_target = n_mc_target
        self.n_mc_min = n_mc_min
        self.n_mc_max = n_mc_max
        self.planning_mode = planning_mode
        self.decision_mode = decision_mode
        self.c_puct = c_puct
        self.initialize_with_beam_search = initialize_with_beam_search
        self.beam_size = beam_size

        self.reward_range = reward_range
        self.verbose = verbose

        self.sim_env = copy.deepcopy(self.env)
        self.sim_env.reset_at_episode_end = False  # Avoids expensive re-sampling of jets every time we parse a path

        self.episode_reward = 0.0
        self.episode_likelihood_evaluations = 0
        self._init_episode()

    def set_env(self, env):
        """ Sets current environment (and initializes episode) """

        self.env = env
        self.sim_env = copy.deepcopy(self.env)
        self.sim_env.reset_at_episode_end = False  # Avoids expensive re-sampling of jets every time we parse a path
        self._init_episode()

    def set_precision(self, n_mc_target, n_mc_min, n_mc_max, planning_mode, c_puct, beam_size):
        """ Sets / changes MCTS precision parameters """

        self.n_mc_target = n_mc_target
        self.n_mc_min = n_mc_min
        self.n_mc_max = n_mc_max
        self.n_mc_max = n_mc_max
        self.planning_mode = planning_mode
        self.c_puct = c_puct
        self.beam_size = beam_size

    def update(self, state, reward, action, done, next_state, next_reward, num_episode, **kwargs):
        """ Updates after environment reaction """

        # Keep track of total reward
        self.episode_reward += next_reward
        if self.verbose > 0:
            logger.debug(
                f"Agent acknowledges receiving a reward of {next_reward}, episode reward so far {self.episode_reward}"
            )

        # MCTS updates
        if done:
            # Reset MCTS when done with an episode
            self._init_episode()
        else:
            # Update MCTS tree when deciding on an action
            self.mcts_head = self.mcts_head.children[action]
            self.mcts_head.prune()  # This updates the node.path

        # Train
        if self.training:
            return self._train(kwargs["log_prob"])
        else:
            return 0.0

    def _init_replay_buffer(self, history_length):
        # No need for a history in this one!
        pass

    def _predict(self, state):
        if self.initialize_with_beam_search:
            self._beam_search(state)
        action, info = self._mcts(state)
        info["likelihood_evaluations"] = self.episode_likelihood_evaluations
        return action, info

    def _parse_path(self, state, path, from_which_env="real"):
        """ Given a path (list of actions), computes the resulting environment state and total reward.

        `from_which_env` defines the start point (either "sim" for self.sim_env, or "real" for self.env) """

        if from_which_env == "real":  # Start in self.env state
            if self.sim_env.state is None or not np.all(np.isclose(self.sim_env.state, self.env.state)):
                self.sim_env.set_internal_state(self.env.get_internal_state())
        elif from_which_env == "sim":  # Use current state of self.sim_env
            pass
        else:
            raise ValueError(from_which_env)

        self.sim_env.verbose = False

        # Follow path
        total_reward = 0.0
        terminal = False

        for action in path:
            state, reward, done, info = self.sim_env.step(action)
            total_reward += reward
            self.episode_likelihood_evaluations += 1

            if done:
                terminal = True
                break

        state = self._tensorize(state)
        return state, total_reward, terminal

    def _parse_action(self, action, from_which_env="sim"):
        """ Given a state and an action, computes the log likelihood """

        if from_which_env == "real":  # Start in self.env state
            if self.sim_env.state is None or not np.all(np.isclose(self.sim_env.state, self.env.state)):
                self.sim_env.set_internal_state(self.env.get_internal_state())
        elif from_which_env == "sim":  # Use current state of self.sim_env
            pass
        else:
            raise ValueError(from_which_env)

        self.sim_env.verbose = False

        try:
            _, _ = action
            log_likelihood = self.sim_env._compute_log_likelihood(action)
        except TypeError:
            log_likelihood = self.sim_env._compute_log_likelihood(self.sim_env.unwrap_action(action))

        self.episode_likelihood_evaluations += 1
        return log_likelihood

    def _init_episode(self):
        """ Initializes MCTS tree and total reward so far """

        self.mcts_head = MCTSNode(None, [], reward_min=self.reward_range[0], reward_max=self.reward_range[1])
        self.episode_reward = 0.0
        self.episode_likelihood_evaluations = 0

    def _mcts(self, state, max_steps=1000):
        """ Run Monte-Carl tree search from state for n trajectories"""

        n_initial_legal_actions = len(self._find_legal_actions(state))
        n = min(max(self.n_mc_target * n_initial_legal_actions - self.mcts_head.n, self.n_mc_min), self.n_mc_max)
        logger.debug(f"Starting MCTS with {n} trajectories")

        for i in range(n):
            if self.verbose > 1:
                logger.debug(f"Initializing MCTS trajectory {i+1} / {n}")
            node = self.mcts_head
            total_reward = 0.0

            for _ in range(max_steps):
                # Parse current state
                if len(node.path) == 0:
                    this_state, total_reward, terminal = self._parse_path(state, node.path)
                else:  # We can speed this up by just doing a single step in self.sim_env
                    this_state, last_step_reward, terminal = self._parse_path(
                        this_state, node.path[-1:], from_which_env="sim"
                    )
                    total_reward += last_step_reward

                node.set_terminal(terminal)
                if self.verbose > 1:
                    logger.debug(f"  Node {node.path}")

                # Termination
                if terminal:
                    if self.verbose > 1:
                        logger.debug(f"  Node is terminal")
                    break

                # Expand
                if not node.children:
                    actions = self._find_legal_actions(this_state)
                    if self.verbose > 1:
                        logger.debug(f"    Expanding: {len(actions)} legal actions")
                    step_rewards = [self._parse_action(action, from_which_env="sim") for action in actions]
                    node.expand(actions, step_rewards=step_rewards)

                    if not node.children:
                        logger.warning(
                            f"Did not find any legal actions even though state was not recognized as terminal. "
                            f"Node path: {node.path}. Children: {node.children}. State: {this_state}. Actions: {actions}."
                        )
                        node.set_terminal(True)
                        break

                # Select
                policy_probs = self._evaluate_policy(
                    this_state, node.children.keys(), step_rewards=node.children_q_steps()
                )
                action = node.select_puct(policy_probs, mode=self.planning_mode, c_puct=self.c_puct)
                if self.verbose > 1:
                    logger.debug(f"    Selecting action {action}")
                node = node.children[action]

            # Backup
            if self.verbose > 1:
                logger.debug(f"  Backing up total reward of {total_reward}")
            node.give_reward(self.episode_reward + total_reward, backup=True)

        # Select best action
        legal_actions = list(self.mcts_head.children.keys())
        if not legal_actions:
            legal_actions = self._find_legal_actions(state)
        step_rewards = self.mcts_head.children_q_steps()

        if self.decision_mode == "max_reward":
            action = self.mcts_head.select_best(mode="max")
        elif self.decision_mode == "max_puct":
            policy_probs = self._evaluate_policy(state, legal_actions, step_rewards=step_rewards)
            action = self.mcts_head.select_puct(policy_probs=policy_probs, mode="max", c_puct=self.c_puct)
        elif self.decision_mode == "mean_puct":
            policy_probs = self._evaluate_policy(state, legal_actions, step_rewards=step_rewards)
            action = self.mcts_head.select_puct(policy_probs=policy_probs, mode="max", c_puct=self.c_puct)
        else:
            raise ValueError(self.decision_mode)

        log_prob = torch.log(self._evaluate_policy(state, legal_actions, step_rewards=step_rewards, action=action))
        info = {"log_prob": log_prob}

        # Debug output
        if self.verbose > 0:
            self._report_decision(action, state)

        return action, info

    def _greedy(self, state):
        """ Expands MCTS tree using a greedy algorithm """

        node = self.mcts_head
        if self.verbose > 1:
            logger.debug(f"Starting greedy algorithm.")

        while not node.terminal:
            # Parse current state
            this_state, total_reward, terminal = self._parse_path(state, node.path)
            node.set_terminal(terminal)
            if self.verbose > 1:
                logger.debug(f"  Analyzing node {node.path}")

            # Expand
            if not node.terminal and not node.children:
                actions = self._find_legal_actions(this_state)
                step_rewards = [self._parse_action(action, from_which_env="sim") for action in actions]
                if self.verbose > 1:
                    logger.debug(f"    Expanding: {len(actions)} legal actions")
                node.expand(actions, step_rewards=step_rewards)

            # If terminal, backup reward
            if node.terminal:
                if self.verbose > 1:
                    logger.debug(f"    Node is terminal")
                if self.verbose > 1:
                    logger.debug(f"    Backing up total reward {total_reward}")
                node.give_reward(self.episode_reward + total_reward, backup=True)

            # Debugging -- this should not happen
            if not node.terminal and not node.children:
                logger.warning(
                    f"Unexpected lack of children! Path: {node.path}, children: {node.children.keys()}, legal actions: {self._find_legal_actions(this_state)}, terminal: {node.terminal}"
                )
                node.set_terminal(True)

            # Greedily select next action
            if not node.terminal:
                action = node.select_greedy()
                node = node.children[action]

        if self.verbose > 0:
            choice = self.mcts_head.select_best(mode="max")
            self._report_decision(choice, state, "Greedy")

    def _beam_search(self, state):
        """ Expands MCTS tree using beam search """

        beam = [(self.episode_reward, self.mcts_head)]
        next_beam = []

        def format_beam():
            return [node.path for _, node in beam]

        if self.verbose > 1:
            logger.debug(f"Starting beam search with beam size {self.beam_size}. Initial beam: {format_beam()}")

        while beam or next_beam:
            for i, (_, node) in enumerate(beam):
                # Parse current state
                this_state, total_reward, terminal = self._parse_path(state, node.path)
                node.set_terminal(terminal)
                if self.verbose > 1:
                    logger.debug(f"  Analyzing node {i+1} / {len(beam)} on beam: {node.path}")

                # Expand
                if not node.terminal and not node.children:
                    actions = self._find_legal_actions(this_state)
                    step_rewards = [self._parse_action(action, from_which_env="sim") for action in actions]
                    if self.verbose > 1:
                        logger.debug(f"    Expanding: {len(actions)} legal actions")
                    node.expand(actions, step_rewards=step_rewards)

                # If terminal, backup reward
                if node.terminal:
                    if self.verbose > 1:
                        logger.debug(f"    Node is terminal")
                    if self.verbose > 1:
                        logger.debug(f"    Backing up total reward {total_reward}")
                    node.give_reward(self.episode_reward + total_reward, backup=True)

                # Did we already process this one? Then skip it
                if node.n_beamsearch >= self.beam_size:
                    if self.verbose > 1:
                        logger.debug(f"    Already beam searched this node sufficiently")
                    continue

                # Beam search selection
                for action in node.select_beam_search(self.beam_size):
                    next_reward = total_reward + node.children[action].q_step
                    next_node = node.children[action]
                    next_beam.append((next_reward, next_node))

                # Mark as visited
                node.in_beam = True

            # Just keep top entries for next step
            beam = sorted(next_beam, key=lambda x: x[0], reverse=True)[: self.beam_size]
            if self.verbose > 1:
                logger.debug(
                    f"Preparing next step, keeping {self.beam_size} / {len(next_beam)} nodes in beam: {format_beam()}"
                )
            next_beam = []

        logger.debug(f"Finished beam search")

        if self.verbose > 0:
            choice = self.mcts_head.select_best(mode="max")
            self._report_decision(choice, state, "Beam search")

    def _report_decision(self, chosen_action, state, label="MCTS"):
        legal_actions = self._find_legal_actions(state)
        probs = self._evaluate_policy(state, legal_actions)

        logger.debug(f"{label} results:")
        for i, (action_, node_) in enumerate(self.mcts_head.children.items()):
            is_chosen = "*" if action_ == chosen_action else " "
            is_greedy = "g" if action_ == np.argmax(self.mcts_head.children_q_steps()) else " "
            logger.debug(
                f" {is_chosen}{is_greedy} {action_:>2d}: "
                f"log likelihood = {node_.q_step:6.2f}, "
                f"policy = {probs[i].detach().item():.2f}, "
                f"n = {node_.n:>2d}, "
                f"mean = {node_.q / (node_.n + 1.e-9):>5.1f} [{node_.get_reward():>4.2f}], "
                f"max = {node_.q_max:>5.1f} [{node_.get_reward(mode='max'):>4.2f}]"
            )

    def _evaluate_policy(self, state, legal_actions, step_rewards=None, action=None):
        """ Evaluates the policy on the state and returns the probabilities for a given action or all legal actions """
        raise NotImplementedError

    def _train(self, log_prob):
        """ Policy updates at end of each step, returns loss """
        raise NotImplementedError


class PolicyMCTSAgent(BaseMCTSAgent):
    def __init__(
        self,
        *args,
        log_likelihood_feature=True,
        hidden_sizes=(100, 100,),
        activation=nn.ReLU(),
        action_factor=0.01,
        log_likelihood_factor=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.log_likelihood_feature = log_likelihood_feature

        self.actor = MultiHeadedMLP(
            1 + 8 + int(self.log_likelihood_feature) + self.state_length,
            hidden_sizes=hidden_sizes,
            head_sizes=(1,),
            activation=activation,
            head_activations=(None,),
        )
        self.softmax = nn.Softmax(dim=0)

        self.action_factor = action_factor
        self.log_likelihood_factor = log_likelihood_factor

    def _evaluate_policy(self, state, legal_actions, step_rewards=None, action=None):
        try:
            policy_input = self._prepare_policy_input(state, legal_actions, step_rewards=step_rewards)
            check_for_nans("Policy input", policy_input)
            (probs,) = self.actor(policy_input)
            check_for_nans("Policy probabilities", probs)
            probs = self.softmax(probs).flatten()
        except NanException:
            logger.error("NaNs appeared when evaluating the policy.")
            logger.error(f"  state:             {state}")
            logger.error(f"  legal actions:     {legal_actions}")
            logger.error(f"  step rewards:      {step_rewards}")
            logger.error(f"  action:            {action}")
            logger.error(f"  policy weights:    {list(self.parameters())}")
            logger.error(f"  mean weight:       {self.get_mean_weight()}")

            raise

        if action is not None:
            assert action in legal_actions
            return probs[legal_actions.index(action)]

        return probs

    def _prepare_policy_input(self, state, legal_actions, step_rewards=None):
        """ Prepares the input to the policy """
        check_for_nans("Raw state", state)
        state_ = state.view(-1)

        if step_rewards is None or not step_rewards:
            step_rewards = [None for _ in legal_actions]
        batch_states = []

        assert legal_actions
        assert step_rewards
        assert len(legal_actions) == len(step_rewards)

        for action, log_likelihood in zip(legal_actions, step_rewards):
            action_ = self.action_factor * torch.tensor([action]).to(self.device, self.dtype)

            i, j = self.env.unwrap_action(action)
            pi = state[i, :]
            pj = state[j, :]
            check_for_nans("Individual momenta", pi, pj)

            if self.log_likelihood_feature:
                if log_likelihood is None:
                    log_likelihood = self._parse_action(action, from_which_env="real")
                if not np.isfinite(log_likelihood):
                    log_likelihood = 0.0
                log_likelihood = np.clip(log_likelihood, self.reward_range[0], self.reward_range[1])
                log_likelihood_ = self.log_likelihood_factor * torch.tensor([log_likelihood]).to(
                    self.device, self.dtype
                )
                check_for_nans("Log likelihood as policy input", log_likelihood_)

                combined_state = torch.cat((action_, pi, pj, log_likelihood_, state_), dim=0)
                check_for_nans("Individual policy input entry", combined_state)
            else:
                combined_state = torch.cat((action_, pi, pj, state_), dim=0)
                check_for_nans("Individual policy input entry", combined_state)

            batch_states.append(combined_state.unsqueeze(0))

        batch_states = torch.cat(batch_states, dim=0)
        check_for_nans("Concatenated policy input", batch_states)
        return batch_states

    def _train(self, log_prob):
        loss = -log_prob
        check_for_nans("Loss", loss)
        self._gradient_step(loss)
        return loss.item()

    def get_mean_weight(self):
        parameters = np.concatenate([param.detach().numpy().flatten() for _, param in self.named_parameters()], 0)
        return np.mean(np.abs(parameters))


class RandomMCTSAgent(BaseMCTSAgent):
    def _evaluate_policy(self, state, legal_actions, step_rewards=None, action=None):
        """ Evaluates the policy on the state and returns the probabilities for a given action or all legal actions """
        if action is not None:
            return torch.tensor(1.0 / len(legal_actions), dtype=self.dtype)
        else:
            return 1.0 / len(legal_actions) * torch.ones(len(legal_actions), dtype=self.dtype)

    def _train(self, log_prob):
        return torch.tensor(0.0)


class LikelihoodMCTSAgent(BaseMCTSAgent):
    def _evaluate_policy(self, state, legal_actions, step_rewards=None, action=None):
        """ Evaluates the policy on the state and returns the probabilities for a given action or all legal actions """
        assert step_rewards is not None
        probabilities = torch.exp(torch.tensor(step_rewards, dtype=self.dtype))
        probabilities = probabilities / torch.sum(probabilities)

        if action is not None:
            return probabilities[action]
        else:
            return probabilities

    def _train(self, log_prob):
        return torch.tensor(0.0)
