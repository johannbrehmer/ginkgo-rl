import random
from collections import deque
import torch
from tqdm import trange
import sys
from torch.distributions import Categorical
import numpy as np
import logging

from .mcts import PolicyMCTSAgent

# Workaround for now until Trellis is better packaged
sys.path.append("/Users/johannbrehmer/work/projects/shower_rl/hierarchical-trellis/src")
from run_physics_experiment_invM import compare_map_gt_and_bs_trees as compute_trellis

logger = logging.getLogger(__name__)


class ImitationLearningPolicyMCTSAgent(PolicyMCTSAgent):
    def predict(self, state, mode="mcts"):
        if mode == "mcts":
            return super().predict(state)
        elif mode == "policy":
            return self._predict_policy(state)
        else:
            raise ValueError(mode)

    def _predict_policy(self, state):
        state = self._tensorize(state)
        legal_actions = self._find_legal_actions(state)
        step_rewards = [self._parse_action(action, from_which_env="real") for action in legal_actions]

        probs = self._evaluate_policy(state, legal_actions, step_rewards)
        cat = Categorical(torch.clamp(probs, 1.e-12, 1.0))
        action_id = cat.sample()

        action = legal_actions[action_id]
        info = {"log_probs": torch.log(probs)}

        return action, info

    def learn(self, total_timesteps, callback=None, mode="imitation"):
        # RL training
        assert mode in ["imitation", "rl"]
        if mode == "rl":
            return super().learn(total_timesteps, callback)

        # Prepare training
        self.train()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay**(1. / (total_timesteps + 1.0e-9)))

        demonstration_actions = None
        while demonstration_actions is None:
            state = self.env.reset()
            demonstration_actions = deque(self._find_demonstration_actions())

        reward = 0.
        rewards = []
        episode = 0
        episode_loss = 0.0
        episode_reward = 0.0
        episode_length = 0

        for _ in trange(total_timesteps):
            # Imitation learning
            demonstration_action = demonstration_actions.popleft()
            _, agent_info = self.predict(state, mode="policy")
            log_probs = agent_info["log_probs"]
            loss = - log_probs[demonstration_action]

            # Transition to next step
            next_state, next_reward, done, env_info = self.env.step(demonstration_action)

            episode_loss += loss
            episode_reward += next_reward
            episode_length += 1
            rewards.append(next_reward)
            state = next_state
            reward = next_reward

            if done or not demonstration_actions:
                if callback is not None:
                    callback(callback_info={"episode": episode, "episode_length": episode_length, "loss": episode_loss, "reward": episode_reward, "likelihood_evaluations": agent_info["likelihood_evaluations"]})

                episode += 1
                episode_loss = 0.0
                episode_reward = 0.0
                episode_length = 0

                demonstration_actions = None
                while demonstration_actions is None:
                    state = self.env.reset()
                    demonstration_actions = deque(self._find_demonstration_actions())

    def _find_demonstration_actions(self, source="mle"):
        """ From self.env.jet, find the sequence of true actions... or the MLE sequence of actions """

        logger.debug("Beginning extraction of demonstrator actions.")

        if source == "truth":
            jet = self.env.jet
        elif source == "mle":
            jet = self._get_maximum_likelihood_tree()
            if jet is None:
                return None
        else:
            raise ValueError(source)

        original_momenta = jet['content']  # original_momenta[i] are unmodified four-momenta of particle with ID i
        original_children = jet['tree']  # original_children[i] are original IDs of children of particle with ID i
        original_parents = {tuple(sorted([i, j])): parent for parent, (i, j) in enumerate(original_children) if i >= 0 and j >= 0}  # original_parent[(i, j)] is parent ID of children IDs (i, j)

        logger.debug("Children list:")
        for parent, (i, j) in enumerate(original_children):
            logger.debug(f"  Parent {parent} -> children {i}, {j}")

        logger.debug("Parents list:")
        for (i, j), parent in original_parents.items():
            logger.debug(f"  Children {i}, {j} -> parent {parent}")

        particles = []  # list of tuples (original ID, four-momenta) of all particles at current state, sorted by the energy
        actions = []  # List of true actions

        # Find leaves
        for key, (p, children) in enumerate(zip(original_momenta, original_children)):
            if np.all(np.isclose(children, -1.0)):
                particles.append((key, p))

        while len(particles) > 1:
            # Sort particles by energy
            particles = sorted(particles, reverse=True, key=lambda x : x[1][0])
            particle_dict = {key : sorted_id for sorted_id, (key, _) in enumerate(particles)}  # keys are particle ID, values are position in energy-sorted list

            logger.debug("Considering next clustering step.")
            logger.debug("  Particle dictionary:")
            for key, val in particle_dict.items():
                logger.debug(f"    ID {key} -> energy-ranked position {val}")

            # Find candidate actions
            candidates = []
            for (i, j) in original_parents.keys():
                if i in particle_dict and j in particle_dict:
                    candidates.append((i, j))

            logger.debug("  Action candidates:")
            for (i, j) in candidates:
                logger.debug(f"    {i}, {j}")

            if not candidates:
                raise RuntimeError("Did not find any feasible splittings!")

            # Pick one randomly
            i, j = random.sample(candidates, k=1)[0]

            # Compute action
            action = self.env.wrap_action((particle_dict[i], particle_dict[j]))
            actions.append(action)

            logger.debug(f"  Chose to merge {i} and {j} (action {action})")

            # Update particle list
            for pos, (idx, _) in enumerate(particles):
                if idx == i:
                    del particles[pos]
                    break
            for pos, (idx, _) in enumerate(particles):
                if idx == j:
                    del particles[pos]
                    break
            parent_ij = original_parents[(i,j)]
            logger.debug(f"  Removing particles {i} and {j} from list and adding {parent_ij}")
            particles.append((parent_ij, original_momenta[parent_ij]))

        return actions

    def _get_maximum_likelihood_tree(self, max_leaves=11):
        """ Based on Sebastian's code at https://github.com/iesl/hierarchical-trellis/blob/sebastian/src/Jet_Experiments_invM_exactTrellis.ipynb """

        if len(self.env.jet["leaves"]) > max_leaves:
            return None

        trellis, _, _, _, _ = compute_trellis(self.env.jet)
        jet = trellis.traverseMAPtree(trellis.root)

        return jet
