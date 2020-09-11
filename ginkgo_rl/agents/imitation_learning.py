import random
from collections import deque
import torch
from tqdm import trange

from .mcts import PolicyMCTSAgent


class ImitationLearningPolicyMCTSAgent(PolicyMCTSAgent):
    def learn(self, total_timesteps, callback=None, mode="demonstration"):
        # RL training
        assert mode in ["demonstration", "rl"]
        if mode == "rl":
            return super().learn(total_timesteps, callback)

        # Prepare training
        self.train()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay**(1. / (total_timesteps + 1.0e-9)))

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
            _, agent_info = self.predict(state)
            log_probs = agent_info["log_probs"]
            loss = - log_probs[demonstration_action]

            # Transition to next step
            next_state, next_reward, done, env_info = self.env.step(demonstration_action)

            _ = self.update(
                mode="imitation",
                state=self._tensorize(state),
                reward=reward,
                action=demonstration_action,
                done=done,
                next_state=self._tensorize(next_state),
                next_reward=next_reward,
                num_episode=episode,
                train=False,
                memorize=False,
                **agent_info
            )

            episode_loss += loss
            episode_reward += next_reward
            episode_length += 1
            rewards.append(next_reward)
            state = next_state
            reward = next_reward

            if done or not demonstration_actions:
                callback(callback_info={"episode": episode, "episode_length": episode_length, "loss": episode_loss, "reward": episode_reward})

                episode += 1
                episode_loss = 0.0
                episode_reward = 0.0
                episode_length = 0
                state = self.env.reset()
                demonstration_actions = deque(self._find_demonstration_actions())

    def _find_demonstration_actions(self, source="truth"):
        """ From self.env.jet, find the sequence of true actions... or the MLE sequence of actions """

        assert source == "truth"  # TODO: implement MLE demonstrations

        original_momenta = self.env.jet['content']  # original_momenta[i] are unmodified four-momenta of particle with ID i
        original_children = self.env.jet['tree']  # original_children[i] are original IDs of children of particle with ID i
        original_parents = {tuple(sorted([i, j])): parent for parent, (i, j) in original_children.items()}  # original_parent[(i, j)] is parent ID of children IDs (i, j)

        particles = []  # list of tuples (original ID, four-momenta) of all particles at current state, sorted by the energy
        actions = []  # List of true actions

        # Find leaves
        for key, (p, children) in enumerate(zip(original_momenta, original_children)):
            if children == (-1, -1):
                particles.append((key, p))

        while len(particles) > 1:
            # Sort particles by energy
            particles = sorted(particles, reverse=True, key=lambda x : x[1][0])
            particle_dict = {key : sorted_id for sorted_id, (key, _) in enumerate(particles)}  # keys are particle ID, values are position in energy-sorted list

            # Find candidate actions
            candidates = []
            for (i, j) in original_parents.keys():
                if i in particle_dict and j in particle_dict:
                    candidates.append((i, j))

            # Pick one randomly
            i, j = random.sample(candidates)

            # Compute action
            action = self.env.wrap_action((particle_dict[i], particle_dict[j]))
            actions.append(action)

            # Update particle list
            del particles[j]  # Do j > i first to avoid index changing
            del particles[i]
            parent_ij = original_parents[(i,j)]
            particles.append((parent_ij, original_momenta[parent_ij]))

        return actions
