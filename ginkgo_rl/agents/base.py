import numpy as np
import torch
from torch import nn
import logging
from tqdm import trange

from ..utils.replay_buffer import History

logger = logging.getLogger(__name__)


class Agent(nn.Module):
    """ Abstract base agent class """

    def __init__(self, env, gamma=1.00, optim_kwargs=None, history_length=None, dtype=torch.float, device=torch.device("cpu"), *args, **kwargs):
        self.env = env
        self.gamma = gamma
        self.device = device
        self.dtype = dtype
        self.action_space = env.action_space
        self.state_shape = env.observation_space.shape
        self.state_length = np.product(self.state_shape)
        self.num_actions = self.action_space.n
        self._init_replay_buffer(history_length)
        self.optimizer = None
        self.optim_kwargs = optim_kwargs

        super().__init__()

    def set_env(self, env):
        self.env = env

    def learn(self, total_timesteps):
        # Prepare training
        self.train()
        if list(self.parameters()):
            optim_kwargs = {} if self.optim_kwargs is None else self.optim_kwargs
            self.optimizer = torch.optim.Adam(params=self.parameters(), **optim_kwargs)
        else:
            self.optimizer = None  # For non-NN methods

        state = self.env.reset()
        reward = 0.
        rewards = []
        episode = 0

        for steps in trange(total_timesteps):
            action, agent_info = self.predict(state)
            next_state, next_reward, done, info = self.env.step(action)

            self.update(
                state=self._tensorize(state),
                reward=reward,
                action=action,
                done=done,
                next_state=self._tensorize(next_state),
                next_reward=next_reward,
                num_episode=episode,
                **agent_info
            )

            rewards.append(next_reward)
            state = next_state
            reward = next_reward

            if done:
                episode += 1
                state = self.env.reset()

    def predict(self, state):
        """
        Given an environment state, pick the next action and return it.

        Parameters
        ----------
        state : ndarray
            Observed state s_t.

        Returns
        -------
        action : int
            Chosen action a_t.

        agent_info : dict
            Additional stuffs.

        """

        state = self._tensorize(state)
        return self._predict(state)

    def update(self, state, reward, action, done, next_state, next_reward, num_episode, **kwargs):
        """
        Is called at the end of each step, gives the agent the chance to a) update the replay buffer and b) learn its weights.
        """
        raise NotImplementedError

    def _init_replay_buffer(self, history_length):
        self.history = History(max_length=history_length, dtype=self.dtype, device=self.device)

    def _tensorize(self, array):
        tensor = array if isinstance(array, torch.Tensor) else torch.tensor(array)
        tensor = tensor.to(self.device, self.dtype)
        return tensor

    def _gradient_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _find_legal_actions(self, state):
        # Compatibility with torch tensors and numpy arrays
        try:
            state = state.numpy()
        except:
            pass

        particles = [i for i, p in enumerate(state) if np.max(p) > 0]

        actions = []
        try:  # 1D-wrapped envs
            for i, pi in enumerate(particles):
                for j, pj in enumerate(particles[:i]):
                    actions.append(self.env.wrap_action((pi, pj)))
        except:
            for i, pi in enumerate(particles):
                for j, pj in enumerate(particles[:i]):
                    actions.append((pi, pj))

        return actions
