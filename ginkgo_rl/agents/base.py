import numpy as np
import torch
from torch import nn
import logging
from tqdm import trange

from ..utils.replay_buffer import History
from ..utils.various import check_for_nans

logger = logging.getLogger(__name__)


class Agent(nn.Module):
    """ Abstract base agent class """

    def __init__(
        self,
        env,
        gamma=1.00,
        lr=1.0e-3,
        lr_decay=0.01,
        weight_decay=0.0,
        history_length=None,
        clip_gradient=None,
        dtype=torch.float,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
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
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.clip_gradient = clip_gradient

        super().__init__()

    def set_env(self, env):
        self.env = env

    def learn(self, total_timesteps, callback=None):
        # Prepare training
        self.train()
        if list(self.parameters()):
            self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=self.lr_decay ** (1.0 / (total_timesteps + 1.0e-9))
            )
        else:
            self.optimizer = None  # For non-NN methods
            self.scheduler = None

        # Prepare episodes
        state = self.env.reset()
        reward = 0.0
        rewards = []
        episode = -1
        done = True
        episode_loss = 0.0
        episode_reward = 0.0
        episode_length = 0

        for steps in trange(total_timesteps):
            # Initialize episode
            if done:
                episode += 1
                episode_loss = 0.0
                episode_reward = 0.0
                episode_length = 0
                state = self.env.reset()
                self.init_episode()

            # Agent and environment step
            action, agent_info = self.predict(state)
            next_state, next_reward, done, env_info = self.env.step(action)

            # Learning
            loss = self.update(
                state=self._tensorize(state),
                reward=reward,
                action=action,
                done=done,
                next_state=self._tensorize(next_state),
                next_reward=next_reward,
                num_episode=episode,
                **agent_info,
            )

            # Book keeping
            episode_loss += loss
            episode_reward += next_reward
            episode_length += 1
            rewards.append(next_reward)
            state = next_state
            reward = next_reward

            if done and callback is not None:
                callback(
                    callback_info={
                        "episode": episode,
                        "episode_length": episode_length,
                        "loss": episode_loss,
                        "reward": episode_reward,
                        "likelihood_evaluations": agent_info["likelihood_evaluations"],
                        "mean_abs_weight": self.get_mean_weight(),
                    }
                )

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

    def init_episode(self):
        """ Is called at the beginning of an episode """
        pass

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
        check_for_nans(f"Tensorizing state {array}", tensor)
        return tensor

    def _gradient_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        if self.clip_gradient is not None:
            if self.verbose > 2:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_gradient)
                logger.debug(f"Gradient norm (clipping at {clip_gradient}): {grad_norm}")
            else:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_gradient)

        self.optimizer.step()
        self.scheduler.step()

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

    def get_mean_weight(self):
        return 0.0
