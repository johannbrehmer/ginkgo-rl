import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import logging
from tqdm import trange

from ..utils.nets import MultiHeadedMLP
from ..utils.legal_actions import ginkgo1d_legal_action_extractor
from ..utils.replay_buffer import History

logger = logging.getLogger(__name__)


class Agent(nn.Module):
    """ Abstract base agent class """

    def __init__(self, env, legal_action_extractor=None, gamma=0.99, optim_kwargs=None, history_length=None, dtype=torch.float, device=torch.device("cpu"), *args, **kwargs):

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
        self._legal_action_extractor = ginkgo1d_legal_action_extractor if legal_action_extractor is None else legal_action_extractor

        super().__init__()

    def learn(self, total_timesteps):
        # Prepare training
        self.train()
        optim_kwargs = {} if self.optim_kwargs is None else self.optim_kwargs
        self.optimizer = torch.optim.Adam(params=self.parameters(), **optim_kwargs)

        state = self.env.reset()
        reward = 0.
        rewards = []
        episode = 0

        for steps in trange(total_timesteps):
            action, agent_info = self.predict(state)
            next_state, next_reward, done, info = self.env.step(action)

            self._update(
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

    def _init_replay_buffer(self, history_length):
        self.history = History(max_length=history_length, dtype=self.dtype, device=self.device)

    def _update(self, state, reward, value, action, done, next_state, next_reward, num_episode):
        """
        Is called at the end of each step, gives the agent the chance to a) update the replay buffer and b) learn its weights.
        """
        raise NotImplementedError

    def _tensorize(self, array):
        return torch.tensor(array).to(self.device, self.dtype)

    def _gradient_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class BatchedActorCriticAgent(Agent):
    """
    Simple actor-critic agent for discrete action spaces.

    The actor implements the policy pi(a|s). The critic estimates the state-action value q(s, a). The base A-C class does not yet implement any training algorithm.
    """

    def __init__(
        self,
        env,
        legal_action_extractor=None,
        gamma=0.99,
        optim_kwargs=None,
        history_length=None,
        dtype=torch.float,
        device=torch.device("cpu"),
        hidden_sizes=(100, 100,),
        activation=nn.ReLU(),
        log_epsilon=-20.0,
        *args,
        **kwargs
    ):
        super().__init__(env, legal_action_extractor, gamma, optim_kwargs, history_length, dtype, device)

        self.log_epsilon = log_epsilon
        self.actor_critic = MultiHeadedMLP(1 + self.state_length, hidden_sizes=hidden_sizes, head_sizes=(1, 1), activation=activation, head_activations=(None, None),)
        self.softmax = nn.Softmax(dim=0)

    def _predict(self, state):
        legal_actions = self._legal_action_extractor(state, self.env)
        batch_states = self._batch_state(state, legal_actions)
        log_probs, qs = self._evaluate_batch_states(batch_states)
        action_id = self._act(log_probs, legal_actions)

        return (
            legal_actions[action_id],
            {"legal_actions": legal_actions, "action_id": action_id, "log_probs": log_probs, "log_prob": log_probs[action_id], "values": qs, "value": qs[action_id]},
        )

    def _evaluate(self, states, legal_actions_list):
        all_qs = []
        all_log_probs = []

        for state, legal_actions in zip(states, legal_actions_list):
            batch_states = self._batch_state(state, legal_actions)
            log_probs, qs = self._evaluate_batch_states(batch_states)
            all_qs.append(qs.flatten().unsqueeze(0))
            all_log_probs.append(log_probs.unsqueeze(0))

        all_qs = torch.cat(all_qs, dim=0)
        all_log_probs = torch.cat(all_log_probs, dim=0)

        return all_log_probs, all_qs

    def _update(self, state, reward, value, action, done, next_state, next_reward, num_episode):
        raise NotImplementedError

    def _batch_state(self, state, legal_actions):
        state_ = state.view(-1)
        batch_states = []
        for action in legal_actions:
            action_ = torch.tensor([action]).to(self.device, self.dtype)
            batch_states.append(torch.cat((action_, state_), dim=0).unsqueeze(0))
        batch_states = torch.cat(batch_states, dim=0)
        return batch_states

    def _evaluate_batch_states(self, batch_states, pad=True):
        (probs, qs) = self.actor_critic(batch_states)
        probs = self.softmax(probs).flatten()
        qs = qs.flatten()
        log_probs = torch.log(probs)

        if pad:
            qs = self._pad(qs, value=0.0)
            log_probs = self._pad(log_probs)

        return log_probs, qs

    def _act(self, log_probs, legal_actions):
        cat = Categorical(torch.exp(torch.clamp(log_probs, -20)))
        action_id = len(legal_actions)
        while action_id >= len(legal_actions):
            try:
                action_id = cat.sample()
            except RuntimeError:  # Sometimes something weird happens here...
                logger.error("Error sampling action! Log probabilities: %s", log_probs)
        return action_id

    def _pad(self, inputs, value=None):
        return torch.nn.functional.pad(
            inputs, (0, self.num_actions - inputs.size()[-1]), mode="constant", value=self.log_epsilon if value is None else value
        )
