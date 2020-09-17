import torch
from torch import nn
import numpy as np
import logging
from torch.distributions import Categorical

from .base import Agent
from ..utils.various import iter_flatten
from ..utils.replay_buffer import SequentialHistory
from ..utils.nets import MultiHeadedMLP

logger = logging.getLogger(__name__)


class BatchedActorCriticAgent(Agent):
    """
    Simple actor-critic agent for discrete action spaces.

    The actor implements the policy pi(a|s). The critic estimates the state-action value q(s, a). The base A-C class does not yet implement any training algorithm.
    """

    def __init__(
        self,
        *args,
        log_likelihood_feature=True, hidden_sizes=(100, 100,), activation=nn.ReLU(),
        log_epsilon=-20.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.log_epsilon = log_epsilon
        self.log_likelihood_feature = log_likelihood_feature

        self.actor_critic = MultiHeadedMLP(1 + self.state_length, hidden_sizes=hidden_sizes, head_sizes=(1, 1), activation=activation, head_activations=(None, None),)
        self.softmax = nn.Softmax(dim=0)

    def _predict(self, state):
        legal_actions = self._find_legal_actions(state)
        batch_states = self._batch_state(state, legal_actions)
        log_probs, qs = self._evaluate_batch_states(batch_states)
        action_id = self._act(log_probs, legal_actions)

        return (
            legal_actions[action_id],
            {"legal_actions": legal_actions, "action_id": action_id, "log_probs": log_probs, "log_prob": log_probs[action_id], "values": qs, "value": qs[action_id], "likelihood_evaluations": 0},
        )

    def _evaluate(self, states, legal_actions_list):
        all_qs = []
        all_log_probs = []

        for state, legal_actions in zip(states, legal_actions_list):
            if not legal_actions:
                logger.warning(f"No legal actions! {legal_actions_list}")
                continue

            batch_states = self._batch_state(state, legal_actions)
            print(batch_states)
            log_probs, qs = self._evaluate_batch_states(batch_states)
            print(log_probs)
            all_qs.append(qs.flatten().unsqueeze(0))
            all_log_probs.append(log_probs.unsqueeze(0))

        all_qs = torch.cat(all_qs, dim=0)
        all_log_probs = torch.cat(all_log_probs, dim=0)

        return all_log_probs, all_qs

    def update(self, state, reward, action, done, next_state, next_reward, num_episode, **kwargs):
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
        probs = torch.exp(torch.clamp(log_probs, -20, 0.))
        cat = Categorical(probs)
        action_id = len(legal_actions)
        while action_id >= len(legal_actions):
            try:
                action_id = cat.sample()
            except RuntimeError:  # Sometimes something weird happens here...
                logger.error(f"Error sampling action! Log probabilities: {log_probs} -> probabilities {probs}")
                raise
        return action_id

    def _pad(self, inputs, value=None):
        return torch.nn.functional.pad(
            inputs, (0, self.num_actions - inputs.size()[-1]), mode="constant", value=self.log_epsilon if value is None else value
        )

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


class BatchedACERAgent(BatchedActorCriticAgent):
    """ Largely following https://github.com/seungeunrho/minimalRL/blob/master/acer.py """

    def __init__(self, *args, rollout_len=10, minibatch=5, truncate=1.0, warmup=100, r_factor=1.0, actor_weight=1.0, critic_weight=1.0, **kwargs):
        self.truncate = truncate
        self.warmup = warmup
        self.batchsize = minibatch
        self.rollout_len = rollout_len
        self.r_factor = r_factor
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight

        super().__init__(*args, **kwargs)

    def _init_replay_buffer(self, history_length):
        self.history = SequentialHistory(max_length=history_length, dtype=self.dtype, device=self.device)

    def update(self, state, reward, action, done, next_state, next_reward, num_episode, **kwargs):
        self.history.store(
            state=state,
            legal_actions=kwargs["legal_actions"],
            log_probs=kwargs["log_probs"],
            action_id=kwargs["action_id"],
            next_reward=next_reward * self.r_factor,
            done=done,
        )

        loss = 0.0

        if self.history.current_sequence_length() >= self.rollout_len or done:
            self.history.flush()

            if len(self.history) > max(self.warmup, self.batchsize):
                loss += self._train(on_policy=True)
                loss += self._train(on_policy=False)

        if done:
            self.episode_likelihood_evaluations = 0

        return loss

    def _train(self, on_policy=True):
        # Rollout
        action_ids, done, is_first, legal_actions_list, log_probs_then, rewards, states = self._rollout(on_policy)
        assert legal_actions_list

        log_probs_now, q = self._evaluate(states, legal_actions_list)
        q_a = q.gather(1, action_ids)
        log_prob_now_a = log_probs_now.gather(1, action_ids)
        v = (q * torch.exp(log_probs_now)).sum(1).unsqueeze(1).detach()

        rho = torch.exp(log_probs_now - log_probs_then).detach()
        rho_a = rho.gather(1, action_ids)
        rho_bar = rho_a.clamp(max=self.truncate)
        correction_coeff = (1.0 - self.truncate / rho).clamp(min=0.0)

        q_ret = self._compute_return(done, is_first, q_a, rewards, rho_bar, v)

        actor_loss = -rho_bar * log_prob_now_a * (q_ret - v)
        actor_loss = actor_loss.mean()
        correction_loss = -correction_coeff * torch.exp(log_probs_then.detach()) * log_probs_now * (q.detach() - v)  # bias correction term
        correction_loss = correction_loss.sum(1).mean()
        critic_loss = self.critic_weight * torch.nn.SmoothL1Loss()(q_a, q_ret)
        loss = actor_loss + correction_loss + critic_loss

        self._gradient_step(loss)

        return loss.item()

    def _rollout(self, on_policy):
        if on_policy:
            rollout = self.history.rollout(n=1)
        else:
            _, _, rollout = self.history.sample(self.batchsize)

        states = torch.stack(list(iter_flatten(rollout["state"])))
        legal_actions_list = list(iter_flatten(rollout["legal_actions"], max_depth=1))
        action_ids = torch.tensor(list(iter_flatten(rollout["action_id"])), dtype=torch.long).unsqueeze(1)
        rewards = np.array(list(iter_flatten(rollout["next_reward"])))
        log_probs_then = torch.stack(list(iter_flatten(rollout["log_probs"])))
        done = np.array(list(iter_flatten(rollout["done"])), dtype=np.bool)
        is_first = np.array(list(iter_flatten(rollout["first"])), dtype=np.bool)

        return action_ids, done, is_first, legal_actions_list, log_probs_then, rewards, states

    def _compute_return(self, done, is_first, q_a, rewards, rho_bar, v):
        q_ret = v[-1] * (1.0 - done[-1])
        q_ret_lst = []
        for i in reversed(range(len(rewards))):
            q_ret = rewards[i] + self.gamma * q_ret
            q_ret_lst.append(q_ret.item())
            q_ret = rho_bar[i] * (q_ret - q_a[i]) + v[i]

            if is_first[i] and i != 0:
                q_ret = v[i - 1] * (1.0 - done[i - 1])  # When a new sequence begins, q_ret is initialized
        q_ret_lst.reverse()
        q_ret = torch.tensor(q_ret_lst, dtype=torch.float).unsqueeze(1)
        return q_ret
