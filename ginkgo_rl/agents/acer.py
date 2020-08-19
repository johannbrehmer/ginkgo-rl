import torch
import numpy as np
import logging

from .base import BatchedActorCriticAgent
from ..utils.various import iter_flatten
from ..utils.replay_buffer import SequentialHistory

logger = logging.getLogger(__name__)


class BatchedACERAgent(BatchedActorCriticAgent):
    """ Based on https://github.com/seungeunrho/minimalRL/blob/master/acer.py """

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

    def _update(self, state, reward, value, action, done, next_state, next_reward, num_episode, *args, **kwargs):
        self.history.store(
            state=state,
            legal_actions=self._legal_action_extractor(state, self.env),
            log_probs=kwargs["log_probs"],
            action_id=kwargs["action_id"],
            next_reward=next_reward * self.r_factor,
            done=done,
        )

        if self.history.current_sequence_length() >= self.rollout_len or done:
            self.history.flush()

            if len(self.history) > max(self.warmup, self.batchsize):
                self._train(on_policy=True)
                self._train(on_policy=False)

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

        self._gradient_step(actor_loss + correction_loss + critic_loss)

        return actor_loss.item(), correction_loss.item(), critic_loss.item()

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
