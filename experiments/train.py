import logging
import sys
import torch
from sacred.observers import FileStorageObserver

from .config import ex, config, env_config, mcts_config, mcts_train_config, technical_config

sys.path.append("../")
from ginkgo_rl import GinkgoLikelihood1DEnv, GinkgoLikelihoodEnv
from ginkgo_rl import MCTSAgent

logger = logging.getLogger(__name__)
__all__ = ["ex", "config", "env_config", "mcts_config", "mcts_train_config", "technical_config"]


@ex.capture
def setup_run(name, debug):
    ex.observers.append(FileStorageObserver(f'./data/runs/{name}/'))
    logging.basicConfig(
        format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
        datefmt='%H:%M',
        level=logging.DEBUG if debug else logging.INFO
    )


@ex.capture
def create_env(env_type, illegal_reward, illegal_actions_patience, n_max, n_min, n_target, min_reward,
               state_rescaling, padding_value,
w_jet,
    w_rate,
    qcd_rate,
    pt_min,
    qcd_mass,
    w_mass,
    jet_momentum,
    jetdir,
    max_n_try):

    if env_type == "1d":
        env = GinkgoLikelihood1DEnv(illegal_reward, illegal_actions_patience, n_max, n_min, n_target, min_reward, state_rescaling, padding_value, w_jet, max_n_try, w_rate, qcd_rate, pt_min, qcd_mass, w_mass, jet_momentum, jetdir)
    elif env_type == "2d":
        env = GinkgoLikelihoodEnv(illegal_reward, illegal_actions_patience, n_max, n_min, n_target, min_reward, state_rescaling, padding_value, w_jet, max_n_try, w_rate, qcd_rate, pt_min, qcd_mass, w_mass, jet_momentum, jetdir)
    else:
        raise ValueError(env_type)

    return env


@ex.capture
def create_mcts_agent(env, reward_range, optim_kwargs, history_length, n_mc_target, n_mc_min, n_mc_max, mcts_mode, c_puct, gamma, device, dtype):
    agent = MCTSAgent(env, reward_range=reward_range, optim_kwargs=optim_kwargs, history_length=history_length, n_mc_target=n_mc_target, n_mc_min=n_mc_min, n_mc_max=n_mc_max, mcts_mode=mcts_mode, c_puct=c_puct, gamma=gamma, device=device, dtype=dtype)
    return agent


@ex.capture
def train(env, agent, train_steps):
    _ = env.reset()
    agent.learn(total_timesteps=train_steps)
    env.close()


@ex.capture
def save(agent, name):
    filename = f'./data/runs/{name}/model.pty'
    torch.save(agent.state_dict(), filename)
    ex.add_artifact(filename)


@ex.automain
def main():
    setup_run()
    env = create_env()
    agent = create_mcts_agent(env=env)
    train(env, agent)
    save(agent)
