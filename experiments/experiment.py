#!/usr/bin/env python

import logging
import sys
import torch
import os
import numpy as np

sys.path.append("../")
from experiments.config import ex, config, env_config, agent_config, train_config, technical_config
from ginkgo_rl import GinkgoLikelihood1DEnv, GinkgoLikelihoodEnv
from ginkgo_rl import MCTSAgent
from ginkgo_rl import GinkgoEvaluator

logger = logging.getLogger(__name__)


@ex.capture
def setup_run(name, run_name, seed):
    logger.info(f"Setting up run {name}")
    os.makedirs(f"./data/runs/{run_name}/", exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)


@ex.capture
def setup_logging(debug):
    silence_list = ["matplotlib", "showerSim", "hierarchical-trellis"]

    for key in logging.Logger.manager.loggerDict:
        logging.getLogger(key).setLevel(logging.DEBUG if debug else logging.INFO)
        for check_key in silence_list:
            if check_key in key:
                logging.getLogger(key).setLevel(logging.ERROR)
                break


@ex.capture
def create_env(
    env_type,
    illegal_reward,
    illegal_actions_patience,
    n_max,
    n_min,
    n_target,
    min_reward,
    state_rescaling,
    padding_value,
    w_jet,
    w_rate,
    qcd_rate,
    pt_min,
    qcd_mass,
    w_mass,
    jet_momentum,
    jetdir,
    max_n_try
):
    logger.info(f"Creating environment")

    if env_type == "1d":
        env = GinkgoLikelihood1DEnv(
            illegal_reward,
            illegal_actions_patience,
            n_max,
            n_min,
            n_target,
            min_reward,
            state_rescaling,
            padding_value,
            w_jet,
            max_n_try,
            w_rate,
            qcd_rate,
            pt_min,
            qcd_mass,
            w_mass,
            jet_momentum,
            jetdir,
        )
    elif env_type == "2d":
        env = GinkgoLikelihoodEnv(
            illegal_reward,
            illegal_actions_patience,
            n_max,
            n_min,
            n_target,
            min_reward,
            state_rescaling,
            padding_value,
            w_jet,
            max_n_try,
            w_rate,
            qcd_rate,
            pt_min,
            qcd_mass,
            w_mass,
            jet_momentum,
            jetdir,
        )
    else:
        raise ValueError(env_type)

    return env


@ex.capture
def create_mcts_agent(
    env, reward_range, optim_kwargs, history_length, train_n_mc_target, train_n_mc_min, train_n_mc_max, train_mcts_mode, train_c_puct, device, dtype,
):
    logger.info(f"Setting up agent")
    agent = MCTSAgent(
        env,
        reward_range=reward_range,
        optim_kwargs=optim_kwargs,
        history_length=history_length,
        n_mc_target=train_n_mc_target,
        n_mc_min=train_n_mc_min,
        n_mc_max=train_n_mc_max,
        mcts_mode=train_mcts_mode,
        c_puct=train_c_puct,
        device=device,
        dtype=dtype,
    )
    return agent


@ex.capture
def train(env, agent, train_n_mc_target, train_n_mc_min, train_n_mc_max, train_mcts_mode, train_c_puct, train_steps):
    logger.info(f"Starting training for {train_steps} steps")
    agent.set_precision(train_n_mc_target, train_n_mc_min, train_n_mc_max, train_mcts_mode, train_c_puct)
    _ = env.reset()
    agent.learn(total_timesteps=train_steps)
    env.close()


@ex.capture
def eval(agent, name, env_type, eval_n_mc_target, eval_n_mc_min, eval_n_mc_max, eval_mcts_mode, eval_c_puct, eval_repeats, eval_jets, eval_filename, eval_figure_path, redraw_eval_jets):
    logger.info("Starting evaluation")
    os.makedirs(os.path.dirname(eval_filename), exist_ok=True)
    os.makedirs(eval_figure_path, exist_ok=True)
    evaluator = GinkgoEvaluator(filename=eval_filename, n_jets=eval_jets, redraw_existing_jets=redraw_eval_jets, auto_eval_truth_mle=True)

    agent.set_precision(eval_n_mc_target, eval_n_mc_min, eval_n_mc_max, eval_mcts_mode, eval_c_puct)
    env_name = "GinkgoLikelihood1D-v0" if env_type=="1d" else "GinkgoLikelihood-v0"
    evaluator.eval(name, agent, env_name=env_name, n_repeats=eval_repeats)

    evaluator.plot_log_likelihoods(filename=f"{eval_figure_path}/{name}.pdf")


@ex.capture
def save_agent(agent, run_name):
    filename = f"./data/runs/{run_name}/model.pty"
    logger.info(f"Saving model at {filename}")
    torch.save(agent.state_dict(), filename)
    ex.add_artifact(filename)


@ex.automain
def main():
    logger.info(f"Hi!")

    setup_run()
    setup_logging()
    env = create_env()
    agent = create_mcts_agent(env=env)

    train(env, agent)
    save_agent(agent)

    eval(agent)

    logger.info(f"That's all, have a nice day!")
