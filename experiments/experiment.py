#!/usr/bin/env python

import logging
import sys
import torch
import os
import numpy as np

sys.path.append("../")
from experiments.config import ex, config, env_config, agent_config, train_config, technical_config
from ginkgo_rl import GinkgoLikelihood1DEnv, GinkgoLikelihoodEnv
from ginkgo_rl import MCTSAgent, GreedyAgent, RandomAgent, MCBSAgent
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
def create_agent(
    env, algorithm, reward_range, history_length, train_n_mc_target, train_n_mc_min, train_n_mc_max, train_mcts_mode, train_c_puct, device, dtype, learning_rate, weight_decay, train_beamsize
):
    logger.info(f"Setting up {algorithm} agent ")

    if algorithm == "mcts":
        agent = MCTSAgent(
            env,
            reward_range=reward_range,
            history_length=history_length,
            n_mc_target=train_n_mc_target,
            n_mc_min=train_n_mc_min,
            n_mc_max=train_n_mc_max,
            mcts_mode=train_mcts_mode,
            c_puct=train_c_puct,
            device=device,
            dtype=dtype,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif algorithm == "mcbs":
        agent = MCBSAgent(
            env,
            reward_range=reward_range,
            history_length=history_length,
            n_mc_target=train_n_mc_target,
            n_mc_min=train_n_mc_min,
            n_mc_max=train_n_mc_max,
            mcts_mode=train_mcts_mode,
            c_puct=train_c_puct,
            device=device,
            dtype=dtype,
            lr=learning_rate,
            weight_decay=weight_decay,
            beam_size=train_beamsize
        )
    elif algorithm == "greedy":
        agent = GreedyAgent(env, device=device, dtype=dtype)
    elif algorithm == "random":
        agent = RandomAgent(env, device=device, dtype=dtype)
    elif algorithm in ["truth", "mle", "beamsearch"]:
        agent = None
    else:
        raise ValueError(algorithm)

    return agent


@ex.capture
def log_training(_run, callback_info):
    loss = callback_info.get('loss')
    reward = callback_info.get('reward')
    episode_length = callback_info.get('episode_length')

    if loss is not None:
        _run.log_scalar("training_loss", loss)
    if reward is not None:
        _run.log_scalar("training_reward", reward)
    if episode_length is not None:
        _run.log_scalar("training_episode_length", episode_length)


@ex.capture
def train(env, agent, algorithm, train_n_mc_target, train_n_mc_min, train_n_mc_max, train_mcts_mode, train_c_puct, train_steps):
    if algorithm in ["greedy", "random", "truth", "mle", "beamsearch"]:
        logger.info(f"No training necessary for algorithm {algorithm}")

    else:
        logger.info(f"Starting {algorithm} training for {train_steps} steps")

        # Initialize MCTS agent settings (this won't do anything for some other types of agent)
        try:
            agent.set_precision(train_n_mc_target, train_n_mc_min, train_n_mc_max, train_mcts_mode, train_c_puct)
        except:
            pass

        # Train
        _ = env.reset()
        agent.learn(total_timesteps=train_steps, callback=log_training)

    env.close()


@ex.capture
def eval(agent, name, algorithm, env_type, eval_n_mc_target, eval_n_mc_min, eval_n_mc_max, eval_mcts_mode, eval_c_puct, eval_repeats, eval_jets, eval_filename, eval_figure_path, redraw_eval_jets, eval_beamsize, _run):
    # Set up evaluator
    logger.info("Starting evaluation")
    os.makedirs(os.path.dirname(eval_filename), exist_ok=True)
    os.makedirs(eval_figure_path, exist_ok=True)
    evaluator = GinkgoEvaluator(filename=eval_filename, n_jets=eval_jets, redraw_existing_jets=redraw_eval_jets)

    # Evaluate
    if algorithm == "mcts":
        try:
            agent.set_precision(eval_n_mc_target, eval_n_mc_min, eval_n_mc_max, eval_mcts_mode, eval_c_puct)
        except:
            pass
        env_name = "GinkgoLikelihood1D-v0" if env_type=="1d" else "GinkgoLikelihood-v0"
        log_likelihood, errors = evaluator.eval(name, agent, env_name=env_name, n_repeats=eval_repeats)

    elif algorithm == "random":
        env_name = "GinkgoLikelihood1D-v0" if env_type=="1d" else "GinkgoLikelihood-v0"
        log_likelihood, errors = evaluator.eval(name, agent, env_name=env_name, n_repeats=eval_repeats)

    elif algorithm == "greedy":
        env_name = "GinkgoLikelihood1D-v0" if env_type=="1d" else "GinkgoLikelihood-v0"
        log_likelihood, errors = evaluator.eval(name, agent, env_name=env_name, n_repeats=1)

    elif algorithm == "beamsearch":
        log_likelihood, errors = evaluator.eval_beam_search(name, beam_size=eval_beamsize)

    elif algorithm == "truth":
        log_likelihood, errors = evaluator.eval_true("Truth")

    elif algorithm == "mle":
        log_likelihood, errors = evaluator.eval_exact_trellis("MLE (Trellis)")

    # Mean results
    log_likelihood = float(np.mean(np.array(log_likelihood).flatten()))
    errors = float(np.mean(np.array(errors).flatten()))

    # Log results
    _run.log_scalar("eval_log_likelihood", log_likelihood)
    _run.log_scalar("eval_illegal_actions", errors)

    # Print and plot results
    logger.info("Results:")
    print(evaluator)
    evaluator.plot_log_likelihoods(filename=f"{eval_figure_path}/{name}.pdf")

    return log_likelihood


@ex.capture
def save_agent(agent, run_name):
    if agent is not None:
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
    agent = create_agent(env=env)

    train(env, agent)
    save_agent(agent)

    result = eval(agent)

    logger.info(f"That's all, have a nice day!")
    return result
