#!/usr/bin/env python

import logging
import sys
import torch
import os
import numpy as np

sys.path.append("../")
from experiments.config import ex, config, env_config, agent_config, train_config, technical_config
from ginkgo_rl import GinkgoLikelihood1DEnv, GinkgoLikelihoodEnv
from ginkgo_rl import PolicyMCTSAgent, GreedyAgent, RandomAgent, RandomMCTSAgent, LikelihoodMCTSAgent
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
    env, algorithm, policy, initialize_mcts_with_beamsearch, log_likelihood_policy_input, reward_range, history_length, train_n_mc_target, train_n_mc_min, train_n_mc_max, train_mcts_mode, train_c_puct, device, dtype, learning_rate, weight_decay, train_beamsize, debug, debug_verbosity
):
    logger.info(f"Setting up {algorithm} agent ")

    if algorithm == "mcts" and policy == "nn":
        agent = PolicyMCTSAgent(
            env,
            reward_range=reward_range,
            history_length=history_length,
            n_mc_target=train_n_mc_target,
            n_mc_min=train_n_mc_min,
            n_mc_max=train_n_mc_max,
            mcts_mode=train_mcts_mode,
            initialize_with_beam_search=initialize_mcts_with_beamsearch,
            log_likelihood_feature=log_likelihood_policy_input,
            c_puct=train_c_puct,
            lr=learning_rate,
            weight_decay=weight_decay,
            device=device,
            dtype=dtype,
            verbose=debug_verbosity if debug else 0,
        )
    elif algorithm == "mcts" and policy == "likelihood":
        agent = LikelihoodMCTSAgent(
            env,
            reward_range=reward_range,
            history_length=history_length,
            n_mc_target=train_n_mc_target,
            n_mc_min=train_n_mc_min,
            n_mc_max=train_n_mc_max,
            mcts_mode=train_mcts_mode,
            initialize_with_beam_search=initialize_mcts_with_beamsearch,
            c_puct=train_c_puct,
            device=device,
            dtype=dtype,
            lr=learning_rate,
            weight_decay=weight_decay,
            beam_size=train_beamsize,
            verbose=debug_verbosity if debug else 0,
        )
    elif algorithm == "mcts" and policy == "random":
        agent = RandomMCTSAgent(
            env,
            reward_range=reward_range,
            history_length=history_length,
            n_mc_target=train_n_mc_target,
            n_mc_min=train_n_mc_min,
            n_mc_max=train_n_mc_max,
            mcts_mode=train_mcts_mode,
            initialize_with_beam_search=initialize_mcts_with_beamsearch,
            c_puct=train_c_puct,
            device=device,
            dtype=dtype,
            verbose=debug_verbosity if debug else 0,
        )
    elif algorithm == "greedy":
        agent = GreedyAgent(env, device=device, dtype=dtype, verbose=debug_verbosity if debug else 0)
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
def train(env, agent, algorithm, policy, train_n_mc_target, train_n_mc_min, train_n_mc_max, train_mcts_mode, train_c_puct, train_steps, train_beamsize):
    if algorithm in ["greedy", "random", "truth", "mle", "beamsearch"]:
        logger.info(f"No training necessary for algorithm {algorithm}")
    elif algorithm == "mcts" and policy in ["random", "likelihood"]:
        logger.info(f"No training necessary for MCTS policy {policy}")
    else:
        logger.info(f"Starting {algorithm} training for {train_steps} steps")

        # Initialize MCTS agent settings (this won't do anything for some other types of agent)
        try:
            agent.set_precision(train_n_mc_target, train_n_mc_min, train_n_mc_max, train_mcts_mode, train_c_puct, train_beamsize)
        except:
            pass

        # Train
        _ = env.reset()
        agent.learn(total_timesteps=train_steps, callback=log_training)

    env.close()


@ex.capture
def eval(agent, env, name, algorithm, eval_n_mc_target, eval_n_mc_min, eval_n_mc_max, eval_mcts_mode, eval_c_puct, eval_repeats, eval_jets, eval_filename, redraw_eval_jets, run_name, eval_beamsize, _run):
    # Set up evaluator
    logger.info("Starting evaluation")
    os.makedirs(os.path.dirname(eval_filename), exist_ok=True)
    evaluator = GinkgoEvaluator(env=env, filename=eval_filename, n_jets=eval_jets, redraw_existing_jets=redraw_eval_jets)
    jet_sizes = evaluator.get_jet_info()["n_leaves"]

    # Evaluate
    if algorithm == "mcts":
        agent.set_precision(eval_n_mc_target, eval_n_mc_min, eval_n_mc_max, eval_mcts_mode, eval_c_puct, eval_beamsize)
        log_likelihood, errors, likelihood_evaluations = evaluator.eval(name, agent, n_repeats=eval_repeats)
    elif algorithm == "random":
        log_likelihood, errors, likelihood_evaluations = evaluator.eval(name, agent, n_repeats=eval_repeats)
    elif algorithm == "greedy":
        log_likelihood, errors, likelihood_evaluations = evaluator.eval(name, agent, n_repeats=1)
    elif algorithm == "beamsearch":
        log_likelihood, errors, likelihood_evaluations = evaluator.eval_beam_search(name, beam_size=eval_beamsize)
    elif algorithm == "truth":
        log_likelihood, errors, likelihood_evaluations = evaluator.eval_true("Truth")
    elif algorithm == "mle":
        log_likelihood, errors, likelihood_evaluations = evaluator.eval_exact_trellis("MLE (Trellis)")
    else:
        raise ValueError(algorithm)

    # Store results
    folder = f"./data/runs/{run_name}"
    logger.info(f"Storing results in {folder}")
    np.save(f"{folder}/eval_log_likelihood.npy", log_likelihood)
    np.save(f"{folder}/eval_illegal_actions.npy", errors)
    np.save(f"{folder}/eval_jet_sizes.npy", jet_sizes)
    np.save(f"{folder}/eval_likelihood_evaluations.npy", likelihood_evaluations)
    ex.add_artifact(f"{folder}/eval_log_likelihood.npy")
    ex.add_artifact(f"{folder}/eval_illegal_actions.npy")
    ex.add_artifact(f"{folder}/eval_jet_sizes.npy")
    ex.add_artifact(f"{folder}/eval_likelihood_evaluations.npy")

    # Log results
    mean_log_likelihood = float(np.mean(np.array(log_likelihood).flatten()))
    mean_errors = float(np.mean(np.array(errors).flatten()))
    mean_likelihood_evaluations = float(np.mean(np.array(likelihood_evaluations).flatten()))
    _run.log_scalar("eval_log_likelihood", mean_log_likelihood)
    _run.log_scalar("eval_illegal_actions", mean_errors)
    _run.log_scalar("eval_likelihood_evaluations", mean_likelihood_evaluations)
    logger.info("Eval results:")
    logger.info(f"  Mean log likelihood: {mean_log_likelihood}")
    logger.info(f"  Mean illegal actions: {mean_errors}")
    logger.info(f"  Mean likelihood evaluations: {mean_likelihood_evaluations}")

    return mean_log_likelihood


@ex.capture
def save_agent(agent, algorithm, policy, run_name):
    if algorithm == "mcts" and policy == "nn" and agent is not None:
        filename = f"./data/runs/{run_name}/model.pty"
        logger.info(f"Saving model at {filename}")
        torch.save(agent.state_dict(), filename)
        ex.add_artifact(filename)
    else:
        logger.debug(f"No model weights need to be saved for algorithm {algorithm} and policy {policy}")


@ex.automain
def main():
    logger.info(f"Hi!")

    setup_run()
    setup_logging()
    env = create_env()
    agent = create_agent(env=env)

    train(env, agent)
    save_agent(agent)

    result = eval(agent, env)

    logger.info(f"That's all, have a nice day!")
    return result
