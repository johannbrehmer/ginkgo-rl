import sys
import numpy as np
from matplotlib import pyplot as plt
import gym
import logging

sys.path.append("../")
from ginkgo_rl import GinkgoLikelihood1DEnv
from ginkgo_rl import RandomMCTSAgent, MCTSAgent
from ginkgo_rl import GinkgoEvaluator


# Logging setup
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

# Ginkgo likes to output a lot of logging info, we don't really want that
for key in logging.Logger.manager.loggerDict:
    if "ginkgo_rl" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

env = GinkgoLikelihood1DEnv()

model = MCTSAgent(env, n_mc_target=50, n_mc_min=1)
model.learn(total_timesteps=10)

evaluator = GinkgoEvaluator(n_jets=2)
evaluator.eval("MCTS", model, "GinkgoLikelihood1D-v0")
