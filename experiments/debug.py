import sys
import numpy as np
from matplotlib import pyplot as plt
import gym
import logging

sys.path.append("../")
from ginkgo_rl import GinkgoLikelihoodEnv

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

env = GinkgoLikelihoodEnv()
