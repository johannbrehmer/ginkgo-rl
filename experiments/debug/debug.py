import sys
import logging

sys.path.append("../")
from ginkgo_rl import GinkgoLikelihood1DEnv
from ginkgo_rl import PolicyMCTSAgent

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)
for key in logging.Logger.manager.loggerDict:
    if "ginkgo_rl" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

env = GinkgoLikelihood1DEnv()

model = MCBSAgent(env, n_mc_target=50, n_mc_min=1, beam_size=5, verbose=2)
model.learn(total_timesteps=10)
