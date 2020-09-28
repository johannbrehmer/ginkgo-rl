from gym.envs.registration import register
from .ginkgo_likelihood import (
    GinkgoLikelihoodEnv,
    GinkgoLikelihood1DEnv,
)  # , GinkgoLikelihoodShuffledEnv, GinkgoLikelihoodShuffled1DEnv

register(
    id="GinkgoLikelihood-v0", entry_point="ginkgo_rl.envs.ginkgo_likelihood:GinkgoLikelihoodEnv", max_episode_steps=100
)
register(
    id="GinkgoLikelihood1D-v0",
    entry_point="ginkgo_rl.envs.ginkgo_likelihood:GinkgoLikelihood1DEnv",
    max_episode_steps=100,
)
# register(
#     id='GinkgoLikelihoodShuffled-v0',
#     entry_point='ginkgo_rl.envs.ginkgo_likelihood:GinkgoLikelihoodShuffledEnv',
#     max_episode_steps=100
# )
# register(
#     id='GinkgoLikelihoodShuffled1D-v0',
#     entry_point='ginkgo_rl.envs.ginkgo_likelihood:GinkgoLikelihoodShuffled1DEnv',
#     max_episode_steps=100
# )
