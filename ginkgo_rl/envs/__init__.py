from gym.envs.registration import register
from .ginkgo_likelihood import GinkgoLikelihoodEnv, GinkgoLikelihood1DWrapper

register(
    id='GinkgoLikelihood-v0',
    entry_point='ginkgo_rl.envs.ginkgo_likelihood:GinkgoLikelihoodEnv',
    max_episode_steps=100
)
register(
    id='GinkgoLikelihood1D-v0',
    entry_point='ginkgo_rl.envs.ginkgo_likelihood:GinkgoLikelihood1DWrapper',
    max_episode_steps=100
)
