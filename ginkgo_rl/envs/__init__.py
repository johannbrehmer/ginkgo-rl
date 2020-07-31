from gym.envs.registration import register
from .ginkgo_likelihood import GinkgoLikelihoodEnv

register(
    id='GinkgoLikelihood-v0',
    entry_point='ginkgo_rl.envs.ginkgo_likelihood:GinkgoLikelihoodEnv',
    max_episode_steps=100
)
