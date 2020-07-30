import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import logging
from showerSim.invMass_ginkgo import Simulator as GinkgoSim
from showerSim.likelihood_invM import split_logLH as ginkgo_log_likelihood
import torch

logger = logging.getLogger(__name__)


class InvalidActionException(Exception):
    pass


class GinkgoLikelihoodEnv(Env):
    """
    Ginkgo likelihood-based clustering environment.

    Each episode represents one clustering process, starting with the leaves and successfully merging pairs of leaves, until just one is left. In each episode

    The states are the leaf four-vectors at any point in the clustering process. They are given as an ndarray with shape (n_max, 4), where n_max is the maximal number of
    particles (not necessarily equal to the actual number of particles n). For i < n, the [i, 0] component represents energy, the remaining components spatial momentum.
    For i >= n, all components are zero and signify that there are no more particles.

    Actions are a tuple of two integers (i, j) with 0 <= i, j < n_max. A tuple (i, j) with i, j < n and i != j means merging the particles i and j. A tuple (i, j) with
    i >= n or j >= n or i = j is illegal.

    Rewards are the log likelihood of a 2 -> 1 merger under the true Ginkgo model. For illegal actions, the reward is instead given by illegal_reward.
    """

    def __init__(self, n_max=20, n_min=10, illegal_reward=-1000.):
        super().__init__()

        # Checks
        assert 0 < self.n_min < self.n_max

        # Hyperparameters
        self.n_max = n_max
        self.n_min = n_min
        self.illegal_reward = illegal_reward

        # Current state
        self.jet = None
        self.n = None
        self.state = None
        self.illegal_action_counter = 0

        # Prepare simulator
        self.sim = self._init_sim()
        self._simulate()

    def reset(self):
        """ Resets the state of the environment and returns an initial observation. """
        self._simulate()
        return self.state

    def reset_to(self, n, state, epsilon=1.e-9):
        """ Resets the state of the environment to a given state """
        self.jet = None  # Hopefully won't need this
        self.state = state
        self.illegal_action_counter = 0
        self.n = self.n_max
        for i in range(self.n_max):
            if np.max(np.abs(self.state[n, :])) < epsilon:
                self.n = i
                break

    def step(self, action):
        """ Environment step. """

        legal = self._check_legality(action)

        if legal:
            reward = self._compute_log_likelihood(action)
            self._merge(action)
            done = self._check_if_done
            info = {"legal": True}
            self.illegal_action_counter = 0
        else:
            reward = self.illegal_reward
            done = False
            info = {"legal": False}
            self.illegal_action_counter += 1

        if done:
            self._simulate()

        return self.state, reward, done, info

    def render(self, mode="human"):
        """ Visualize / report what's happening """
        pass

    def _init_sim(self, rate2=8., pt_min=0.3**2, jetdir=(1, 1, 1), M2start=80.**2, jetP=400., max_n_try=1000):
        pt_min = torch.tensor(pt_min)
        jetM = M2start**0.5
        jetdir = np.array(jetdir)
        jetvec = jetP * jetdir / np.linalg.norm(jetdir)
        jet4vec = np.concatenate(([np.sqrt(jetP ** 2 + jetM ** 2)], jetvec))

        return  GinkgoSim(
            jet_p=jet4vec,
            pt_cut=float(pt_min),
            Delta_0=torch.tensor(M2start),
            M_hard=jetM,
            num_samples=1,
            minLeaves=self.n_max,
            maxLeaves=self.n_min,
            maxNTry=max_n_try
        )

    def _simulate(self, W_rate = 3., QCD_rate = 1.5):
        """ Initiates an episode by simulating a new jet """
        rate = torch.tensor([W_rate, QCD_rate])
        self.jet = self.sim(rate)[0]
        self.n = len(self.jet["leaves"])
        self.state = np.zeros((self.n_max, 4))
        self.state[:self.n] = self.jet["leaves"]
        self.illegal_action_counter = 0

    def _check_legality(self, action):
        i, j = action
        return i != j and i >= 0 and j >= 0 and i < self.n and j < self.n

    def _compute_log_likelihood(self, action):
        """ Compute log likelihood of the splitting (i + j) -> i, j, where i, j is the current action """
        i, j = action
        assert self._check_legality(action)
        parent = self.state[i, :] + self.state[j, :]
        log_likelihood = ginkgo_log_likelihood(self.state[i], )

    def _merge(self, action):
        """ Perform action, updating self.n and self.state """
        i, j = action
        assert self._check_legality(action)
        self.state[i, :] = self.state[i, :] + self.state[j, :]
        self.state[j, :] = np.zeros(4)
        self.n -= 1
