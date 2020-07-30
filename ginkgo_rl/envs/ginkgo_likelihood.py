import numpy as np
from gym import Env
from gym.spaces import Discrete, Box, Tuple
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

    def __init__(
        self,
        illegal_reward=-1000.,
        n_max=20,
        n_min=10,
        max_n_try=1000,
        w_rate=3.,
        qcd_rate=1.5,
        pt_min=0.3**2,
        jet_mass=80.,
        jet_momentum=400.,
        jetdir=(1, 1, 1),
    ):
        super().__init__()

        # Checks
        assert 1 < self.n_min < self.n_max

        # Hyperparameters
        self.illegal_reward = illegal_reward
        self.n_max = n_max
        self.n_min = n_min
        self.max_n_try = max_n_try
        self.w_rate = w_rate
        self.qcd_rate = qcd_rate
        self.pt_min = pt_min
        self.jet_mass = jet_mass
        jetdir = np.array(jetdir)
        jetvec = jet_momentum * jetdir / np.linalg.norm(jetdir)
        self.jet_momentum = np.concatenate(([np.sqrt(jet_momentum ** 2 + self.jet_mass ** 2)], jetvec))

        # Current state
        self.jet = None
        self.n = None
        self.state = None
        self.illegal_action_counter = 0

        # Prepare simulator
        self.sim = self._init_sim()
        self._simulate()

        # Spaces
        self.action_space = Tuple((Discrete(self.n_max), Discrete(self.n_max)))
        self.observation_space = Box(low=None, high=None, shape=(self.n_max, 3), dtype=np.float)

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
            done = self._check_if_done()
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

        logger.info(f"{n} particles:")
        for i, p in self.state[:self.n]:
            logger.info(f"  p[{i}]")

    def _init_sim(self):
        """ Initializes simulator """

        return GinkgoSim(
            jet_p=self.jet_momentum,
            pt_cut=self.pt_min,
            Delta_0=torch.tensor(self.jet_mass**2.),
            M_hard=self.jet_mass,
            num_samples=1,
            minLeaves=self.n_max,
            maxLeaves=self.n_min,
            maxNTry=self.max_n_try
        )

    def _simulate(self):
        """ Initiates an episode by simulating a new jet """
        rate = torch.tensor([self.w_rate, self.qcd_rate])
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

        assert self._check_legality(action)
        i, j = action
        ti, tj = self._compute_virtuality(self.state[i]), self._compute_virtuality(self.state[j])
        return ginkgo_log_likelihood(self.state[i], ti, self.state[j], tj, t_cut=self.pt_min, lam=1.)  # TODO: check t_cut, what is lam?!

    def _merge(self, action):
        """ Perform action, updating self.n and self.state """

        assert self._check_legality(action)
        i, j = action

        self.state[i, :] = self.state[i, :] + self.state[j, :]
        for k in range(j, self.n_max - 1):
            self.state[k, :] = self.state[k+1, :]
        self.state[-1, :] = np.zeros(4)
        self.n -= 1

    def _check_if_done(self):
        """ Checks if the current episode is done, i.e. if the clustering has reduced all particles to a single one """
        return self.n <= 1

    @staticmethod
    def _compute_virtuality(p):
        """ Computes the virtuality t form a four-vector p """
        return p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2
