import numpy as np
from gym import Env
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
import logging
from showerSim.invMass_ginkgo import Simulator as GinkgoSim
from showerSim.likelihood_invM import split_logLH as ginkgo_log_likelihood
import torch

logger = logging.getLogger(__name__)


class InvalidActionException(Exception):
    pass


class GinkgoLikelihoodEnv(Env):
    """
    Ginkgo clustering environment with likelihood-based rewards.

    Each episode represents one clustering process, starting with all leaves and successfully merging pairs of leaves, until just n_target particles are left. In each episode,
    a new jet is sampled (with the same initial conditions).

    The states are the particle four-vectors at any point in the clustering process. They are given as an ndarray with shape (n_max, 4), where n_max is the maximal number of
    particles (not necessarily equal to the actual number of particles n). For i < n, the [i, 0] component represents energy, the remaining components spatial momentum.
    For i >= n, all components are zero and signify that there are no more particles.

    Actions are a tuple of two integers (i, j) with 0 <= i, j < n_max. A tuple (i, j) with i, j < n and i != j means merging the particles i and j. A tuple (i, j) with
    i >= n or j >= n or i = j is illegal.

    Rewards are the log likelihood of a 2 -> 1 merger under the true Ginkgo model.

    For illegal actions, the reward is instead given by illegal_reward. The first illegal_actions_patience illegal actions in a row are accepted, incur the illegal_reward reward,
    but do not change the system. Another illegal action after that incurs the illegal_reward reward and a merger is picked at random.
    """

    def __init__(
        self,
        illegal_reward=-20.0,
        illegal_actions_patience=3,
        n_max=16,
        n_min=3,
        n_target=2,
        min_reward=-20.0,
        state_rescaling=0.01,
        w_jet=True,
        max_n_try=1000,
        w_rate=3.0,
        qcd_rate=1.5,
        pt_min=4.0 ** 2,
        qcd_mass=30.0,
        w_mass=80.0,
        jet_momentum=400.0,
        jetdir=(1, 1, 1),
    ):
        super().__init__()

        logger.debug("Initializing environment")

        # Checks
        assert 1 <= n_target < n_min < n_max

        # Main hyperparameters
        self.illegal_reward = illegal_reward
        self.illegal_actions_patience = illegal_actions_patience
        self.n_max = n_max
        self.n_target = n_target
        self.min_reward = min_reward
        self.state_rescaling = state_rescaling

        # Simulator settings
        self.n_min = n_min
        self.max_n_try = max_n_try
        self.w_jet = w_jet
        self.w_rate = w_rate
        self.qcd_rate = qcd_rate
        self.pt_min = pt_min
        self.qcd_mass = qcd_mass
        self.w_mass = w_mass
        self.jet_mass = self.w_mass if w_jet else self.qcd_mass
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
        self.action_space = MultiDiscrete((self.n_max, self.n_max))  # Tuple((Discrete(self.n_max), Discrete(self.n_max)))
        self.observation_space = Box(low=0., high=state_rescaling * max(self.jet_momentum), shape=(self.n_max, 4), dtype=np.float)

    def reset(self):
        """ Resets the state of the environment and returns an initial observation. """

        logger.debug("Resetting environment")

        self._simulate()
        return self.state

    def reset_to(self, state, epsilon=1.0e-9):
        """ Resets the state of the environment to a given state """

        logger.debug("Resetting environment to a given state")

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

        logger.debug(f"Environment step. Action: {action}")

        legal = self.check_legality(action)

        if legal:
            reward = self._compute_log_likelihood(action)
            self._merge(action)
            done = self._check_if_done()
            self.illegal_action_counter = 0
            info = {
                "legal": True,
                "illegal_action_counter": self.illegal_action_counter,
                "replace_illegal_action": False,
                "i": action[0],
                "j": action[1],
            }

        else:
            logger.debug(f"Action {action} is illegal (n = {self.n}).")

            reward = self.illegal_reward
            self.illegal_action_counter += 1

            if self.illegal_action_counter > self.illegal_actions_patience:
                new_action = self._draw_random_legal_action()
                logger.debug(f"This is the {self.illegal_action_counter}th illegal action in a row. That's enough. Executing random action {new_action} instead.")

                self._merge(new_action)
                done = self._check_if_done()
                info = {
                    "legal": False,
                    "illegal_action_counter": self.illegal_action_counter,
                    "replace_illegal_action": True,
                    "i": new_action[0],
                    "j": new_action[1],
                }
                self.illegal_action_counter = 0
            else:
                logger.debug(f"This is the {self.illegal_action_counter}th illegal action in a row. Try again. (Environment state is unchanged.)")

                done = False
                info = {
                    "legal": False,
                    "illegal_action_counter": self.illegal_action_counter,
                    "replace_illegal_action": False,
                    "i": None,
                    "j": None,
                }

        if done:
            logger.debug("Episode is done.")
            self._simulate()

        return self.state, reward, done, info

    def render(self, mode="human"):
        """ Visualize / report what's happening """

        logger.info(f"{self.n} particles:")
        for i, p in enumerate(self.state[: self.n]):
            logger.info(f"  p[{i:>2d}] = ({p[0]:5.1f}, {p[1]:5.1f}, {p[2]:5.1f}, {p[3]:5.1f})")

    def _init_sim(self):
        """ Initializes simulator """

        return GinkgoSim(
            jet_p=self.jet_momentum,
            pt_cut=self.pt_min,
            Delta_0=torch.tensor(self.jet_mass ** 2.0),
            M_hard=self.jet_mass,
            num_samples=1,
            minLeaves=self.n_min,
            maxLeaves=self.n_max,
            maxNTry=self.max_n_try,
        )

    def _simulate(self):
        """ Initiates an episode by simulating a new jet """

        rate = torch.tensor([self.w_rate, self.qcd_rate]) if self.w_jet else torch.tensor([self.qcd_rate, self.qcd_rate])
        jets = self.sim(rate)
        if not jets:
            raise RuntimeError(f"Could not generate any jets: {jets}")

        self.jet = self.sim(rate)[0]
        self.n = len(self.jet["leaves"])
        self.state = np.zeros((self.n_max, 4))
        self.state[: self.n] = self.state_rescaling * self.jet["leaves"]
        self.illegal_action_counter = 0

        logger.debug(f"Sampling new jet with {self.n} leaves")

    def _check_acceptability(self, action):
        i, j = action
        return i >= 0 and j >= 0 and i < self.n_max and j < self.n_max

    def check_legality(self, action):
        """ Check legality of an action """
        i, j = action
        return self._check_acceptability(action) and i != j and i < self.n and j < self.n

    def _compute_log_likelihood(self, action):
        """ Compute log likelihood of the splitting (i + j) -> i, j, where i, j is the current action """

        assert self.check_legality(action)
        i, j = action

        ti, tj = self._compute_virtuality(self.state[i] / self.state_rescaling), self._compute_virtuality(self.state[j]  / self.state_rescaling)
        t_cut = self.jet["pt_cut"]
        lam = self.jet["Lambda"]
        if self.n == 2 and self.w_jet:
            lam = self.jet["LambdaRoot"]  # W jets have a different lambda for the first split

        log_likelihood = ginkgo_log_likelihood(self.state[i] / self.state_rescaling, ti, self.state[j] / self.state_rescaling, tj, t_cut=t_cut, lam=lam)
        try:
            log_likelihood = log_likelihood.item()
        except:
            pass
        log_likelihood = np.clip(log_likelihood, self.min_reward, None)

        logger.debug(f"Computing log likelihood of action {action}: {log_likelihood}")

        return log_likelihood

    def _merge(self, action):
        """ Perform action, updating self.n and self.state """

        assert self.check_legality(action)
        i, j = action

        self.state[i, :] = self.state[i, :] + self.state[j, :]
        for k in range(j, self.n_max - 1):
            self.state[k, :] = self.state[k + 1, :]
        self.state[-1, :] = np.zeros(4)
        self.n -= 1

        logger.debug(f"Merging particles {i} and {j}. New state has {self.n} particles.")

    def _check_if_done(self):
        """ Checks if the current episode is done, i.e. if the clustering has reduced all particles to a single one """
        return self.n <= self.n_target

    @staticmethod
    def _compute_virtuality(p):
        """ Computes the virtuality t form a four-vector p """
        return p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2

    def _draw_random_legal_action(self):
        assert not self._check_if_done() and self.n > 1
        i, j = -1, -1
        while i == j:
            i = np.random.randint(low=0, high=self.n - 1)
            j = np.random.randint(low=0, high=self.n - 1)
        assert self.check_legality((i, j))
        return i, j


class GinkgoLikelihood1DWrapper(GinkgoLikelihoodEnv):
    """
    Wrapper around GinkgoLikelihoodEnv to support baseline RL algorithms designed for 1D discrete, non-tuple action spaces.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_actions = self.n_max * (self.n_max - 1) // 2
        self.action_space = Discrete(self.n_actions)

    def wrap_action(self, action_tuple):
        assert self._check_acceptability(action_tuple)
        i, j = max(action_tuple), min(action_tuple)
        return i  * (i - 1) // 2 + j

    def unwrap_action(self, action_int):
        i = 1
        for k in range(1, self.n_max + 1):
            if k * (k - 1) // 2 > action_int:
                i = k - 1
                break

        j = action_int - i * (i - 1) // 2

        assert self._check_acceptability((i, j))
        return i, j

    def step(self, action):
        try:
            _, _ = action
            return super().step(action)
        except TypeError:
            return super().step(self.unwrap_action(action))

    def check_legality(self, action):
        try:
            _, _ = action
            return super().check_legality(action)
        except TypeError:
            return super().check_legality(self.unwrap_action(action))
