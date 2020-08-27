import numpy as np
from gym import Env
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
import logging
from showerSim.invMass_ginkgo import Simulator as GinkgoSim
from showerSim.likelihood_invM import split_logLH as ginkgo_log_likelihood
import torch
import itertools
import copy

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
    For i >= n, all components are set to -1. and signify that there are no more particles.

    Actions are a tuple of two integers (i, j) with 0 <= i, j < n_max. A tuple (i, j) with i, j < n and i != j means merging the particles i and j. A tuple (i, j) with
    i >= n or j >= n or i = j is illegal.

    Rewards are the log likelihood of a 2 -> 1 merger under the true Ginkgo model.

    For illegal actions, the reward is instead given by illegal_reward. The first illegal_actions_patience illegal actions in a row are accepted, incur the illegal_reward reward,
    but do not change the system. Another illegal action after that incurs the illegal_reward reward and a merger is picked at random.
    """

    def __init__(
        self,
        illegal_reward=-50.0,
        illegal_actions_patience=5,
        n_max=10,
        n_min=2,
        n_target=1,
        min_reward=-100.0,
        state_rescaling=0.01,
        padding_value=-1.0,
        w_jet=True,
        max_n_try=1000,
        w_rate=3.0,
        qcd_rate=1.5,
        pt_min=4.0 ** 2,
        qcd_mass=30.0,
        w_mass=80.0,
        jet_momentum=400.0,
        jetdir=(1, 1, 1),
        verbose=True,
        reset_at_episode_end=True,
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
        self.padding_value = padding_value
        self.verbose = verbose
        self.reset_at_episode_end = reset_at_episode_end

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
        self.is_leaf = None

        # Prepare simulator
        self.sim = self._init_sim()
        # self._simulate()

        # Spaces
        self.action_space = MultiDiscrete((self.n_max, self.n_max))  # Tuple((Discrete(self.n_max), Discrete(self.n_max)))
        self.observation_space = Box(low=self.padding_value, high=state_rescaling * max(self.jet_momentum), shape=(self.n_max, 4), dtype=np.float)

    def reset(self):
        """ Resets the state of the environment and returns an initial observation. """

        if self.verbose:
            logger.debug("Resetting environment")

        self._simulate()
        return self.state

    def get_state(self):
        return self.state

    def get_internal_state(self):
        return copy.deepcopy((self.jet, self.n, self.state, self.is_leaf, self.illegal_action_counter))

    def set_internal_state(self, internal_state):
        self.jet, self.n, self.state, self.is_leaf, self.illegal_action_counter = copy.deepcopy(internal_state)

    def step(self, action):
        """ Environment step. """

        if self.verbose:
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
            if self.verbose:
                logger.debug(f"Action {action} is illegal (n = {self.n}).")

            reward = self.illegal_reward
            self.illegal_action_counter += 1

            if self.illegal_action_counter > self.illegal_actions_patience:
                new_action = self._draw_random_legal_action()
                if self.verbose:
                    logger.debug(f"This is the {self.illegal_action_counter}th illegal action in a row. That's enough. Executing random action {new_action} instead.")

                reward += self._compute_log_likelihood(new_action)
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
                if self.verbose:
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
            if self.verbose:
                logger.debug("Episode is done.")
            if self.reset_at_episode_end:
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
        self.state = self.padding_value * np.ones((self.n_max, 4))
        self.state[: self.n] = self.state_rescaling * self.jet["leaves"]
        self.is_leaf = [(i < self.n) for i in range(self.n_max)]
        self.illegal_action_counter = 0
        self._sort_state()

        if self.verbose:
            logger.debug(f"Sampling new jet with {self.n} leaves")

    def _check_acceptability(self, action):
        i, j = action
        return i >= 0 and j >= 0 and i < self.n_max and j < self.n_max

    def check_legality(self, action):
        """ Check legality of an action """
        i, j = action
        return self._check_acceptability(action) and i != j and i < self.n and j < self.n

    def _sort_state(self):
        pass
        # idx = sorted(list(range(self.n_max)), reverse=True, key=lambda i : self.state[i, 0])
        # self.state = self.state[idx, :]
        # self.is_leaf = np.asarray(self.is_leaf, dtype=np.bool)[idx]

    def _compute_log_likelihood(self, action):
        """ Compute log likelihood of the splitting (i + j) -> i, j, where i, j is the current action """

        assert self.check_legality(action)
        i, j = action

        ti, tj = self._compute_virtuality(i), self._compute_virtuality(j)
        t_cut = self.jet["pt_cut"]
        lam = self.jet["Lambda"]
        if self.n == 2 and self.w_jet:
            lam = self.jet["LambdaRoot"]  # W jets have a different lambda for the first split

        log_likelihood = ginkgo_log_likelihood(self.state[i] / self.state_rescaling, ti, self.state[j] / self.state_rescaling, tj, t_cut=t_cut, lam=lam)
        try:
            log_likelihood = log_likelihood.item()
        except:
            pass

        if self.min_reward is not None:
            log_likelihood = np.clip(log_likelihood, self.min_reward, None)

        if self.verbose:
            logger.debug(f"Computing log likelihood of action {action}: ti = {ti}, tj = {tj}, t_cut = {t_cut}, lam = {lam} -> log likelihood = {log_likelihood}")
            # logger.debug(f"Computing log likelihood of action {action}: log likelihood = {log_likelihood}")

        return log_likelihood

    def _merge(self, action):
        """ Perform action, updating self.n and self.state """

        assert self.check_legality(action)
        i, j = action

        self.state[i, :] = self.state[i, :] + self.state[j, :]
        self.is_leaf[i] = False

        for k in range(j, self.n_max - 1):
            self.state[k, :] = self.state[k + 1, :]
            self.is_leaf[k] = self.is_leaf[k+1]

        self.state[-1, :] = self.padding_value * np.ones(4)
        self.is_leaf[-1] = False

        self.n -= 1
        self._sort_state()

        if self.verbose:
            logger.debug(f"Merging particles {i} and {j}. New state has {self.n} particles.")

    def _check_if_done(self):
        """ Checks if the current episode is done, i.e. if the clustering has reduced all particles to a single one """
        return self.n <= self.n_target

    def _compute_virtuality(self, i):
        """ Computes the virtuality t of particle i """
        if self.is_leaf[i]:
            return 0.0  # See discussion with Sebastian

        p = self.state[i, :] / self.state_rescaling
        return p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2

    def _draw_random_legal_action(self):
        assert not self._check_if_done() and self.n > 1
        i, j = -1, -1
        while i == j:
            i = np.random.randint(low=0, high=self.n)
            j = np.random.randint(low=0, high=self.n)
        assert self.check_legality((i, j))
        return i, j


class GinkgoLikelihood1DEnv(GinkgoLikelihoodEnv):
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
        return i, j

    def step(self, action):
        try:
            _, _ = action
            return super().step(action)
        except TypeError:
            return super().step(self.unwrap_action(action))


class PermutationMixin():
    @staticmethod
    def _create_permutation(n):
        permutation, inverse_permutation = [None for _ in range(n)], [None for _ in range(n)]
        for i, j in enumerate(np.random.permutation(n)):
            permutation[i] = j
            inverse_permutation[j] = i
        return permutation, inverse_permutation


class GinkgoLikelihoodShuffledEnv(PermutationMixin, GinkgoLikelihoodEnv):
    """
    Wrapper around GinkgoLikelihoodEnv that shuffles the particles so they show up in random positions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.permutation, self.inverse_permutation = self._create_permutation(self.n_max)

    def reset(self):
        state = super().reset()
        self.permutation, self.inverse_permutation = self._create_permutation(self.n_max)
        state = state[self.permutation, :]
        return state

    def step(self, action):
        action_ = self.permutation[action[0]], self.permutation[action[1]]
        state_, reward, done, info = super().step(action_)
        self.permutation, self.inverse_permutation = self._create_permutation(self.n_max)
        state = state_[self.permutation, :]
        return state, reward, done, info

    def render(self, mode="human"):
        """ Visualize / report what's happening """

        logger.info(f"{self.n} particles:")
        for i in range(self.n_max):
            i_ = self.permutation[i]
            p = self.state[i_]
            if np.max(p) > 0.:
                logger.info(f"  p[{i:>2d}] = ({p[0]:5.1f}, {p[1]:5.1f}, {p[2]:5.1f}, {p[3]:5.1f})")

    def get_state(self):
        return self.state[self.permutation, :]


class GinkgoLikelihoodShuffled1DEnv(GinkgoLikelihoodShuffledEnv):
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

        return i, j

    def step(self, action):
        try:
            _, _ = action
            return super().step(action)
        except TypeError:
            return super().step(self.unwrap_action(action))
