from tqdm import tqdm
import pickle
import logging
import torch
from copy import deepcopy
import numpy as np

from ginkgo_rl.envs import GinkgoLikelihood1DEnv
from ginkgo_rl.agents import PolicyMCTSAgent

logger = logging.getLogger(__name__)


class GinkgoRLInterface:
    def __init__(self, state_dict_filename, **kwargs):
        self.env = self._make_env(**kwargs)
        self.agent = self._make_agent(state_dict_filename, **kwargs)

    def generate(self, n):
        """ Generates a number of jets and returns the jet dictionary """

        logger.info(f"Generating {n} jets")
        jets = []

        for _ in range(n):
            self.env.reset()
            jets.append(self.env.get_internal_state()[0])

        logger.info(f"Done")

        return jets

    def cluster(self, jets, filename=None, mode=None):
        """ Clusters all jets in a jet dictionary with the MCTS agent. """

        jets = self._load_jets(jets)

        logger.info(f"Clustering {len(jets)} jets")

        reclustered_jets = []
        log_likelihoods = []
        illegal_actions = []
        likelihood_evaluations = []

        for jet in tqdm(jets):
            with torch.no_grad():
                reclustered_jet, log_likelihood, error, likelihood_evaluation = self._episode(jet, mode=mode)

            reclustered_jets.append(reclustered_jet)
            log_likelihoods.append(log_likelihood)
            illegal_actions.append(error)
            likelihood_evaluations.append(likelihood_evaluation)

        if filename is not None:
            self._save_jets(reclustered_jets, filename)

        logger.info("Done")

        return reclustered_jets, log_likelihoods, illegal_actions, likelihood_evaluations

    def _episode(self, jet, mode=None):
        """ Clusters a single jet """

        # Initialize
        self.agent.eval()
        self.env.set_internal_state(self._jet_to_internal_state(jet))

        state = self.env.get_state()
        done = False
        log_likelihood = 0.0
        errors = 0
        reward = 0.0
        likelihood_evaluations = 0

        reclustered_jet = self._init_reclustered_jet(jet)

        # Point agent to correct env and initialize episode: this only works for *our* models, not the baselines
        try:
            self.agent.set_env(self.env)
            self.agent.init_episode()
        except:
            pass

        while not done:
            # Agent step
            if self.agent is None:
                action = self.env.action_space.sample()
                agent_info = {}
            elif mode is None:
                action, agent_info = self.agent.predict(state)
                likelihood_evaluations = max(agent_info["likelihood_evaluations"], likelihood_evaluations)
            else:
                action, agent_info = self.agent.predict(state, mode=mode)
                likelihood_evaluations = max(agent_info["likelihood_evaluations"], likelihood_evaluations)

            # Environment step
            next_state, next_reward, done, info = self.env.step(action)

            # Keep track of clustered tree
            if info["legal"]:
                self._update_reclustered_jet_with_action(reclustered_jet, action, next_reward)

            # Keep track of metrics
            log_likelihood += next_reward
            if not info["legal"]:
                errors += 1

            # Update model: this only works for *our* models, not the baselines
            try:
                self.agent.update(
                    state, reward, action, done, next_state, next_reward=next_reward, num_episode=0, **agent_info
                )
            except:
                pass

            reward, state = next_reward, next_state

        self._finalize_reclustered_jet(reclustered_jet)

        return reclustered_jet, float(log_likelihood), int(errors), int(likelihood_evaluations)

    def _init_reclustered_jet(self, jet, delete_keys=("deltas", "draws", "dij", "ConstPhi", "PhiDelta", "PhiDeltaRel")):
        reclustered_jet = deepcopy(jet)

        reclustered_jet["content"] = list(deepcopy(reclustered_jet["leaves"]))
        reclustered_jet["tree"] = [[-1, -1] for _ in reclustered_jet["content"]]
        reclustered_jet["logLH"] = [0.0 for _ in reclustered_jet["content"]]
        reclustered_jet["root_id"] = None
        reclustered_jet["algorithm"] = "mcts"
        reclustered_jet["current_particles"] = set(
            range(len(reclustered_jet["content"]))
        )  # dict IDs of current particles

        for key in delete_keys:
            try:
                del reclustered_jet[key]
            except:
                logger.info(f"Jet dict did not contain field {key}")

        return reclustered_jet

    def _update_reclustered_jet_with_action(self, reclustered_jet, action, step_log_likelihood):
        # Parse action
        i_en, j_en = self.env.unwrap_action(action)  # energy-sorted IDs of the particles to be merged
        particles = [(dict_id, reclustered_jet["content"][dict_id]) for dict_id in reclustered_jet["current_particles"]]
        particles = sorted(
            particles, reverse=True, key=lambda x: x[1][0]
        )  # (dict_ID, four_momentum) of current particles, sorted by E
        i_dict, j_dict = particles[i_en][0], particles[j_en][0]  # dict IDs of the particles to be merged

        logger.debug(f"Parsing action {action}:")
        logger.debug("  E-ranking | dict ID | momentum ")
        for en_id, (dict_id, four_momentum) in enumerate(particles):
            logger.debug(
                f"  {'x' if dict_id in (i_dict, j_dict) else ' '} {en_id:>7d} | {dict_id:>7d} | {four_momentum} "
            )

        # Perform action
        new_momentum = reclustered_jet["content"][i_dict] + reclustered_jet["content"][j_dict]
        reclustered_jet["content"].append(new_momentum)
        reclustered_jet["tree"].append([i_dict, j_dict])
        k_dict = len(reclustered_jet["content"]) - 1

        reclustered_jet["root_id"] = k_dict
        reclustered_jet["logLH"].append(step_log_likelihood)

        reclustered_jet["current_particles"].remove(i_dict)
        reclustered_jet["current_particles"].remove(j_dict)
        reclustered_jet["current_particles"].add(k_dict)

    def _finalize_reclustered_jet(self, reclustered_jet):
        reclustered_jet["content"] = np.asarray(reclustered_jet["content"])
        reclustered_jet["logLH"] = np.asarray(reclustered_jet["logLH"])
        reclustered_jet["tree"] = np.asarray(reclustered_jet["tree"], dtype=np.int)

        del reclustered_jet["current_particles"]

    def _make_agent(
        self,
        state_dict,
        initialize_mcts_with_beamsearch=True,
        log_likelihood_policy_input=True,
        decision_mode="max_reward",
        reward_range=(-500.0, 0.0),
        hidden_sizes=(100, 100),
        activation=torch.nn.ReLU(),
        n_mc_target=2,
        n_mc_min=0,
        n_mc_max=50,
        beamsize=20,
        planning_mode="mean",
        c_puct=1.0,
        device=torch.device("cpu"),
        dtype=torch.float,
        **kwargs,
    ):
        agent = PolicyMCTSAgent(
            self.env,
            reward_range=reward_range,
            n_mc_target=n_mc_target,
            n_mc_min=n_mc_min,
            n_mc_max=n_mc_max,
            planning_mode=planning_mode,
            initialize_with_beam_search=initialize_mcts_with_beamsearch,
            log_likelihood_feature=log_likelihood_policy_input,
            c_puct=c_puct,
            device=device,
            dtype=dtype,
            verbose=0,
            decision_mode=decision_mode,
            beam_size=beamsize,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        try:
            state_dict = torch.load(state_dict)
        except:
            pass
        agent.load_state_dict(state_dict)

        return agent

    def _make_env(
        self,
        illegal_reward=-100.0,
        illegal_actions_patience=3,
        n_max=20,
        n_min=2,
        n_target=1,
        min_reward=-100.0,
        state_rescaling=0.01,
        padding_value=-1.0,
        w_jet=True,
        w_rate=3.0,
        qcd_rate=1.5,
        pt_min=4.0 ** 2,
        qcd_mass=30.0,
        w_mass=80.0,
        jet_momentum=400.0,
        jetdir=(1, 1, 1),
        max_n_try=1000,
        **kwargs,
    ):
        env = GinkgoLikelihood1DEnv(
            illegal_reward,
            illegal_actions_patience,
            n_max,
            n_min,
            n_target,
            min_reward,
            state_rescaling,
            padding_value,
            w_jet,
            max_n_try,
            w_rate,
            qcd_rate,
            pt_min,
            qcd_mass,
            w_mass,
            jet_momentum,
            jetdir,
        )

        return env

    def _load_jets(self, jets):
        try:
            with open(jets, "rb") as f:
                return pickle.load(f)
        except:
            return jets

    def _save_jets(self, jets, filename):
        logger.info(f"Saving clustered jets at {filename}")

        with open(filename, "wb") as f:
            pickle.dump(jets, f)

    def _internal_state_to_jet(self, internal_state):
        return internal_state[0]

    def _jet_to_internal_state(self, jet):
        """
        Translates a jet dict to the environment internal state, a 5-tuple of the form (jet_dict, n_particles, state, is_leaf, illegal_action_counter).

        Only works for "initial" states (no clustering so far, only observed particles).
        """

        n = len(jet["leaves"])
        state = self.env.padding_value * np.ones((self.env.n_max, 4))
        state[:n] = self.env.state_rescaling * jet["leaves"]
        is_leaf = [(i < n) for i in range(self.env.n_max)]
        illegal_action_counter = 0

        # energy sorting
        idx = sorted(list(range(self.env.n_max)), reverse=True, key=lambda i: state[i, 0])
        state = state[idx, :]
        is_leaf = np.asarray(is_leaf, dtype=np.bool)[idx]

        internal_state = (jet, n, state, is_leaf, illegal_action_counter)
        return internal_state
