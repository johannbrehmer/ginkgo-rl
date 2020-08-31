import gym
import numpy as np
from matplotlib import pyplot as plt
import sys
from tqdm import trange

from ginkgo_rl import GinkgoLikelihoodEnv, GinkgoLikelihoodShuffled1DEnv, GinkgoLikelihood1DEnv, GinkgoLikelihoodShuffledEnv

# Workaround for now until Trellis is better packaged
sys.path.append("/Users/johannbrehmer/work/projects/shower_rl/hierarchical-trellis/src")
from run_physics_experiment_invM import compare_map_gt_and_bs_trees as compute_trellis


class GinkgoEvaluator():
    def __init__(self, n_jets=8):
        self.n_jets = n_jets
        self.jets = self._init_jets()

        self.methods = []  # Method names
        self.log_likelihoods = {}  # Log likelihood results
        self.illegal_actions = {}  # Number of illegal actions

    def eval_true(self, method):
        self.methods.append(method)
        self.log_likelihoods[method] = [[self._compute_true_log_likelihood(jet)] for jet in self.jets]
        self.illegal_actions[method] = [[0] for _ in self.jets]

    def eval_exact_trellis(self, method):
        self.methods.append(method)
        self.log_likelihoods[method] = [[self._compute_maximum_log_likelihood(jet)] for jet in self.jets]
        self.illegal_actions[method] = [[0] for _ in self.jets]

    def eval(self, method, model, env_name, n_repeats=100):
        env = self._init_env(env_name)

        self.methods.append(method)
        self.log_likelihoods[method] = [[] for _ in range(self.n_jets)]
        self.illegal_actions[method] = [[] for _ in range(self.n_jets)]

        for i in trange(len(self.jets) * n_repeats):
            i_jet = i // n_repeats
            jet = self.jets[i_jet]

            env.set_internal_state(jet)
            log_likelihood, error = self._episode(model, env)
            self.log_likelihoods[method][i_jet].append(log_likelihood)
            self.illegal_actions[method][i_jet].append(error)

    def eval_random(self, method, env_name, n_repeats=100):
        self.eval(method, None, env_name, n_repeats)

    def get_results(self):
        for method in self.methods:
            yield method, self.log_likelihoods[method], self.illegal_actions[method]

    def plot_log_likelihoods(self, cols=2, rows=4, ymax=0.25, deltax_min=5., deltax_max=20., xbins=50, panelsize=4.):
        colors = [f"C{i}" for i in range(20)]
        fig = plt.figure(figsize=(rows*panelsize, cols*panelsize))

        for j in range(self.n_jets):
            if j > cols * rows:
                break

            plt.subplot(cols, rows, j + 1)

            xs = np.concatenate([logp[j] for logp in self.log_likelihoods.values()], axis=0)
            xmin, xmax = np.min(xs), np.max(xs)
            xmin = np.clip(xmin, xmax - deltax_max, xmax - deltax_min)
            xmax = xmax + 0.05 * (xmax - xmin)
            xmin = xmin - 0.05 * (xmax - xmin)

            for i, (name, logp, _) in enumerate(self.get_results()):
                logp_ = np.clip(logp, xmin + 1.e-9, xmax - 1.e-9)

                if len(logp[j]) == 1:
                    plt.plot([logp_[j][0], logp_[j][0]], [0., ymax], color=colors[i], ls="--", label=name)
                else:
                    plt.hist(logp_[j], histtype="stepfilled", range=(xmin, xmax), color=colors[i], bins=xbins, lw=1.5, density=True, alpha=0.2)
                    plt.hist(logp_[j], histtype="step", range=(xmin, xmax), bins=xbins, color=colors[i], lw=1.5, density=True, label=name)

            if j == 0:
                plt.legend()

            plt.xlabel("Log likelihood")
            plt.ylabel("Histogram")

            plt.xlim(xmin, xmax)
            plt.ylim(0., ymax)

        plt.tight_layout()
        return fig

    def _init_env(self, env_name="GinkgoLikelihood-v0"):
        if env_name == "GinkgoLikelihood-v0":
            env = GinkgoLikelihoodEnv(min_reward=None, illegal_reward=0., illegal_actions_patience=3)
        elif env_name == "GinkgoLikelihood1D-v0":
            env = GinkgoLikelihood1DEnv(min_reward=None, illegal_reward=0., illegal_actions_patience=3)
        elif env_name == "GinkgoLikelihoodShuffled-v0":
            env = GinkgoLikelihoodShuffledEnv(min_reward=None, illegal_reward=0., illegal_actions_patience=3)
        elif env_name == "GinkgoLikelihoodShuffled1D-v0":
            env = GinkgoLikelihoodShuffled1DEnv(min_reward=None, illegal_reward=0., illegal_actions_patience=3)
        else:
            raise ValueError(env_name)

        env.reset()
        return env

    def _init_jets(self):
        env = self._init_env()
        jets = []

        for _ in range(self.n_jets):
            env.reset()
            jets.append(env.get_internal_state())

        env.close()
        return jets

    def _episode(self, model, env):
        state = env.get_state()
        done = False
        log_likelihood = 0.
        errors = 0
        reward = 0.0

        # Point model to correct env: this only works for *our* models, not the baselines
        try:
            model.set_env(env)
        except:
            pass

        while not done:
            if model is None:
                action = env.action_space.sample()
                agent_info = {}
            else:
                action, agent_info = model.predict(state)
            next_state, next_reward, done, info = env.step(action)

            log_likelihood += next_reward
            if not info["legal"]:
                errors += 1

            # Update model: this only works for *our* models, not the baselines
            try:
                model.update(state, reward, action, done, next_state, next_reward=reward, num_episode=0, **agent_info)
            except:
                pass

            reward, state = next_reward, next_state

        return log_likelihood, errors

    @staticmethod
    def _compute_true_log_likelihood(jet):
        return sum(jet[0]["logLH"])

    def _compute_maximum_log_likelihood(self, jet):
        """ Based on Sebastian's code at https://github.com/iesl/hierarchical-trellis/blob/sebastian/src/Jet_Experiments_invM_exactTrellis.ipynb """
        _, _, max_log_likelihood, _, _ = compute_trellis(jet[0])
        return max_log_likelihood
