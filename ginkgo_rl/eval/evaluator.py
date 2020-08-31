import gym
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from tqdm import trange
import pickle

from ginkgo_rl import GinkgoLikelihoodEnv, GinkgoLikelihoodShuffled1DEnv, GinkgoLikelihood1DEnv, GinkgoLikelihoodShuffledEnv

# Workaround for now until Trellis is better packaged
sys.path.append("/Users/johannbrehmer/work/projects/shower_rl/hierarchical-trellis/src")
from run_physics_experiment_invM import compare_map_gt_and_bs_trees as compute_trellis


class GinkgoEvaluator():
    def __init__(self, filename, redraw_existing_jets=False, n_jets=8, auto_eval_truth_mle=True):
        self.filename = filename

        if os.path.exists(filename) and not redraw_existing_jets:
            self._load()
        else:
            self.n_jets = n_jets
            self.jets = self._init_jets()
            self.methods = []  # Method names
            self.log_likelihoods = {}  # Log likelihood results
            self.illegal_actions = {}  # Number of illegal actions
            self._save()

            if auto_eval_truth_mle:
                self.eval_true("Truth")
                self.eval_exact_trellis("MLE (Trellis)")

    def eval_true(self, method):
        log_likelihoods = [[self._compute_true_log_likelihood(jet)] for jet in self.jets]
        illegal_actions = [[0] for _ in self.jets]
        self._update_results(method, log_likelihoods, illegal_actions)

    def eval_exact_trellis(self, method):
        log_likelihoods = [[self._compute_maximum_log_likelihood(jet)] for jet in self.jets]
        illegal_actions = [[0] for _ in self.jets]
        self._update_results(method, log_likelihoods, illegal_actions)

    def eval(self, method, model, env_name, n_repeats=100):
        env = self._init_env(env_name)

        log_likelihoods = [[] for _ in range(self.n_jets)]
        illegal_actions = [[] for _ in range(self.n_jets)]

        for i in trange(len(self.jets) * n_repeats):
            i_jet = i // n_repeats
            jet = self.jets[i_jet]

            env.set_internal_state(jet)
            log_likelihood, error = self._episode(model, env)
            log_likelihoods[i_jet].append(log_likelihood)
            illegal_actions[i_jet].append(error)

        self._update_results(method, log_likelihoods, illegal_actions)

    def eval_random(self, method, env_name, n_repeats=100):
        self.eval(method, None, env_name, n_repeats)

    def get_results(self):
        for method in self.methods:
            yield method, self.log_likelihoods[method], self.illegal_actions[method]

    def __str__(self):
        lengths = 20, 6, 3

        results = []
        for method, log_likelihood, illegals in self.get_results():
            mean_log_likelihood = np.nanmean([np.nanmean(x) for x in log_likelihood])
            mean_illegals = np.nanmean([np.nanmean(x) for x in illegals])
            results.append((method, mean_log_likelihood, mean_illegals))

        lines = []
        lines.append("")
        lines.append("-"*(lengths[0] + lengths[1] + lengths[2] + (3-1)*3))
        lines.append(f"{'Method':>{lengths[0]}s} | {'Log p':>{lengths[1]}s} | {'Err':>{lengths[2]}s}")
        lines.append("-"*(lengths[0] + lengths[1] + lengths[2] + (3-1)*3))

        for method, mean_log_likelihood, mean_illegals in sorted(results, key=lambda x : x[1], reverse=True):
            lines.append(f"{method:>{lengths[0]}s} | {mean_log_likelihood:>{lengths[1]}.{lengths[1] - 4}f} | {mean_illegals:>{lengths[2]}.{lengths[2] - 2}f}")

        lines.append("-"*(lengths[0] + lengths[1] + lengths[2] + (3-1)*3))
        lines.append("")

        return "\n".join(lines)

    def plot_log_likelihoods(self, cols=2, rows=4, ymax=0.5, deltax_min=5., deltax_max=20., xbins=25, panelsize=4., filename=None):
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
        if filename is not None:
            plt.savefig(filename)
        return fig

    def _update_results(self, method, log_likelihoods, illegal_actions):
        self._load()  # Just in case another process changed the data in the file in the mean time

        while method in self.methods:
            self.methods.remove(method)
        self.methods.append(method)

        self.log_likelihoods[method] = log_likelihoods
        self.illegal_actions[method] = illegal_actions

        self._save()

    def _save(self):
        data = {"n_jets": self.n_jets, "jets": self.jets, "methods": self.methods, "log_likelihoods": self.log_likelihoods, "illegal_actions": self.illegal_actions}
        with open(self.filename, 'wb') as file:
            pickle.dump(data, file)

    def _load(self):
        with open(self.filename, 'rb') as file:
            data = pickle.load(file)

        self.n_jets = data["n_jets"]
        self.jets = data["jets"]
        self.methods = data["methods"]
        self.log_likelihoods = data["log_likelihoods"]
        self.illegal_actions = data["illegal_actions"]

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

    @staticmethod
    def _episode(model, env):
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

        return float(log_likelihood), int(errors)

    @staticmethod
    def _compute_true_log_likelihood(jet):
        return sum(jet[0]["logLH"])

    @staticmethod
    def _compute_maximum_log_likelihood(jet):
        """ Based on Sebastian's code at https://github.com/iesl/hierarchical-trellis/blob/sebastian/src/Jet_Experiments_invM_exactTrellis.ipynb """
        _, _, max_log_likelihood, _, _ = compute_trellis(jet[0])
        return max_log_likelihood