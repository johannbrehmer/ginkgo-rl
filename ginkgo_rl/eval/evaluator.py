import gym
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from tqdm import trange
import pickle
import logging
import torch

from ginkgo_rl import GinkgoLikelihoodEnv, GinkgoLikelihood1DEnv

logger = logging.getLogger(__name__)

# Workaround for now until Trellis is better packaged
try:
    sys.path.append("/Users/johannbrehmer/work/projects/shower_rl/hierarchical-trellis/src")
    sys.path.append("/scratch/jb6504/hierarchical-trellis/src")
    from run_physics_experiment_invM import compare_map_gt_and_bs_trees as compute_trellis
except Exception:
    logger.warning("Error importing hierarchical trellis code.")
    compute_trellis = None

try:
    sys.path.insert(0, "/Users/johannbrehmer/work/projects/shower_rl/ReclusterTreeAlgorithms/scripts")
    sys.path.insert(0, "/scratch/jb6504/ReclusterTreeAlgorithms/scripts")
    sys.path.insert(0, "/scratch/jb6504/ReclusterTreeAlgorithms")
    import beamSearchOptimal_invM as beam_search
except Exception:
    logger.warning("Error importing beam search code.")
    beam_search = None


class GinkgoEvaluator:
    def __init__(self, filename, env, redraw_existing_jets=False, n_jets=100):
        self.filename = filename
        self.env = env

        self.methods = []  # Method names
        self.log_likelihoods = {}  # Log likelihood results
        self.illegal_actions = {}  # Number of illegal actions

        if os.path.exists(filename) and not redraw_existing_jets:
            self._load()
        else:
            self.n_jets = n_jets
            self.jets = self._init_jets()
            self._save()

    def eval_true(self, method):
        log_likelihoods = [[self._compute_true_log_likelihood(jet)] for jet in self.jets]
        illegal_actions = [[0] for _ in self.jets]
        likelihood_evaluations = [[0] for _ in self.jets]
        self._update_results(method, log_likelihoods, illegal_actions)
        return log_likelihoods, illegal_actions, likelihood_evaluations

    def eval_exact_trellis(self, method):
        log_likelihoods = [[self._compute_maximum_log_likelihood(jet)] for jet in self.jets]
        illegal_actions = [[0] for _ in self.jets]
        likelihood_evaluations = [[0] for _ in self.jets]
        self._update_results(method, log_likelihoods, illegal_actions)
        return log_likelihoods, illegal_actions, likelihood_evaluations

    def eval_beam_search(self, method, beam_size):
        log_likelihoods = [[self._compute_beam_search_log_likelihood(jet, beam_size)] for jet in self.jets]
        illegal_actions = [[0] for _ in self.jets]
        likelihood_evaluations = [
            [self._compute_beam_search_likelihood_evaluations(jet, beam_size)] for jet in self.jets
        ]
        self._update_results(method, log_likelihoods, illegal_actions)
        return log_likelihoods, illegal_actions, likelihood_evaluations

    def eval(self, method, model, n_repeats=1, mode=None):
        log_likelihoods = [[] for _ in range(self.n_jets)]
        illegal_actions = [[] for _ in range(self.n_jets)]
        likelihood_evaluations = [[] for _ in range(self.n_jets)]

        model.eval()

        for i in trange(len(self.jets) * n_repeats):
            i_jet = i // n_repeats
            jet = self.jets[i_jet]

            self.env.set_internal_state(jet)

            with torch.no_grad():
                log_likelihood, error, likelihood_evaluation = self._episode(model, mode=mode)

            log_likelihoods[i_jet].append(log_likelihood)
            illegal_actions[i_jet].append(error)
            likelihood_evaluations[i_jet].append(likelihood_evaluation)

        self._update_results(method, log_likelihoods, illegal_actions)
        return log_likelihoods, illegal_actions, likelihood_evaluations

    def eval_random(self, method, n_repeats=1):
        return self.eval(method, None, n_repeats)

    def get_jet_info(self):
        return {"n_leaves": np.array([len(jet[0]["leaves"]) for jet in self.jets], dtype=np.int)}

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
        lines.append("-" * (lengths[0] + lengths[1] + lengths[2] + (3 - 1) * 3))
        lines.append(f"{'Method':>{lengths[0]}s} | {'Log p':>{lengths[1]}s} | {'Err':>{lengths[2]}s}")
        lines.append("-" * (lengths[0] + lengths[1] + lengths[2] + (3 - 1) * 3))

        for method, mean_log_likelihood, mean_illegals in sorted(results, key=lambda x: x[1], reverse=True):
            lines.append(
                f"{method:>{lengths[0]}s} | {mean_log_likelihood:>{lengths[1]}.{lengths[1] - 4}f} | {mean_illegals:>{lengths[2]}.{lengths[2] - 2}f}"
            )

        lines.append("-" * (lengths[0] + lengths[1] + lengths[2] + (3 - 1) * 3))
        lines.append("")

        return "\n".join(lines)

    def plot_log_likelihoods(
        self,
        cols=2,
        rows=4,
        ymax=0.5,
        deltax_min=1.0,
        deltax_max=10.0,
        xbins=25,
        panelsize=4.0,
        filename=None,
        linestyles=["-", "--", ":", "-."],
        colors=[f"C{i}" for i in range(9)],
    ):
        colors = colors * 10
        linestyles = linestyles * 10
        fig = plt.figure(figsize=(rows * panelsize, cols * panelsize))

        for j in range(self.n_jets):
            if j > cols * rows:
                break

            plt.subplot(cols, rows, j + 1)

            xs = np.concatenate([logp[j] for logp in self.log_likelihoods.values()], axis=0)
            xmin, xmax = np.min(xs), np.max(xs)
            xmin = np.clip(xmin, xmax - deltax_max, xmax - deltax_min)
            xmax = xmax + 0.05 * (xmax - xmin)
            xmin = xmin - 0.05 * (xmax - xmin)

            ls_counter = 0
            for i, (name, logp, _) in enumerate(self.get_results()):
                logp_ = np.clip(logp, xmin + 1.0e-9, xmax - 1.0e-9)

                if len(logp[j]) == 1:
                    plt.plot(
                        [logp_[j][0], logp_[j][0]], [0.0, ymax], color=colors[i], ls=linestyles[ls_counter], label=name
                    )
                    ls_counter += 1
                else:
                    plt.hist(
                        logp_[j],
                        histtype="stepfilled",
                        range=(xmin, xmax),
                        color=colors[i],
                        bins=xbins,
                        lw=1.5,
                        density=True,
                        alpha=0.15,
                    )
                    plt.hist(
                        logp_[j],
                        histtype="step",
                        range=(xmin, xmax),
                        bins=xbins,
                        color=colors[i],
                        lw=1.5,
                        density=True,
                        label=name,
                    )

            if j == 0:
                plt.legend()

            plt.xlabel("Log likelihood")
            plt.ylabel("Histogram")

            plt.xlim(xmin, xmax)
            plt.ylim(0.0, ymax)

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        return fig

    def _update_results(self, method, log_likelihoods, illegal_actions):
        self.log_likelihoods[method] = log_likelihoods
        self.illegal_actions[method] = illegal_actions

    def _save(self):
        data = {"n_jets": self.n_jets, "jets": self.jets}
        with open(self.filename, "wb") as file:
            pickle.dump(data, file)

    def _load(self):
        with open(self.filename, "rb") as file:
            data = pickle.load(file)

        self.n_jets = data["n_jets"]
        self.jets = data["jets"]

    def _init_jets(self):
        logger.info("Generating evaluation jets")

        jets = []

        for _ in range(self.n_jets):
            self.env.reset()
            jets.append(self.env.get_internal_state())

        sizes = np.array([len(jet[0]["leaves"]) for jet in jets])
        logger.info(
            f"  Generated jets with min size {np.min(sizes)}, mean size {np.mean(sizes)}, max size {np.max(sizes)}"
        )

        return jets

    def _episode(self, model, mode=None):
        state = self.env.get_state()
        done = False
        log_likelihood = 0.0
        errors = 0
        reward = 0.0
        likelihood_evaluations = 0

        # Point agent to correct env and initialize episode: this only works for *our* models, not the baselines
        try:
            model.set_env(self.env)
            model.init_episode()
        except:
            pass

        while not done:
            if model is None:
                action = self.env.action_space.sample()
                agent_info = {}
            elif mode is None:
                action, agent_info = model.predict(state)
                likelihood_evaluations = max(agent_info["likelihood_evaluations"], likelihood_evaluations)
            else:
                action, agent_info = model.predict(state, mode=mode)
                likelihood_evaluations = max(agent_info["likelihood_evaluations"], likelihood_evaluations)

            next_state, next_reward, done, info = self.env.step(action)

            log_likelihood += next_reward
            if not info["legal"]:
                errors += 1

            # Update model: this only works for *our* models, not the baselines
            try:
                model.update(
                    state, reward, action, done, next_state, next_reward=next_reward, num_episode=0, **agent_info
                )
            except:
                pass

            reward, state = next_reward, next_state

        return float(log_likelihood), int(errors), int(likelihood_evaluations)

    @staticmethod
    def _compute_true_log_likelihood(jet):
        return sum(jet[0]["logLH"])

    @staticmethod
    def _compute_maximum_log_likelihood(jet, max_leaves=11):
        """ Based on Sebastian's code at https://github.com/iesl/hierarchical-trellis/blob/sebastian/src/Jet_Experiments_invM_exactTrellis.ipynb """
        if len(jet[0]["leaves"]) > max_leaves:
            return np.nan

        _, _, max_log_likelihood, _, _ = compute_trellis(jet[0])
        return max_log_likelihood

    @staticmethod
    def _compute_beam_search_log_likelihood(jet, beam_size):
        n = len(jet[0]["leaves"])
        bs_jet = beam_search.recluster(
            jet[0],
            beamSize=min(beam_size, n * (n - 1) // 2),
            delta_min=jet[0]["pt_cut"],
            lam=float(jet[0]["Lambda"]),
            N_best=1,
            visualize=True,
        )[0]
        return sum(bs_jet["logLH"])

    @staticmethod
    def _compute_beam_search_likelihood_evaluations(jet, beam_size):
        n = len(jet[0]["leaves"])
        beam = 1
        evaluations = 0

        while n > 1:
            evaluations += beam * n * (n - 1) // 2
            beam = beam_size
            n -= 1

        return evaluations
