# Hierarchical clustering in particle physics through reinforcement learning

[Johann Brehmer](johann.brehmer@nyu.edu), [Sebastian Macaluso](sm4511@nyu.edu),
[Duccio Pappadopulo](dpappadopulo@bloomberg.net), and [Kyle Cranmer](kyle.cranmer@nyu.edu)

[![arXiv](http://img.shields.io/badge/arXiv-arXiv:2011.08191-B31B1B.svg)](https://arxiv.org/abs/2011.08191)
[![ML4PS](http://img.shields.io/badge/ML4PS-2020-000000.svg)](https://ml4physicalsciences.github.io/2020/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Particle physics experiments often require the reconstruction of  decay patterns through a hierarchical clustering of the observed final-state particles. We show that this task can be phrased as a Markov Decision Process and adapt reinforcement learning algorithms to solve it. In particular, we show that Monte-Carlo Tree Search guided by a neural policy can construct high-quality hierarchical clusterings and outperform established greedy and beam search baselines.

Please see [our paper](https://arxiv.org/abs/2011.08191) for more details.

### Getting started

- A conda environment with all package dependencies can be installed with `conda env create -f environment.yml`
- Ideally, also get a MongoDB to run and use OmniBoard to monitor experiments (though this is optional)
- For the Ginkgo simulator, install [ToyJetsShower](https://github.com/johannbrehmer/ToyJetsShower) (`pip install -e .` works, it's missing the `pyro-ppl` dependency though)
- For beam search and MLE estimates through the trellis, clone [ReclusterTreeAlgorithms](https://github.com/SebastianMacaluso/ReclusterTreeAlgorithms) and [hierarchical-trellis](https://github.com/iesl/hierarchical-trellis), and adapt the paths hard-coded in [evaluator.py](ginkgo_rl/eval/evaluator.py)


### Running experiments

To run individual experiments:
```
cd experiments

./experiment.py with truth         # Ground truth
./experiment.py with mle           # Trellis MLE
./experiment.py with greedy_s      # Greedy baseline
./experiment.py with beamsearch_s  # Beam search baseline
./experiment.py with mcts_s        # MCTS
./experiment.py with lfd_s         # Pure learning from demonstration
./experiment.py with lfd_mcts_s    # BC policy in MCTS algorithm
```

For a full list of configurations, see the [experiments/config.py](experiments/config.py). 

To automate the whole process on a SLURM HPC system:
```
cd experiments/hpc
sbatch --array 0-59 run.sh
```

You can monitor the training live with Omniboard, and plot the final results with [experiments/plot_results.ipynb](experiments/plot_results.ipynb).
