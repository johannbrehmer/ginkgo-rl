# Hierarchical clustering in particle physics with reinforcement learning

## Getting started

- Environment with all package dependencies can be installed with `conda env create -f environment.yml`
- Ideally, also get a MongoDB to run and use OmniBoard to monitor experiments (though this is optional)
- For the Ginkgo simulator, install [ToyJetsShower](https://github.com/johannbrehmer/ToyJetsShower) (`pip install -e .` works, it's missing the `pyro-ppl` dependency though)
- For beam search and MLE estimates through the trellis, clone [ReclusterTreeAlgorithms](https://github.com/SebastianMacaluso/ReclusterTreeAlgorithms) and [hierarchical-trellis](https://github.com/iesl/hierarchical-trellis), and adapt the paths hard-coded in [evaluator.py](ginkgo_rl/eval/evaluator.py)


## Running experiments

To run individual experiments:
```
cd experiments

./experiment.py with truth         # Ground truth
./experiment.py with mle           # Trellis MLE
./experiment.py with greedy_s      # Greedy baseline
./experiment.py with beamsearch_s  # Beam search baseline
./experiment.py with mcts_s        # MCTS
./experiment.py with lfd_s         # Pure learning from demonstration
./experiment.py with lfd_mcts_s    # LfD policy in MCTS algorithm

# For a full list of configurations, see experiments/config.py
# Monitor results live e.g. with Omniboard
# Plot results with experiments/plot_results.ipynb
```

To automate the whole process on a SLURM HPC system:
```
cd experiments/hpc
sbatch --array 0-59 run.sh  # 5 runs (with different seeds) of each NN-based configuration
```
