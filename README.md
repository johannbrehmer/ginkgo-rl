# Hierarchical clustering in particle physics with reinforcement learning

## Getting started

Need PyTorch + OpenAI gym + sacred with MongoDB backend

## Running experiments

```
cd experiments

./experiment.py with mcts_s        # MCTS
./experiment.py with lfd_s         # Pure learning from demonstration
./experiment.py with lfd_mcts_s    # LfD policy in MCTS algorithm
./experiment.py with greedy_s      # Greedy baseline
./experiment.py with beamsearch_s  # Beam search baseline
# For a full list of configurations, see experiments/config.py

# Monitor results live e.g. with Omniboard
# Plot results with experiments/plot_results.ipynb
```
