#!/usr/bin/env bash

#SBATCH --job-name=ginkgo-rl
#SBATCH --output=log_ginkgo_rl_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=5-00:00:00
# #SBATCH --gres=gpu:1

dir=/scratch/jb6504/ginkgo-rl/experiments
seed=$((SLURM_ARRAY_TASK_ID + 1000))
setup=$((SLURM_ARRAY_TASK_ID % 12))

cd $dir
source activate rl

case ${setup} in
0) python -u experiment.py with mcts_xs "seed=$seed";;
1) python -u experiment.py with mcts_s "seed=$seed";;
2) python -u experiment.py with mcts_m "seed=$seed";;
3) python -u experiment.py with mcts_l "seed=$seed";;
4) python -u experiment.py with lfd_s "seed=$seed";;
5) python -u experiment.py with lfd_mcts_s "seed=$seed";;
6) python -u experiment.py with mcts_exploit "seed=$seed";;
7) python -u experiment.py with mcts_explore "seed=$seed";;
8) python -u experiment.py with mcts_raw "seed=$seed";;
9) python -u experiment.py with mcts_puct_decisions "seed=$seed";;
10) python -u experiment.py with mcts_no_beamsearch "seed=$seed";;
11) python -u experiment.py with mcts_only_beamsearch "seed=$seed";;
*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
