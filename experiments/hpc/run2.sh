#!/usr/bin/env bash

#SBATCH --job-name=ginkgo-rl
#SBATCH --output=log_ginkgo_rl_run2_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=5-00:00:00
# #SBATCH --gres=gpu:1

dir=/scratch/jb6504/ginkgo-rl/experiments
seed=$((SLURM_ARRAY_TASK_ID + 1000))
setup=$((SLURM_ARRAY_TASK_ID / 5))

cd $dir
source activate rl

case ${setup} in
0) python -u experiment.py with mcts_xs "seed=$seed" "database=False";;
1) python -u experiment.py with mcts_s "seed=$seed" "database=False";;
2) python -u experiment.py with mcts_m "seed=$seed" "database=False";;
3) python -u experiment.py with mcts_l "seed=$seed" "database=False";;
4) python -u experiment.py with lfd_s "seed=$seed" "database=False";;
5) python -u experiment.py with lfd_mcts_s "seed=$seed" "database=False";;
6) python -u experiment.py with mcts_exploit "seed=$seed" "database=False";;
7) python -u experiment.py with mcts_explore "seed=$seed" "database=False";;
8) python -u experiment.py with mcts_raw "seed=$seed" "database=False";;
9) python -u experiment.py with mcts_puct_decisions "seed=$seed" "database=False";;
10) python -u experiment.py with mcts_no_beamsearch "seed=$seed" "database=False";;
11) python -u experiment.py with mcts_random "seed=$seed" "database=False";;
*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
