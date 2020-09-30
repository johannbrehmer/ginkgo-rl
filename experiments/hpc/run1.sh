#!/usr/bin/env bash

#SBATCH --job-name=ginkgo-rl
#SBATCH --output=log_ginkgo_rl_run1_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=5-00:00:00
# #SBATCH --gres=gpu:1

dir=/scratch/jb6504/ginkgo-rl/experiments
seed=$((SLURM_ARRAY_TASK_ID + 1000))
setup=$((SLURM_ARRAY_TASK_ID))

cd $dir
source activate rl

case ${setup} in
0) python -u experiment.py with truth "seed=$seed" "database=False";;
1) python -u experiment.py with mle "seed=$seed" "database=False";;
2) python -u experiment.py with random "seed=$seed" "database=False";;
3) python -u experiment.py with greedy "seed=$seed" "database=False";;
4) python -u experiment.py with beamsearch_s "seed=$seed" "database=False";;
5) python -u experiment.py with beamsearch_m "seed=$seed" "database=False";;
6) python -u experiment.py with beamsearch_l "seed=$seed" "database=False";;
7) python -u experiment.py with beamsearch_xl "seed=$seed" "database=False";;

8) python -u experiment.py with mcts_likelihood "seed=$seed" "database=False";;
9) python -u experiment.py with mcts_only_beamsearch "seed=$seed" "database=False";;

*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
