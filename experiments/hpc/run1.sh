#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00
#SBATCH --mem=16GB
#SBATCH --job-name=ginkgo-rl
#SBATCH --mail-type=END
#SBATCH --mail-user=sm4511@nyu.edu
#SBATCH --output=logs/log_ginkgo_rl_run1_%a.log
#SBATCH --gres=gpu:1


##SBATCH --time=5-00:00:00


dir=/scratch/sm4511/ginkgo-rl/experiments
seed=$((SLURM_ARRAY_TASK_ID + 1000))
setup=$((SLURM_ARRAY_TASK_ID))

cd $dir
source activate rl

case ${setup} in
0) python -u experiment.py with truth "seed=$seed" "database=False";;
1) python -u experiment.py with mle "seed=$seed" "database=False";;
2) python -u experiment.py with greedy "seed=$seed" "database=False";;
3) python -u experiment.py with beamsearch_s "seed=$seed" "database=False";;
4) python -u experiment.py with beamsearch_m "seed=$seed" "database=False";;
5) python -u experiment.py with beamsearch_l "seed=$seed" "database=False";;
6) python -u experiment.py with beamsearch_xl "seed=$seed" "database=False";;
7) python -u experiment.py with mcts_likelihood "seed=$seed" "database=False";;
8) python -u experiment.py with mcts_only_beamsearch "seed=$seed" "database=False";;
*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
