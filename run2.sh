#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=ginkgo-rl
#SBATCH --mail-type=END
#SBATCH --mail-user=sm4511@nyu.edu
#SBATCH --output=logs/log_ginkgo_rl_run1_%a.log
#SBATCH --gres=gpu:1



dir=/scratch/sm4511/ginkgo-rl/experiments
seed=$((SLURM_ARRAY_TASK_ID + 1000))
setup=$((SLURM_ARRAY_TASK_ID / 5))

cd $dir
#source activate rl

singularity exec --nv \
        --overlay /scratch/sm4511/pytorch1.7.0-cuda11.0.ext3:ro \
        /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
        bash -c "source /ext3/env.sh; python -u $SCRATCH/ginkgo-rl/experiments/experiment.py with mcts_l seed=$seed database=False"


#case ${setup} in
#0) python -u experiment.py with mcts_xs "seed=$seed" "database=False";;
#1) python -u experiment.py with mcts_s "seed=$seed" "database=False";;
#2) python -u experiment.py with mcts_m "seed=$seed" "database=False";;
#3) python -u experiment.py with mcts_l "seed=$seed" "database=False";;
#4) python -u experiment.py with lfd "seed=$seed" "database=False";;
#5) python -u experiment.py with lfd_mcts_s "seed=$seed" "database=False";;
#6) python -u experiment.py with mcts_exploit "seed=$seed" "database=False";;
#7) python -u experiment.py with mcts_explore "seed=$seed" "database=False";;
#8) python -u experiment.py with mcts_raw "seed=$seed" "database=False";;
#9) python -u experiment.py with mcts_puct_decisions "seed=$seed" "database=False";;
#10) python -u experiment.py with mcts_no_beamsearch "seed=$seed" "database=False";;
#11) python -u experiment.py with mcts_random "seed=$seed" "database=False";;
#12) python -u experiment.py with random "seed=$seed" "database=False";;
#13) python -u experiment.py with lfd_mcts_xs "seed=$seed" "database=False";;
#14) python -u experiment.py with lfd_mcts_m "seed=$seed" "database=False";;
#15) python -u experiment.py with lfd_mcts_l "seed=$seed" "database=False";;
#16) python -u experiment.py with lfd_mleteacher "seed=$seed" "database=False";;
#17) python -u experiment.py with lfd_mcts_mleteacher_xs "seed=$seed" "database=False";;
#18) python -u experiment.py with lfd_mcts_mleteacher_s "seed=$seed" "database=False";;
#19) python -u experiment.py with lfd_mcts_mleteacher_m "seed=$seed" "database=False";;
#20) python -u experiment.py with lfd_mcts_mleteacher_l "seed=$seed" "database=False";;
#*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
#esac

## to submit(for 3 jobs): sbatch --array 0-2 submitHPC_HCmanager.s