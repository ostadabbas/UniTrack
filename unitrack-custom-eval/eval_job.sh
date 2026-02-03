#!/bin/bash
#SBATCH -J train                            # Job name
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                                # Number of tasks
#SBATCH -o eval_%j.txt                     # Standard output file
#SBATCH -e eval_%j.txt                     # Standard error file
#SBATCH --gres=gpu:1                       # Request one GPU
#SBATCH -p journey_gpu                      # Partition name
#SBATCH --time=02:00:00
#SBATCH --mem=32G                           # Memory per node
#SBATCH --cpus-per-task=8

cd USER_HOME/aclab/CO-MOT 

# Properly initialize conda
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate WORK_DIR/comot

python evaluate_mot17_combined.py \
       --ckpt WORK_DIR/exps/motrv2_mot17/run1/exps/motrv2_mot17/run1/checkpoint0019.pth \
       --mot_root DATASET_ROOT/

echo "Evaluation completed"
