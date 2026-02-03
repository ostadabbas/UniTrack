#!/bin/bash
#SBATCH -J train                            # Job name
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                                # Number of tasks
#SBATCH -o train_%j.txt                     # Standard output file
#SBATCH -e train_%j.txt                     # Standard error file
#SBATCH --gres=gpu:1                        # Request one GPU
#SBATCH -p journey_gpu                      # Partition name
#SBATCH --time=167:59:59
#SBATCH --mem=32G                           # Memory per node
#SBATCH --cpus-per-task=8

cd USER_HOME/aclab/CO-MOT 

# Properly initialize conda
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate WORK_DIR/comot

# Run the training script
./tools/train.sh configs/motrv2_mot17.args
# ./tools/train.sh configs/motrv2_mot20.args
# ./tools/train.sh configs/motrv2_dancetrack.args

echo "Training completed"
