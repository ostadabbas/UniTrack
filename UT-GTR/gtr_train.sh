#!/bin/bash
#SBATCH -J gtr_train                            # Job name
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 4                               # Number of tasks
#SBATCH --cpus-per-task=4                     # Number of CPU cores per task
#SBATCH -o gtr_train_%j.txt                    # Standard output file
#SBATCH -e gtr_train_%j.txt                     # Standard error file
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1                          # Request 1 GPU
#SBATCH --partition=journey_gpu                     # Partition name
#SBATCH --mem=64G                             # Memory per node


cd WORK_DIR/
source env_mag/bin/activate
cd ../ICLR/UT-MOTR
python train_net.py --num-gpus 1 --config-file configs/GTR_Dance_UniTrack.yaml --eval-only