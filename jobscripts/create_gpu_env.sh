#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=R2D2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurm_output_%A.out

module purge
module load 2021
module load Anaconda3/2021.05

# Create an environment named "relfm-v1.0"
echo "Creating an environment named 'relfm-v1.0'..."
echo "----------------------------------------"
# get path to the repository
repository_path=$(git rev-parse --show-toplevel)
echo "repository_path: $repository_path"
cd $repository_path

# NOTE: This does not work well since some dependencies are pip
# and others are conda
# conda env create -f relfm-v1.0.yml

conda create -y -n relfm-v1.0 python=3.9

source deactivate
source activate relfm-v1.0

conda install -y tqdm pillow numpy matplotlib scipy
pip install ipdb ipython jupyter jupyterlab gdown termcolor natsort opencv-python
pip install torch==1.8.1+cu101 \
    torchvision==0.9.1+cu101 \
    torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install escnn wandb
echo "----------------------------------------"
