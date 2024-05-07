#!/bin/bash

#SBATCH --exclusive --gres=gpu:volta:2 -o logs/training_and_compilation.log-%j

# Load the required modules
source /etc/profile
module load anaconda/Python-ML-2023b

# Activate venv
source ../venv_embeddings/bin/activate

# Run training
python training.py
# Run the compilation
python compilation.py