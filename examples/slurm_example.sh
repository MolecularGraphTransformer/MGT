#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=mgt_train

# load modules and environments
module load module_name # where module_name is the name of the python module used to activate your environment
source env_path/bin/activate # where env_path is the path of your environment (if using conda: source activate env_path)

# for training
srun python training.py --root ./examples/example_data/ --model_path ./saved_models/ --train_split 0.8 --val_split 0.2

# for inference
srun python testing.py --root ./examples/example_data/ --model_path ./saved_models/

# for running on new data
srun python run.py --root ./examples/example_data/ --model_path ./saved_models/