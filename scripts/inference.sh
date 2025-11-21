#!/bin/bash

#SBATCH --partition=hgnodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --time=01-23:59:59
#SBATCH --job-name=nlp_project
#SBATCH --mail-user=jjung2@uvm.edu
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

source ~/.bashrc
module load cuda/12.6.2
conda activate nlp_project

cd ${SLURM_SUBMIT_DIR}

model_id=$1
city=$2
mode=$3

python3 /gpfs3/scratch/jjung2/nlp_project/inference.py $model_id $city $mode