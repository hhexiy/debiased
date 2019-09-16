#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=10GB
#SBATCH --output=scratch/slurm/slurm_%j.out

source activate gluonnlp
cd ~/projects/debiased
$command
