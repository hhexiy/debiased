#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=p40_4,p100_4,v100_pci_2,v100_sxm2_4,k80_8,p1080_4,k80_4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=10GB
#SBATCH --output=scratch/slurm/slurm_%j.out

source activate gluonnlp
cd ~/projects/debiased
$command
