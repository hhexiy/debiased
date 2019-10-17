#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=v100_pci_2,v100_sxm2_4,p100_4,p40_4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=10GB
#SBATCH --output=scratch/slurm/slurm_%j.out

source activate debiased
cd ~/projects/debiased
$command
