
#!/bin/bash
#SBATCH --nodes=1              # <-- Request 1 separate node
#SBATCH --partition=a100         # <-- Specify the a100 partition
#SBATCH --ntasks-per-node=1       # <-- 1 task on each node
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a100:1         # <-- 1 A100 GPU on each node
#SBATCH --job-name=vis_a100
#SBATCH -o vis_a100.out
#SBATCH --time=10:00:00   

eval "$(conda shell.bash hook)"
conda activate torch
python visualization.py
#ssh s4839921@rangpur.compute.eait.uq.edu.au 