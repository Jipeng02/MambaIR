(base) s4839921@login0:~$ cat vae_sbatch
#!/bin/bash
#SBATCH --nodes=1              # <-- Request 1 separate node
#SBATCH --partition=p100         # <-- Specify the p100 partition
#SBATCH --ntasks-per-node=1       # <-- 1 task on each node
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:p100:1         # <-- 1 P100 GPU on each node
#SBATCH --job-name=vae_p100
#SBATCH -o vae_p100.out
#SBATCH --time=10:00:00   

eval "$(conda shell.bash hook)"
conda activate torch

#ssh s4839921@rangpur.compute.eait.uq.edu.au 