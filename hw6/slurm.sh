#!/bin/bash

#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00

export NNODES=2
export GPUS_PER_NODE=8

export head_node=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
export head_node_ip=$(srun --nodes=1 --ntasks=1 --gpus=0 -w "$head_node" hostname --ip-address)

export HF_HOME=/scratch/s02210660/workdir/data/.cashe/

echo Head Node: $head_node
echo Head Node IP: $head_node_ip
echo "${head_node_ip: -1}"
srun --container-image /scratch/s02210660/practical_llm/ngc_cuda_pytorch_24_04_v1+latest.sqsh --container-workdir /scratch/s02210660/practical_llm --container-mounts /scratch/s02210660/practical_llm:/scratch/s02210660/practical_llm bash -c "cd /scratch/s02210660/practical_llm/workdir/ && ./run_train_ruadapt.sh"