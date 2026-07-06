#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --account=myproject
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=80G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load cuda/12.2
module load nccl/2.19

export NCCL_DEBUG=INFO
export NCCL_ALGO=Ring
export NCCL_IB_DISABLE=0

# torchrun needs one address to rendezvous on across the nodes Slurm allocated
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun --container-image=nvcr.io/nvidia/pytorch:24.01-py3 \
    --gpu-bind=none \
    torchrun \
        --nnodes="$SLURM_NNODES" \
        --nproc-per-node="$SLURM_GPUS_ON_NODE" \
        --node-rank="$SLURM_NODEID" \
        --rdzv-id="$SLURM_JOB_ID" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
        train.py --deepspeed ds_config.json
