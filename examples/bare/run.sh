#!/bin/bash
# Bare-metal multi-node launch - no scheduler, nodes are just a static hostfile.

set -euo pipefail

export NCCL_ALGO=Ring
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

NUM_NODES=$(wc -l < hostfile.txt)
MASTER_ADDR=$(head -n 1 hostfile.txt)
MASTER_PORT=29500

NODE_RANK=0
while read -r host; do
  ssh "$host" \
    "cd $(pwd) && torchrun \
      --nnodes=${NUM_NODES} \
      --nproc-per-node=8 \
      --node-rank=${NODE_RANK} \
      --master-addr=${MASTER_ADDR} \
      --master-port=${MASTER_PORT} \
      train.py --deepspeed ds_config.json" &
  NODE_RANK=$((NODE_RANK + 1))
done < hostfile.txt

wait
