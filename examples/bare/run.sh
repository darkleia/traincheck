#!/bin/bash
# Bare-metal multi-node launch - no scheduler. Copy this same script onto
# every node listed in hostfile.txt and run it there; each node works out
# its own rank from its position in the file.

set -euo pipefail

export NCCL_ALGO=Ring
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

MASTER_ADDR=$(head -n 1 hostfile.txt)
MASTER_PORT=29500
NODE_RANK=$(($(grep -n "$(hostname)" hostfile.txt | cut -d: -f1) - 1))

torchrun \
  --nnodes=8 \
  --nproc-per-node=8 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  train.py --deepspeed ds_config.json
