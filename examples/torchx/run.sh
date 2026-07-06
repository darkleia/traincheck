#!/bin/bash
export NCCL_ALGO=Ring
export NCCL_IB_DISABLE=0

torchx run -s slurm dist.ddp -j 8x8 --script train.py -- --deepspeed ds_config.json
