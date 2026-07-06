#!/bin/bash
torchx run -s slurm dist.ddp -j 8x8 --script train.py -- --deepspeed ds_config.json
