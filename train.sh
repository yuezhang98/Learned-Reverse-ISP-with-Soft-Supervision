#!/usr/bin/env bash

CONFIG=$1

python -m torch.distributed.run --nproc_per_node=8 \
            --master_port=2508 basicsr/train.py -opt $CONFIG --launcher pytorch