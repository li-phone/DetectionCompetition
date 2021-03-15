#!/usr/bin/env bash

CONFIG="../configs/track/bs_r50_all_cat_ovlap_samp_x2_mst_dcn_anchor_k9_track.py"
GPUS=2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
