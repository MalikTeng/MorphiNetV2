#!/bin/bash

# ---- Run the training script ----

python main.py \
    --save_on cap \
    --unet_ckpt /mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/cap--myo--f0--2024-07-04-1423/ \
    --subdiv_levels 2 \
    --control_mesh_dir /home/yd21/Documents/MorphiNet/template/template_mesh-myo.obj \
    --max_epochs 200 \
    --pretrain_epochs 100 \
    --train_epochs 120 \
    --reduce_count_down -1 \
    --val_interval 20 \
    --lr 0.001 \
    --pixdim 4 4 4 \
    --lambda_ 10.0 0.02 0.2 \
    --batch_size 1 \
    --mode online \

