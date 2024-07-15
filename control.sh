#!/bin/bash

# ---- Run the training script ----

    # --use_ckpt /mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/cap--myo--f0--2024-07-10-0836/ \
python main.py \
    --save_on sct \
    --_4d n \
    --mr_json_dir /home/yd21/Documents/MorphiNet/dataset/dataset_task11_f0.json \
    --mr_data_dir /mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX \
    --subdiv_levels 2 \
    --control_mesh_dir /home/yd21/Documents/MorphiNet/template/template_mesh-myo.obj \
    --max_epochs 150 \
    --pretrain_epochs 100 \
    --train_epochs 110 \
    --reduce_count_down -1 \
    --val_interval 20 \
    --lr 0.001 \
    --pixdim 8 8 8 \
    --lambda_ 1.0 1.0 1.0 \
    --batch_size 1 \
    --mode online \

