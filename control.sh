#!/bin/bash

# ---- Run the training script ----

    # --_4d \
python main.py \
    --save_on cap \
    \
    --mr_json_dir ./dataset/dataset_task11_f0.json \
    --mr_data_dir /mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX \
    \
    --control_mesh_dir ./template/template_mesh-myo.obj \
    --use_ckpt /mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/sct--myo--f0--2024-07-30-1649 \
    \
    --max_epochs 200 \
    --pretrain_epochs 100 \
    --train_epochs 150 \
    --val_interval 10 \
    \
    --hidden_features_gsn 16 \
    --pixdim 4 4 4 \
    --lambda_0 2.29 \
    --lambda_1 0.57 \
    --lambda_2 1.41 \
    --temperature 1.66 \
    \
    --lr 0.001 \
    --batch_size 1 \
    --mode online \

