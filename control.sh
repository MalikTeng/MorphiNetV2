#!/bin/bash

# ---- Run the training script ----

python main.py \
    --_4d \
    --save_on cap \
    \
    --mr_json_dir ./dataset/dataset_task10_f0.json \
    --mr_data_dir /mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset010_CAP_SAX_NRRD \
    \
    --subdiv_levels 2 \
    --control_mesh_dir ./template/template_mesh-myo.obj \
    --use_ckpt /mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/sct--myo--f0--2024-07-29-2042 \
    \
    --max_epochs 200 \
    --pretrain_epochs 100 \
    --train_epochs 150 \
    --val_interval 10 \
    \
    --hidden_features_gsn 64 \
    --pixdim 4 4 4 \
    --lambda_0 2.07 \
    --lambda_1 0.89 \
    --lambda_2 2.79 \
    --temperature 2.42 \
    \
    --lr 0.001 \
    --batch_size 1 \
    --mode online \

