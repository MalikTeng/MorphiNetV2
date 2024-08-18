#!/bin/bash

# ---- Run the training script with only mr data ----
python main.py \
    --save_on cap \
    \
    --mr_json_dir ./dataset/dataset_task11_f0.json \
    --mr_data_dir /mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX \
    \
    --template_mesh_dir ./template/template_mesh-myo.obj \
    --use_ckpt /mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/sct--myo--f0--2024-08-10-2338 \
    \
    --max_epochs 150 \
    --pretrain_epochs 1 \
    --train_epochs 81 \
    --val_interval 10 \
    \
    --hidden_features_gsn 16 \
    --pixdim 4 4 4 \
    --lambda_0 0.56 \
    --lambda_1 0.12 \
    --iteration 10 \
    \
    --lr 0.001 \
    --batch_size 1 \
    --mode online \
    --_mr

# ---- Run the training script with both ct & mr data----

CT_RATIO=(0.2 0.4 0.6 0.8)
for ratio in "${CT_RATIO[@]}"
do
    python main.py \
        --save_on sct \
        \
        --mr_json_dir ./dataset/dataset_task11_f0.json \
        --mr_data_dir /mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX \
        \
        --template_mesh_dir ./template/template_mesh-myo.obj \
        --use_ckpt /mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/sct--myo--f0--2024-08-10-2338 \
        \
        --max_epochs 150 \
        --pretrain_epochs 1 \
        --train_epochs 81 \
        --val_interval 10 \
        \
        --hidden_features_gsn 16 \
        --pixdim 4 4 4 \
        --lambda_0 0.56 \
        --lambda_1 0.12 \
        --iteration 10 \
        \
        --lr 0.001 \
        --batch_size 1 \
        --mode online \
        --ct_ratio "$ratio"
done

# # ---- Run the training script with both mr data for 4D creation----

# python main.py \
#     --save_on cap \
#     \
#     --mr_json_dir ./dataset/dataset_task10_f0.json \
#     --mr_data_dir /mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset010_CAP_SAX_NRRD \
#     \
#     --template_mesh_dir ./template/template_mesh-myo.obj \
#     --use_ckpt /mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/sct--myo--f0--2024-08-13-1838 \
#     \
#     --max_epochs 200 \
#     --pretrain_epochs 100 \
#     --train_epochs 150 \
#     --val_interval 10 \
#     \
#     --hidden_features_gsn 16 \
#     --pixdim 4 4 4 \
#     --lambda_0 1.06 \
#     --lambda_1 1.05 \
#     --iteration 5 \
#     \
#     --lr 0.001 \
#     --batch_size 1 \
#     --mode online \
#     --_4d \
#     --ct_ratio 1.0
