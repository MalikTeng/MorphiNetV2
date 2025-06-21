#!/bin/bash

# MorphiNet Training Control Script - Modular Architecture
# This script launches the modular MorphiNet training pipeline

echo "================================================="
echo "MorphiNet Training - Modular Architecture"
echo "================================================="

# Default parameters - modify as needed
VALIDATION_MODALITY="ct"  # 'ct' for CT validation, 'mr' for MR validation
MR_JSON_DIR="./dataset/dataset_task11_f0.json"
MR_DATA_DIR="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX"
CT_JSON_DIR="./dataset/dataset_task20_f0.json"
CT_DATA_DIR="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset020_SCOTHEART"
TEMPLATE_MESH_DIR="./template/template_mesh-myo.obj"
CKPT_DIR="/mnt/data/Experiment/MorphiNet/Checkpoint/"

# Training parameters
MAX_EPOCHS=3
PRETRAIN_EPOCHS=1
TRAIN_EPOCHS=2
VAL_INTERVAL=1
BATCH_SIZE=1
LR=0.001
MAX_SAMPLES=2  # Set to 0 for full dataset, or positive number to limit samples for testing

# Model parameters
SUBDIV_LEVELS=2
HIDDEN_FEATURES_GSN=64
LAMBDA_0=2.07
LAMBDA_1=0.89
ITERATION=5

# Run the modular training
python main.py \
    --validation_modality $VALIDATION_MODALITY \
    --mr_json_dir $MR_JSON_DIR \
    --mr_data_dir $MR_DATA_DIR \
    --ct_json_dir $CT_JSON_DIR \
    --ct_data_dir $CT_DATA_DIR \
    --template_mesh_dir $TEMPLATE_MESH_DIR \
    --ckpt_dir $CKPT_DIR \
    --max_samples $MAX_SAMPLES \
    --max_epochs $MAX_EPOCHS \
    --pretrain_epochs $PRETRAIN_EPOCHS \
    --train_epochs $TRAIN_EPOCHS \
    --val_interval $VAL_INTERVAL \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --subdiv_levels $SUBDIV_LEVELS \
    --hidden_features_gsn $HIDDEN_FEATURES_GSN \
    --lambda_0 $LAMBDA_0 \
    --lambda_1 $LAMBDA_1 \
    --iteration $ITERATION \
    --mode online

echo "================================================="
echo "Training completed!"
echo "================================================="