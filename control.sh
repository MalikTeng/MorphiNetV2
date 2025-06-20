#!/bin/bash

# MorphiNet Training Control Script - Modular Architecture
# This script launches the modular MorphiNet training pipeline

echo "================================================="
echo "MorphiNet Training - Modular Architecture"
echo "================================================="

# Default parameters - modify as needed
SAVE_ON="sct"
MR_JSON_DIR="./dataset/dataset_task11_f0.json"
MR_DATA_DIR="/path/to/your/mr/data"
CT_JSON_DIR="./dataset/dataset_task20_f0.json"
CT_DATA_DIR="/path/to/your/ct/data"
TEMPLATE_MESH_DIR="./template/template_mesh-myo.obj"

# Training parameters
MAX_EPOCHS=10
PRETRAIN_EPOCHS=3
TRAIN_EPOCHS=5
VAL_INTERVAL=1
BATCH_SIZE=1
LR=0.001

# Model parameters
SUBDIV_LEVELS=2
HIDDEN_FEATURES_GSN=64
LAMBDA_0=2.07
LAMBDA_1=0.89
ITERATION=5

# Run the modular training
python main.py \
    --save_on $SAVE_ON \
    --mr_json_dir $MR_JSON_DIR \
    --mr_data_dir $MR_DATA_DIR \
    --ct_json_dir $CT_JSON_DIR \
    --ct_data_dir $CT_DATA_DIR \
    --template_mesh_dir $TEMPLATE_MESH_DIR \
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