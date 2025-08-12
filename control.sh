#!/bin/bash

# MorphiNet Training Control Script - Three-Phase Pipeline Test
# This script launches the modular MorphiNet training pipeline with refactored validation method names

# Activate the morphinet conda environment
source /mnt/data/Experiment/miniconda3/etc/profile.d/conda.sh
conda activate morphinet

# Default parameters - modify as needed
MR_JSON_DIR="./dataset/dataset_task11_f0.json"  # Using CAP dataset for testing
MR_DATA_DIR="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX"
CT_JSON_DIR="./dataset/dataset_task20_f0.json"
CT_DATA_DIR="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset020_SCOTHEART"
TEMPLATE_MESH_DIR="./template/template_mesh-myo.obj"
CKPT_DIR="/mnt/data/Experiment/MorphiNet/Checkpoint/"
USE_CKPT_DIR="/mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/ct--myo--f0--2025-08-07-1708/"

# Training parameters - Quick test run
PRETRAIN_EPOCHS=0  # UNet phase
TRAIN_EPOCHS=50  # ResNet phase
MAX_EPOCHS=100  # Total epochs for quick test
VAL_INTERVAL=10  # Validate every epoch to test all phases
BATCH_SIZE=1
LR=0.0001
MAX_SAMPLES=0  # Use only 2 samples for quick testing
CACHE_RATE=1.0
WANDB_MODE="online"  # Offline for quick testing

# Model parameters
KERNEL_SIZE="3 3 3 3 3"
STRIDE="1 2 1 2 2"

LAYERS="1 2 2 4"
LAMBDA_0=0.66
LAMBDA_1=0.75

MASK_THRESHOLD=0.12
SIGMOID_SCALE_FACTOR=0.83
HIDDEN_FEATURES_GSN=64

# Run the modular training
python main.py \
    --mr_json_dir $MR_JSON_DIR \
    --mr_data_dir $MR_DATA_DIR \
    --ct_json_dir $CT_JSON_DIR \
    --ct_data_dir $CT_DATA_DIR \
    --template_mesh_dir $TEMPLATE_MESH_DIR \
    --ckpt_dir $CKPT_DIR \
    --cache_rate $CACHE_RATE \
    --max_samples $MAX_SAMPLES \
    --max_epochs $MAX_EPOCHS \
    --pretrain_epochs $PRETRAIN_EPOCHS \
    --train_epochs $TRAIN_EPOCHS \
    --val_interval $VAL_INTERVAL \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --layers $LAYERS \
    --kernel_size $KERNEL_SIZE \
    --stride $STRIDE \
    --hidden_features_gsn $HIDDEN_FEATURES_GSN \
    --lambda_0 $LAMBDA_0 \
    --lambda_1 $LAMBDA_1 \
    --mask_threshold $MASK_THRESHOLD \
    --sigmoid_scale_factor $SIGMOID_SCALE_FACTOR \
    --mode $WANDB_MODE \
    --use_ckpt $USE_CKPT_DIR
    