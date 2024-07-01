#!/bin/bash

# ---- Run the training script ----

python main.py \
--mode online \
--max_epochs 200 \
--pretrain_epochs 100 \
--train_epochs 120 \
--reduce_count_down -1 \
--val_interval 20 \
--batch_size 1 \
--lr 0.001 \
--pixdim 4 4 4 \
--lambda_ 10.0 10.0 0.5 \
--save_on cap \
--subdiv_levels 0 \
--control_mesh_dir /home/yd21/Documents/MorphiNet_Abdul/template/initial_mesh-myo.obj

