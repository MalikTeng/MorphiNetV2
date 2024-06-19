#!/bin/bash

# ---- Run the training script ----
surfaces=("lv" "myo" "rv")
datasets=("sct" "cap")

for ((i=0; i<${#surfaces[@]}; i++)); do
    surface=${surfaces[$i]}
    for data in "${datasets[@]}"; do
        python main.py \
        --mode online \
        --max_epochs 200 \
        --pretrain_epochs 100 \
        --delay_epochs 100 \
        --reduce_count_down -1 \
        --val_interval 20 \
        --batch_size 8 \
        --lr 0.001 \
        --pixdim 8 8 8 \
        --lambda_ 0.02 10.0 10.0 0.25 \
        --save_on "$data" \
        --subdiv_levels 2 \
        --control_mesh_dir /home/yd21/Documents/MorphiNet/template/control_mesh-"$surface".obj
    done
done


# ---- Run the testing script ----
# fold=0
# parts=("lv" "myo" "rv")
# save_on=sct     # cap sct
# test_on=mmwhs     # cap sct mmwhs

# for part in "${parts[@]}"; do
#     python test.py \
#     --best_epoch 181 \
#     --ct_json_dir /home/yd21/Documents/MorphiNet/dataset/dataset_task22_f0.json \
#     --ct_data_dir /mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset022_MMWHS \
#     --mr_json_dir /home/yd21/Documents/MorphiNet/dataset/dataset_task17_f0.json \
#     --mr_data_dir /mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset017_CAP_COMBINED \
#     --out_dir /mnt/data/Experiment/MICCAI_24/"$test_on"/MorphiNet/"$part"/f0/ \
#     --save_on "$save_on" \
#     --run_id "$save_on"-"$part"-f0 \
#     --ckpt_dir /mnt/data/Experiment/MICCAI_24/Baselines/MorphiNet/"$save_on"-"$part"-f0/trained_weights \
#     --control_mesh_dir /home/yd21/Documents/MorphiNet/template/control_mesh-"$part".obj \
#     --pixdim 8 8 8
# done