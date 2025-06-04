import os, sys
sys.path.extend([
    os.path.join(os.path.dirname(__file__), "data"),
    os.path.join(os.path.dirname(__file__), "model"),
    os.path.join(os.path.dirname(__file__), "utils"),
])
import time
import glob
import argparse
import torch
import wandb

from utils.tools import draw_eval_score
wandb.login()

from run import TrainPipeline
from utils import *

import warnings
warnings.filterwarnings('ignore')

torch.multiprocessing.set_sharing_strategy('file_system')

def train(sweep_params=None):

    with wandb.init():
        cfg = wandb.config

        if hasattr(cfg, 'architecture') and isinstance(cfg.architecture, dict):
            for key, value in cfg.architecture.items():
                setattr(cfg, key, value)
        else:
            print("Warning: cfg.architecture not found or not a dict. Using fallback architecture params.")
            cfg.kernel_size = getattr(cfg, 'kernel_size', (3, 3, 3, 3, 3)) 
            cfg.strides = getattr(cfg, 'strides', (1, 2, 2, 2, 2))
            cfg.filters = getattr(cfg, 'filters', (8, 16, 32, 64, 128))

        pipeline = TrainPipeline(
            super_params=cfg,
            seed=8, num_workers=0,
            is_training=True
            )

        current_training_phase = "unet"
        pipeline._data_warper(rotation=False, training_phase=current_training_phase)

        has_unet_ckpt = False

        if cfg.use_ckpt is not None and cfg.use_ckpt.lower() != 'n':
            print(f"Loading pretrained weights from {cfg.use_ckpt}")
            base_ckpt_path = cfg.use_ckpt
            potential_trained_weights_path = os.path.join(base_ckpt_path, "trained_weights")
            
            if os.path.isfile(os.path.join(potential_trained_weights_path, "best_UNet_MR.pth")):
                actual_ckpt_dir = potential_trained_weights_path
            elif os.path.isfile(os.path.join(base_ckpt_path, "best_UNet_MR.pth")):
                actual_ckpt_dir = base_ckpt_path
            else:
                print(f"Warning: Could not find UNet weights in {base_ckpt_path} or {potential_trained_weights_path}. Check --use_ckpt path.")
                actual_ckpt_dir = None

            if actual_ckpt_dir:
                unet_mr_path = glob.glob(f"{actual_ckpt_dir}/best_UNet_MR.pth")
                unet_ct_path = glob.glob(f"{actual_ckpt_dir}/best_UNet_CT.pth")
                has_unet_ckpt = bool(unet_mr_path and unet_ct_path)
            
            pipeline.load_pretrained_weight("all")
        
        for epoch in range(cfg.max_epochs):
            torch.cuda.empty_cache()
            
            if epoch < cfg.pretrain_epochs:
                new_phase = "unet"
                if not has_unet_ckpt:
                    if current_training_phase != new_phase:
                        current_training_phase = new_phase
                        pipeline._data_warper(rotation=False, training_phase=current_training_phase)
                    pipeline.train_iter(epoch, "unet")
                else:
                    print(f"Skipping UNet training (epoch {epoch}) - using checkpoint")
            
            elif epoch < cfg.train_epochs:
                new_phase = "resnet"
                if current_training_phase != new_phase:
                    current_training_phase = new_phase
                    pipeline._data_warper(rotation=False, training_phase=current_training_phase)
                
                pipeline.train_iter(epoch, "resnet")
            
            else:
                new_phase = "gsn"
                if current_training_phase != new_phase:
                    current_training_phase = new_phase
                    pipeline._data_warper(rotation=False, training_phase=current_training_phase)

                will_validate_this_epoch = (epoch - cfg.train_epochs) % cfg.val_interval == 0
                pipeline.train_iter(epoch, "gsn", commit_log=not will_validate_this_epoch)
                
                if hasattr(cfg, 'reduce_count_down') and epoch - cfg.train_epochs == cfg.reduce_count_down:
                    pipeline.update_precomputed_faces()
                
                if will_validate_this_epoch:
                    pipeline.prepare_validation_specific_dataloaders(rotation=False)
                    pipeline.valid(epoch, cfg.save_on)
                    pipeline._clear_dataloader("ct" if cfg.save_on == "sct" else "mr", "valid")


if __name__ == '__main__':
    run_id_suffix = f"{time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))}"

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'eval_score',
            'goal': 'maximize'
        },
        'parameters': {
            'save_on': {'value': 'sct'},
            'ct_ratio': {'value': 1.0},
            '_mr': {'value': False},
            'template_mesh_dir': {'value': '/home/yd21/Documents/MorphiNet/template/template_mesh-lv_myo.obj'},
            'max_epochs': {'value': 200},
            'pretrain_epochs': {'value': 100},
            'train_epochs': {'value': 150},
            'val_interval': {'value': 10},
            'reduce_count_down': {'value': -1},
            'batch_size': {'value': 1},
            'cache_rate': {'value': 1.0},
            'crop_window_size': {'value': [128, 128, 128]},
            'pixdim': {'value': [4, 4, 4]},
            'iteration': {'value': 10},
            'ct_json_dir': {'value': '/home/yd21/Documents/MorphiNet/dataset/dataset_task20_f0.json'},
            'mr_json_dir': {'value': '/home/yd21/Documents/MorphiNet/dataset/dataset_task11_f0.json'},
            'ct_data_dir': {'value': '/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset020_SCOTHEART'},
            'mr_data_dir': {'value': '/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX'},
            'ckpt_dir': {'value': '/mnt/data/Experiment/MorphiNet/Checkpoint'},
            'use_ckpt': {'value': 'n'},
            'out_dir': {'value': '/mnt/data/Experiment/MorphiNet/Result'},
            'num_classes': {'value': 5},
            'subdiv_levels': {'value': 2},
            'mask_threshold': {'value': 0.1},
            'run_id': {'value': f"sct--myo--f0--sweep--{run_id_suffix}"},
            'lr': {'distribution': 'uniform', 'min': 1e-4, 'max': 1e-2},
            'lambda_0': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
            'lambda_1': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
            'architecture': {
                'values': [
                    {'kernel_size': (3,3,3), 'strides': (1,2,2), 'filters': (8,16,32)},
                    {'kernel_size': (3,3,3,3), 'strides': (1,2,2,2), 'filters': (8,16,32,64)},
                    {'kernel_size': (3,3,3,3,3), 'strides': (1,2,2,2,2), 'filters': (8,16,32,64,128)},
                    {'kernel_size': (3,3,3,3,3,3), 'strides': (1,2,2,2,2,2), 'filters': (8,16,32,64,128,256)}
                ]
            },
            'layers': {'values': [(1,2,2,4), (1,4,4,8), (1,8,8,16)]},
            'hidden_features_gsn': {'values': [8, 16, 32, 64]},
            'sigmoid_scale_factor': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="MorphiNet-sweeps")
    wandb.agent(sweep_id, function=train, count=10)
