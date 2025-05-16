import os, sys
sys.path.extend([
    os.path.join(os.path.dirname(__file__), "data"),
    os.path.join(os.path.dirname(__file__), "model"),
    os.path.join(os.path.dirname(__file__), "utils"),
])
import time
from glob import glob
import argparse
import torch
import wandb

from utils.tools import draw_eval_score
wandb.login()

from run import *
from utils import *

import warnings
warnings.filterwarnings('ignore')


torch.multiprocessing.set_sharing_strategy('file_system')


def train(sweep_params=None):

    with wandb.init(config=sweep_params):

        sweep_params = wandb.config

        # Extract architecture parameters
        if 'architecture' in sweep_params:
            sweep_params.kernel_size = sweep_params.architecture['kernel_size']
            sweep_params.strides = sweep_params.architecture['strides']
            sweep_params.filters = sweep_params.architecture['filters']

        pipeline = TrainPipeline(
            super_params=sweep_params,
            seed=8, num_workers=0,
            )

        if sweep_params.save_on == "cap" and sweep_params._4d:
            # refine 4D mesh with NDF
            pipeline.load_pretrained_weight("all")
            for epoch in range(sweep_params.max_epochs, sweep_params.max_epochs + 50):
                # 5. refine the 4D mesh with NDF
                pipeline.train_iter(epoch, "ndf")
                # 6. validate network
                if epoch % sweep_params.val_interval == 0:
                    pipeline.valid(epoch, sweep_params.save_on)

        else:
            # Simplified training workflow
            # Check for existing checkpoints
            has_unet_ckpt = False
            if sweep_params.use_ckpt is not None:
                print(f"Loading pretrained weights from {sweep_params.use_ckpt}")
                ckpt_dir = f"{sweep_params.use_ckpt}/trained_weights"
                unet_mr_path = glob.glob(f"{ckpt_dir}/best_UNet_MR.pth")
                unet_ct_path = glob.glob(f"{ckpt_dir}/best_UNet_CT.pth")
                
                has_unet_ckpt = bool(unet_mr_path and unet_ct_path)
                
                # Load all available checkpoints
                pipeline.load_pretrained_weight("all")
            
            # train the network
            for epoch in range(sweep_params.max_epochs):
                torch.cuda.empty_cache()
                
                # Phase 1: UNet training (segmentation encoder)
                if epoch < sweep_params.pretrain_epochs:
                    if not has_unet_ckpt:
                        pipeline.train_iter(epoch, "unet")
                    else:
                        print(f"Skipping UNet training (epoch {epoch}) - using checkpoint")
                        # Jump to next phase
                        epoch = sweep_params.pretrain_epochs - 1
                
                # Phase 2: ResNet training (distance field prediction)
                elif epoch < sweep_params.train_epochs:
                    # First epoch of ResNet phase - ensure UNet weights are loaded
                    if epoch == sweep_params.pretrain_epochs:
                        pipeline.load_pretrained_weight("unet")
                    
                    # if not has_resnet_ckpt:
                    pipeline.train_iter(epoch, "resnet")
                
                # Phase 3: GSN training (graph subdivision) - always train this phase
                else:
                    # First epoch of GSN phase - ensure all weights are loaded
                    if epoch == sweep_params.train_epochs:
                        pipeline.load_pretrained_weight("all")
                    
                    # Always train the GSN module
                    pipeline.train_iter(epoch, "gsn")
                    
                    # Reduce mesh face numbers if needed
                    if epoch - sweep_params.train_epochs == sweep_params.reduce_count_down:
                        pipeline.update_precomputed_faces()
                    
                    # Validate network
                    if (epoch - sweep_params.train_epochs) % sweep_params.val_interval == 0:
                        pipeline.valid(epoch, sweep_params.save_on)


if __name__ == '__main__':
    # initialize the sweep parameters
    run_id = f"{time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))}"

    sweep_params = {
        'method': 'random',
        'metric': {
            'name': 'eval_score',
            'goal': 'maximize'
        },
        'parameters': {
            'save_on': {
                'value': 'sct'
            },
            'ct_ratio': {
                'value': 1.0
            },
            '_4d': {
                'value': False
            },
            'template_mesh_dir': {
                'value': '/home/yd21/Documents/MorphiNet/template/template_mesh-myo.obj'
            },
            'max_epochs': {
                'value': 200
            },
            'pretrain_epochs': {
                'value': 100
            },
            'train_epochs': {
                'value': 150
            },
            'val_interval': {
                'value': 5
            },
            'reduce_count_down': {
                'value': -1
            },
            'lr': {
                'value': 1e-3
            },
            'batch_size': {
                'value': 1
            },
            'cache_rate': {
                'value': 1.0
            },
            'crop_window_size': {
                'value': [128, 128, 128]
            },
            'pixdim': {
                'value': [4, 4, 4]
            },
            'lambda_0': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 1.0
            },
            'lambda_1': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 1.0
            },
            "iteration": {
                'value': 10
            },
            'ct_json_dir': {
                'value': '/home/yd21/Documents/MorphiNet/dataset/dataset_task20_f0.json'
            },
            'mr_json_dir': {
                'value': '/home/yd21/Documents/MorphiNet/dataset/dataset_task11_f0.json'
            },
            'ct_data_dir': {
                'value': '/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset020_SCOTHEART'
            },
            'mr_data_dir': {
                'value': '/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX'
            },
            'ckpt_dir': {
                'value': '/mnt/data/Experiment/MorphiNet/Checkpoint'
            },
            'use_ckpt': {
                'value': 'n',
            },
            'out_dir': {
                'value': '/mnt/data/Experiment/MorphiNet/Result'
            },
            'num_classes': {
                'value': 4
            },
            'architecture': {
                'values': [
                    {
                        'kernel_size': [3, 3, 3],
                        'strides': [1, 2, 2],
                        'filters': [8, 16, 32]
                    },
                    {
                        'kernel_size': [3, 3, 3, 3],
                        'strides': [1, 2, 2, 2],
                        'filters': [8, 16, 32, 64]
                    },
                    {
                        'kernel_size': [3, 3, 3, 3, 3],
                        'strides': [1, 2, 2, 2, 2],
                        'filters': [8, 16, 32, 64, 128]
                    },
                    {
                        'kernel_size': [3, 3, 3, 3, 3, 3],
                        'strides': [1, 2, 2, 2, 2, 2],
                        'filters': [8, 16, 32, 64, 128, 256]
                    }
                ]
            },
            'layers': {
                'values': [
                    [1, 2, 2, 4],
                    [1, 4, 4, 8],
                    [1, 8, 8, 16],
                ]
            },
            'subdiv_levels': {
                'value': 2
            },
            'hidden_features_gsn': {
                'values': [8, 16, 32, 64]
            },
            'run_id': {
                'value': f"sct--myo--f0--{run_id}"
            },
            'sigmoid_scale_factor': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 1.0
            },
            'mask_threshold': {
                'value': 0.1
            }
        }
    }

    # initialise the sweep
    sweep_id = wandb.sweep(sweep_params, project="MorphiNet-sweeps")

    # run the sweep
    wandb.agent(sweep_id, function=train, count=20)
