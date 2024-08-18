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

        pipeline = TrainPipeline(
            super_params=sweep_params,
            seed=8, num_workers=0,
            )

        if sweep_params.save_on == "cap" and sweep_params._4d.lower() == 'y':
            # refine 4D mesh with NDF
            pipeline.load_pretrained_weight("all")
            for epoch in range(sweep_params.max_epochs, sweep_params.max_epochs + 50):
                # 5. refine the 4D mesh with NDF
                pipeline.train_iter(epoch, "ndf")
                # 6. validate network
                if epoch % sweep_params.val_interval == 0:
                    pipeline.valid(epoch, sweep_params.save_on)

        else:
            # train the network
            CKPT = False
            for epoch in range(sweep_params.max_epochs):
                torch.cuda.empty_cache()
                if epoch < sweep_params.pretrain_epochs:
                    if sweep_params.use_ckpt is not None and CKPT is False:
                        pipeline.load_pretrained_weight("unet")
                        CKPT = True
                    elif CKPT is False:
                        # 1. train segmentation encoder
                        pipeline.train_iter(epoch, "unet")
                elif epoch < sweep_params.train_epochs:
                    # drop the rotation and flip augmentation
                    if epoch == sweep_params.pretrain_epochs:
                        pipeline._data_warper(rotation=False)
                    # 2. train distance field prediction module
                    pipeline.train_iter(epoch, "resnet")
                else:
                    # 3. fine-tune the subdiv module
                    pipeline.train_iter(epoch, "gsn")
                    # 3.1 reduce the mesh face numbers
                    if epoch - sweep_params.train_epochs == sweep_params.reduce_count_down:
                        pipeline.update_precomputed_faces()
                    # 4. validate network
                    if epoch >= sweep_params.train_epochs and \
                        (epoch - sweep_params.train_epochs) % sweep_params.val_interval == 0:
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
                'value': 120
            },
            'pretrain_epochs': {
                'value': 1
            },
            'train_epochs': {
                'value': 81
            },
            'val_interval': {
                'value': 10
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
                "distribution": "int_uniform",
                "min": 5,
                "max": 15
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
                'value': '/mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/sct--myo--f0--2024-08-10-2338'
            },
            'out_dir': {
                'value': '/mnt/data/Experiment/MorphiNet/Result'
            },
            'num_classes': {
                'value': 4
            },
            'kernel_size': {
                'value': [3, 3, 3, 3, 3]
            },
            'strides': {
                'value': [1, 2, 2, 2, 2]
            },
            'filters': {
                'value': [8, 16, 32, 64, 128]
            },
            'layers': {
                'value': [1, 2, 2, 4]
            },
            'block_inplanes': {
                'value': [8, 16, 32, 64]
            },
            'subdiv_levels': {
                'value': 2
            },
            'hidden_features_gsn': {
                'value': 16
            },
            'run_id': {
                'value': f"sct--myo--f0--{run_id}"
            },
        }
    }

    # initialise the sweep
    sweep_id = wandb.sweep(sweep_params, project="MorphiNet-sweeps")

    # run the sweep
    wandb.agent(sweep_id, function=train, count=20)
