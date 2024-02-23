import os
import time
from glob import glob
import argparse
import torch
import wandb

from utils.tools import draw_eval_score
wandb.login()

from run_stationary import *
from utils import *

import warnings
warnings.filterwarnings('ignore')


torch.multiprocessing.set_sharing_strategy('file_system')

# def config():
#     """
#         This function is for parsing commandline arguments.
#     """
#     parser = argparse.ArgumentParser()
#     # mode parameters
#     parser.add_argument("--mode", type=str, default="train", help="the mode of the script, can be 'train' or 'test'")
#     parser.add_argument("--test_on", type=str, default="sct", help="the dataset for validation, can be 'cap' or 'sct'")
#     parser.add_argument("--control_mesh_dir", type=str,
#                         default="/home/yd21/Documents/Nasreddin/template/control_mesh-lv.obj",
#                         help="the path to your initial meshes")

#     # training parameters
#     parser.add_argument("--max_epochs", type=int, default=20, help="the maximum number of epochs for training")
#     parser.add_argument("--pretrain_epochs", type=int, default=6, help="the number of epochs to train the segmentation encoder")
#     parser.add_argument("--delay_epochs", type=int, default=8, help="the number of epochs to delay the fine-tuning and validation from start of pretrain")
#     parser.add_argument("--reduce_count_down", type=int, default=1, help="the count down for reduce the mesh face numbers.")
#     parser.add_argument("--val_interval", type=int, default=1, help="the interval of validation")

#     parser.add_argument("--lr", type=float, default=1e-3, help="the learning rate for training")
#     parser.add_argument("--batch_size", type=int, default=16, help="the batch size for training")
#     parser.add_argument("--cache_rate", type=float, default=1.0, help="the cache rate for training, see MONAI document for more details")
#     parser.add_argument("--crop_window_size", type=int, nargs='+', default=[128, 128, 128], help="the size of the crop window for training")
#     parser.add_argument("--pixdim", type=float, nargs='+', default=[8, 8, 8], help="the pixel dimension of downsampled images")
#     parser.add_argument("--lambda_", type=float, nargs='+', default=[0.1, 3.58, 6.3, 0.02], help="the loss coefficients for DF MSE, Chamfer verts distance, face squared distance, and laplacian smooth term")

#     # data parameters
#     parser.add_argument("--ct_json_dir", type=str,
#                         default="/home/yd21/Documents/Nasreddin/dataset/dataset_task20_f0.json", 
#                         help="the path to the json file with named list of CTA train/valid/test sets")
#     parser.add_argument("--mr_json_dir", type=str,
#                         default="/home/yd21/Documents/Nasreddin/dataset/dataset_task17_f0.json", 
#                         help="the path to the json file with named list of CMR train/valid/test sets")
#     parser.add_argument("--ct_data_dir", type=str, 
#                         default="/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset020_SCOTHEART", 
#                         help="the path to your processed images, must be in nifti format")
#     parser.add_argument("--mr_data_dir", type=str, 
#                         default="/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset017_CAP_COMBINED", 
#                         help="the path to your processed images")
#     parser.add_argument("--ckpt_dir", type=str, 
#                         default="/mnt/data/Experiment/Nasreddin/Checkpoint", 
#                         help="the path to your checkpoint directory, for holding trained models and wandb logs")
#     parser.add_argument("--out_dir", type=str, 
#                         default="/mnt/data/Experiment/Nasreddin/Result", 
#                         help="the path to your output directory, for saving outputs")
     
#     # path to the pretrained modules
#     parser.add_argument("--pretrained_pretext_mr_dir", type=str, default=None, help="the path to the pretrained pretext-mr")
#     parser.add_argument("--pretrained_ae_dir", type=str, default=None, help="the path to the pretrained autoencoder")

#     # structure parameters for df-predict module
#     parser.add_argument("--num_classes", type=int, default=4, help="the number of segmentation classes of foreground exclude background")
#     parser.add_argument("--channels", type=int, default=(16, 32, 64, 128, 256), nargs='+', help="the number of output channels in each layer of the encoder")
#     parser.add_argument("--strides", type=int, default=(2, 2, 2, 2), nargs='+', help="the stride of the convolutional layer in the encoder")
#     parser.add_argument("--layers", type=int, default=(1, 2, 2, 4), nargs='+', help="the number of layers in each residual block of the decoder")
#     parser.add_argument("--block_inplanes", type=int, default=(8, 16, 32, 64), nargs='+', help="the number of intermedium channels in each residual block")

#     # structure parameters for subdiv module
#     parser.add_argument("--subdiv_levels", type=int, default=2, help="the number of subdivision levels for the mesh")
#     parser.add_argument("--hidden_features_gsn", type=int, default=32, help="the number of hidden features for the graph subdivide network")

#     # run_id for wandb, will create automatically if not specified for training
#     parser.add_argument("--run_id", type=str, default=None, help="the run name for wandb and local machine")

#     # the best epoch for testing
#     parser.add_argument("--best_epoch", type=int, default=None, help="the best epoch for testing")

#     args = parser.parse_args()

#     return args

def train(config=None):

    def config():
        """
            This function is for parsing commandline arguments.
        """
        parser = argparse.ArgumentParser()
        # mode parameters
        parser.add_argument("--mode", type=str, default="train", help="the mode of the script, can be 'train' or 'test'")
        parser.add_argument("--test_on", type=str, default="sct", help="the dataset for validation, can be 'cap' or 'sct'")
        parser.add_argument("--control_mesh_dir", type=str,
                            default="/home/yd21/Documents/Nasreddin/template/control_mesh-lv.obj",
                            help="the path to your initial meshes")

        # training parameters
        parser.add_argument("--max_epochs", type=int, default=20, help="the maximum number of epochs for training")
        parser.add_argument("--pretrain_epochs", type=int, default=6, help="the number of epochs to train the segmentation encoder")
        parser.add_argument("--delay_epochs", type=int, default=8, help="the number of epochs to delay the fine-tuning and validation from start of pretrain")
        parser.add_argument("--reduce_count_down", type=int, default=1, help="the count down for reduce the mesh face numbers.")
        parser.add_argument("--val_interval", type=int, default=1, help="the interval of validation")

        parser.add_argument("--lr", type=float, default=1e-3, help="the learning rate for training")
        parser.add_argument("--batch_size", type=int, default=16, help="the batch size for training")
        parser.add_argument("--cache_rate", type=float, default=1.0, help="the cache rate for training, see MONAI document for more details")
        parser.add_argument("--crop_window_size", type=int, nargs='+', default=[128, 128, 128], help="the size of the crop window for training")
        parser.add_argument("--pixdim", type=float, nargs='+', default=[8, 8, 8], help="the pixel dimension of downsampled images")
        parser.add_argument("--lambda_", type=float, nargs='+', default=[0.1, 3.58, 6.3, 0.02], help="the loss coefficients for DF MSE, Chamfer verts distance, face squared distance, and laplacian smooth term")

        # data parameters
        parser.add_argument("--ct_json_dir", type=str,
                            default="/home/yd21/Documents/Nasreddin/dataset/dataset_task20_f0.json", 
                            help="the path to the json file with named list of CTA train/valid/test sets")
        parser.add_argument("--mr_json_dir", type=str,
                            default="/home/yd21/Documents/Nasreddin/dataset/dataset_task17_f0.json", 
                            help="the path to the json file with named list of CMR train/valid/test sets")
        parser.add_argument("--ct_data_dir", type=str, 
                            default="/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset020_SCOTHEART", 
                            help="the path to your processed images, must be in nifti format")
        parser.add_argument("--mr_data_dir", type=str, 
                            default="/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset017_CAP_COMBINED", 
                            help="the path to your processed images")
        parser.add_argument("--ckpt_dir", type=str, 
                            default="/mnt/data/Experiment/Nasreddin/Checkpoint", 
                            help="the path to your checkpoint directory, for holding trained models and wandb logs")
        parser.add_argument("--out_dir", type=str, 
                            default="/mnt/data/Experiment/Nasreddin/Result", 
                            help="the path to your output directory, for saving outputs")
        
        # path to the pretrained modules
        parser.add_argument("--pretrained_pretext_mr_dir", type=str, default=None, help="the path to the pretrained pretext-mr")
        parser.add_argument("--pretrained_ae_dir", type=str, default=None, help="the path to the pretrained autoencoder")

        # structure parameters for df-predict module
        parser.add_argument("--num_classes", type=int, default=4, help="the number of segmentation classes of foreground exclude background")
        parser.add_argument("--channels", type=int, default=(16, 32, 64, 128, 256), nargs='+', help="the number of output channels in each layer of the encoder")
        parser.add_argument("--strides", type=int, default=(2, 2, 2, 2), nargs='+', help="the stride of the convolutional layer in the encoder")
        parser.add_argument("--layers", type=int, default=(1, 2, 2, 4), nargs='+', help="the number of layers in each residual block of the decoder")
        parser.add_argument("--block_inplanes", type=int, default=(8, 16, 32, 64), nargs='+', help="the number of intermedium channels in each residual block")

        # structure parameters for subdiv module
        parser.add_argument("--subdiv_levels", type=int, default=2, help="the number of subdivision levels for the mesh")
        parser.add_argument("--hidden_features_gsn", type=int, default=32, help="the number of hidden features for the graph subdivide network")

        # run_id for wandb, will create automatically if not specified for training
        parser.add_argument("--run_id", type=str, default=None, help="the run name for wandb and local machine")

        # the best epoch for testing
        parser.add_argument("--best_epoch", type=int, default=None, help="the best epoch for testing")

        args = parser.parse_args()

        return args

    # # initialize the training pipeline
    # run_id = f"{time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))}"
    # super_params.run_id = f"{super_params.test_on}--" + \
    #     f"{os.path.basename(super_params.control_mesh_dir).split('-')[-1][:-4]}--" + \
    #         f"{os.path.basename(super_params.ct_json_dir).split('_')[-1][:-5]}--{run_id}"
    # wandb.init(project="Nasreddin_Stationary", name=super_params.run_id, config=super_params, mode="offline")

    super_params = config()
    
    with wandb.init(config=config, mode="offline"):
        agent_params = wandb.config

        # replace arguments in super_params if these arguments are specified in agent_params
        for key, value in agent_params.items():
            if hasattr(super_params, key):
                setattr(super_params, key, value)

        pipeline = TrainPipeline(
            super_params=super_params,
            seed=8, num_workers=0,
            )

        # train the network
        if super_params.test_on == "cap":
            for epoch in range(super_params.max_epochs):
                torch.cuda.empty_cache()
                # 1. train segmentation encoder
                pipeline.train_iter(epoch, "pretrain")
                if epoch > super_params.pretrain_epochs:
                    # 2. train whole network
                    pipeline.train_iter(epoch, "train")
                    if epoch > super_params.delay_epochs:
                        # 3. fine-tune the Subdiv Module
                        pipeline.fine_tune(epoch)
                        # 4. validate the pipeline
                        if epoch % super_params.val_interval == 0:
                            pipeline.valid(epoch, super_params.test_on)
        else:
            for epoch in range(super_params.max_epochs):
                torch.cuda.empty_cache()
                # 1. train segmentation encoder
                pipeline.train_iter(epoch, "pretrain")
                if epoch > super_params.pretrain_epochs:
                    # 2. train whole network
                    pipeline.train_iter(epoch, "train")
                    # 2.1 reduce the mesh face numbers
                    if epoch - super_params.pretrain_epochs == super_params.reduce_count_down:
                        pipeline.update_precomputed_faces()
                # 3. validate network
                if epoch > super_params.delay_epochs and \
                    (epoch - super_params.delay_epochs) % super_params.val_interval == 0:
                    pipeline.valid(epoch, super_params.test_on)


def test(**kwargs):
    # load MR images and labels (here used all cases from CAP)
    mr_image_paths = sorted(glob(f"{super_params.mr_image_dir}/*.nii.gz"))[-10:]
    mr_label_paths = sorted(glob(f"{super_params.mr_label_dir}/*.nii.gz"))[-10:]
    # check if images and labels are paired
    pairing_check(mr_image_paths, mr_label_paths)

    if super_params.run_id is None:
        if kwargs.get("run_id") is not None:
            super_params.run_id = kwargs.get("run_id")
        else:
            raise Exception(f"run_id is not specified")
    if super_params.best_epoch is None:
        if kwargs.get("best_epoch") is not None:
            super_params.best_epoch = kwargs.get("best_epoch")
        else:
            raise Exception(f"best_epoch is not specified")
    wandb.init(mode="disabled")
    pipeline = TrainPipeline(
        super_params=super_params,
        seed=2048, num_workers=0,
        data_bundle={
        "mr_image_paths": mr_image_paths, "mr_label_paths": mr_label_paths,
        },
        is_training=False
    )
    pipeline.test()


if __name__ == '__main__':
    sweep_config = {
        "method": "random",
        "metric": {"name": "eval_score", "goal": "maximize"},
        "parameters": {
            "lr": {"values": [1e-3, 1e-4, 1e-5]},
            "batch_size": {"values": [8, 16, 32]},
            "lambda_": {"values": [[0.1, 3.58, 6.3, 0.02], [0.1, 3.58, 6.3, 0.05], [0.1, 3.58, 6.3, 0.1]]},
            "hidden_features_gsn": {"values": [16, 32, 64]},
            "pretrain_epochs": {"values": [6, 8, 10]},
            "reduce_count_down": {"values": [3, 5, 7]},
            "delay_epochs": {"values": [12, 14, 16]},
            "val_interval": {"values": [2]},

            "test_on": {"values": ["cap"]},
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="Nasreddin_Stationary")

    wandb.agent(sweep_id, function=train, count=3)

    # super_params = config()
    # train(super_params)
