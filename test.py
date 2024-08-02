import os, sys
import time
from glob import glob
import argparse
import torch
import wandb

from utils.tools import draw_eval_score
wandb.login()
from utils import *

import warnings
warnings.filterwarnings('ignore')


torch.multiprocessing.set_sharing_strategy('file_system')

def config():
    """
        This function is for parsing commandline arguments.
    """
    parser = argparse.ArgumentParser()
    # mode parameters
    parser.add_argument("--mode", type=str, default="offline", help="choose the mode for wandb, can be 'disabled', 'offline', 'online'")
    parser.add_argument("--_4d", action="store_true", help="toggle to train on 4D image data")
    parser.add_argument("--_mr", action="store_true", help="toggle to ONLY use MR data for training")
    parser.add_argument("--save_on", type=str, default="cap", help="the dataset for validation, can be 'cap' or 'sct'")
    parser.add_argument("--control_mesh_dir", type=str,
                        default="./template/template_mesh-myo.obj",
                        help="the path to your initial meshes")

    # training parameters
    parser.add_argument("--max_epochs", type=int, default=20, help="the maximum number of epochs for training")
    parser.add_argument("--pretrain_epochs", type=int, default=10, help="the number of epochs to train the segmentation UNet")
    parser.add_argument("--train_epochs", type=int, default=12, help="the number of epochs to train the distance field prediction ResNet")
    parser.add_argument("--reduce_count_down", type=int, default=-1, help="the count down for reduce the mesh face numbers.")
    parser.add_argument("--val_interval", type=int, default=1, help="the interval of validation")

    parser.add_argument("--lr", type=float, default=1e-3, help="the learning rate for training")
    parser.add_argument("--batch_size", type=int, default=1, help="the batch size for training")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="the cache rate for training, see MONAI document for more details")
    parser.add_argument("--crop_window_size", type=int, nargs='+', default=[128, 128, 128], help="the size of the crop window for training")
    parser.add_argument("--pixdim", type=float, nargs='+', default=[4, 4, 4], help="the pixel dimension of downsampled images")
    parser.add_argument("--lambda_0", type=float, default=2.29, help="the loss coefficients for Chamfer verts distance term")
    parser.add_argument("--lambda_1", type=float, default=0.57, help="the loss coefficients for point to mesh distance term")
    parser.add_argument("--lambda_2", type=float, default=1.41, help="the loss coefficients for laplacian smooth term")
    parser.add_argument("--temperature", type=float, default=1.66, help="the temperature for the distance field warping")

    # data parameters
    parser.add_argument("--ct_json_dir", type=str,
                        default="./dataset/dataset_task20_f0.json", 
                        help="the path to the json file with named list of CT train/valid/test sets")
    parser.add_argument("--mr_json_dir", type=str,
                        # default="./dataset/dataset_task11_f0.json",    # less data less burden
                        default="./dataset/dataset_task10_f0.json",  # use only for 4d
                        help="the path to the json file with named list of MR train/valid/test sets")
    parser.add_argument("--ct_data_dir", type=str, 
                        default="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset020_SCOTHEART", 
                        help="the path to your processed images, must be in nifti format")
    parser.add_argument("--mr_data_dir", type=str, 
                        # default="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX", 
                        default="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset010_CAP_SAX_NRRD", 
                        help="the path to your processed images")
    parser.add_argument("--ckpt_dir", type=str, 
                        default="/mnt/data/Experiment/MorphiNet/Checkpoint", 
                        help="the path to your checkpoint directory, for holding trained models and wandb logs")
    parser.add_argument("--out_dir", type=str, 
                        default="/mnt/data/Experiment/MorphiNet/Result", 
                        help="the path to your output directory, for saving outputs")
     
    # path to the pretrained modules
    parser.add_argument("--use_ckpt", type=str, 
                        # default=None,
                        default="/mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/sct--myo--f0--2024-07-30-1649/", 
                        help="the path to the pretrained models")

    # structure parameters for df-predict module
    parser.add_argument("--num_classes", type=int, default=4, help="the number of segmentation classes including the background")
    parser.add_argument("--kernel_size", type=int, default=(3, 3, 3, 3, 3), nargs='+', help="the kernel size of the convolutional layer in the encoder")
    parser.add_argument("--strides", type=int, default=(1, 2, 2, 2, 2), nargs='+', help="the stride of the convolutional layer in the encoder")
    parser.add_argument("--filters", type=int, default=(8, 16, 32, 64, 128), nargs='+', help="the number of output channels in each layer of the encoder")
    parser.add_argument("--layers", type=int, default=(1, 2, 2, 4), nargs='+', help="the number of layers in each residual block of the decoder")
    parser.add_argument("--block_inplanes", type=int, default=(8, 16, 32, 64), nargs='+', help="the number of intermedium channels in each residual block")

    # structure parameters for subdiv module
    parser.add_argument("--subdiv_levels", type=int, default=2, help="the number of subdivision levels for the mesh (should be an integer larger than 0, where 0 means no subdivision)")
    parser.add_argument("--hidden_features_gsn", type=int, default=16, help="the number of hidden features for the graph subdivide network")

    # run_id for wandb, will create automatically if not specified for training
    parser.add_argument("--run_id", type=str, default=None, help="the run name for wandb and local machine")

    # the best epoch for testing
    parser.add_argument("--best_epoch", type=int, default=None, help="the best epoch for testing")

    args = parser.parse_args()

    return args


def test(super_params):
    wandb.init(mode="disabled")
    pipeline = TrainPipeline(
        super_params=super_params,
        seed=42, num_workers=19,
        is_training=False
    )
    pipeline.test(super_params.save_on)


def ablation(super_params):
    wandb.init(mode="disabled")
    pipeline = TrainPipeline(
        super_params=super_params,
        seed=42, num_workers=19,
        is_training=False
    )
    pipeline.ablation_study(super_params.save_on)


if __name__ == '__main__':
    super_params = config()

    if super_params._mr:
        from run_mr import *
    else:
        from run import *

    # # checkpoint info
    # # super_params._4d = True
    # super_params.save_on = "cap"
    # ckpt = "cap--myo--f0--2024-07-31-2023"
    # super_params.best_epoch = 191

    # # data info
    # target = "ablation"
    # super_params.ct_json_dir = f"/home/yd21/Documents/MorphiNet/dataset/dataset_task20_f0.json"
    # super_params.ct_data_dir = f"/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset020_SCOTHEART"
    # super_params.mr_json_dir = f"/home/yd21/Documents/MorphiNet/dataset/dataset_task11_f0.json"
    # super_params.mr_data_dir = f"/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX"
    # # super_params.mr_data_dir = f"/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset021_ACDC"

    # # output info
    # super_params.out_dir = f"/mnt/data/Experiment/TMI_2024/{target}/MorphiNet/myo/f0/"

    # # model info
    # super_params.run_id = ckpt
    # super_params.ckpt_dir = f"/mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/{super_params.run_id}/trained_weights"
    # super_params.control_mesh_dir = f"/home/yd21/Documents/MorphiNet/template/template_mesh-myo.obj"

    test(super_params)

    # super_params.out_dir = f"/mnt/data/Experiment/TMI_2024/{target}/myo/"
    # ablation(super_params)
