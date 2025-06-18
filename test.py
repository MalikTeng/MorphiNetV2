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
    parser.add_argument("--save_on", type=str, default="sct", help="the dataset for validation, can be 'cap' or 'sct'")
    parser.add_argument("--target", type=str, default=None, help="the target dataset for test, particularly for a different process on acdc data")
    parser.add_argument("--template_mesh_dir", type=str,
                        default="./template/template_mesh-lv_myo.obj",
                        help="the path to your initial meshes")

    # training parameters
    parser.add_argument("--max_epochs", type=int, default=200, help="the maximum number of epochs for training")
    parser.add_argument("--pretrain_epochs", type=int, default=100, help="the number of epochs to train the segmentation UNet")
    parser.add_argument("--train_epochs", type=int, default=150, help="the number of epochs to train the distance field prediction ResNet")
    parser.add_argument("--reduce_count_down", type=int, default=-1, help="the count down for reduce the mesh face numbers.")
    parser.add_argument("--val_interval", type=int, default=5, help="the interval of validation")

    parser.add_argument("--lr", type=float, default=1e-3, help="the learning rate for training")
    parser.add_argument("--batch_size", type=int, default=1, help="the batch size for training")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="the cache rate for training, see MONAI document for more details")
    parser.add_argument("--crop_window_size", type=int, nargs='+', default=[128, 128, 128], help="the size of the crop window for training")
    parser.add_argument("--pixdim", type=float, nargs='+', default=[4, 4, 4], help="the pixel dimension of downsampled images")
    parser.add_argument("--lambda_0", type=float, default=0.24, help="the loss coefficients for Chamfer verts distance term")
    parser.add_argument("--lambda_1", type=float, default=0.63, help="the loss coefficients for point to mesh distance term")
    parser.add_argument("--iteration", type=int, default=10, help="the iterations for the distance field warping")
    parser.add_argument("--sigmoid_scale_factor", type=float, default=0.55, help="the scale factor for the sigmoid mask transition")
    parser.add_argument("--mask_threshold", type=float, default=0.1, help="the threshold for applying the distance map mask")

    # data parameters
    parser.add_argument("--ct_ratio", type=float, default=1.0, help="the portion of CT data for training")
    parser.add_argument("--ct_json_dir", type=str,
                        default="./dataset/dataset_task20_f0.json", 
                        help="the path to the json file with named list of CT train/valid/test sets")
    parser.add_argument("--mr_json_dir", type=str,
                        default="./dataset/dataset_task11_f0.json",    # less data less burden
                        # default="./dataset/dataset_task10_f0.json",  # use only for 4d
                        help="the path to the json file with named list of MR train/valid/test sets")
    parser.add_argument("--ct_data_dir", type=str, 
                        default="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset020_SCOTHEART", 
                        help="the path to your processed images, must be in nifti format")
    parser.add_argument("--mr_data_dir", type=str, 
                        default="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX", 
                        # default="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset010_CAP_SAX_NRRD", 
                        help="the path to your processed images")
    parser.add_argument("--ckpt_dir", type=str, 
                        default="/mnt/data/Experiment/MorphiNet/Checkpoint", 
                        help="the path to your checkpoint directory, for holding trained models and wandb logs")
    parser.add_argument("--out_dir", type=str, 
                        default="/mnt/data/Experiment/MorphiNet/Result", 
                        help="the path to your output directory, for saving outputs")
     
    # path to the pretrained modules
    parser.add_argument("--use_ckpt", type=lambda x: None if x.lower() == 'n' else x, 
                        default=None, 
                        # default="/mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/sct--myo--f0--2025-05-01-0949", 
                        help="path to pretrained models ('n' for no checkpoint, or specify a path)")

    # structure parameters for df-predict module
    parser.add_argument("--num_classes", type=int, default=4, help="the number of segmentation classes for LV-only template (background, LV, LV/RV-MYO, RV)")
    parser.add_argument("--kernel_size", type=int, default=(3, 3, 3), nargs='+', help="the kernel size of the convolutional layer in the encoder")
    parser.add_argument("--strides", type=int, default=(1, 2, 2), nargs='+', help="the stride of the convolutional layer in the encoder")
    parser.add_argument("--filters", type=int, default=(8, 16, 32), nargs='+', help="the number of output channels in each layer of the encoder")
    parser.add_argument("--layers", type=int, default=(1, 2, 2, 4), nargs='+', help="the number of layers in each residual block of the decoder")

    # structure parameters for subdiv module
    parser.add_argument("--subdiv_levels", type=int, default=2, help="the number of subdivision levels for the mesh (should be an integer larger than 0, where 0 means no subdivision)")
    parser.add_argument("--hidden_features_gsn", type=int, default=8, help="the number of hidden features for the graph subdivide network")

    # run_id for wandb, will create automatically if not specified for training
    parser.add_argument("--run_id", type=str, default=None, help="the run name for wandb and local machine")

    # the best epoch for testing
    parser.add_argument("--best_epoch", type=int, default=None, help="the best epoch for testing")

    args = parser.parse_args()

    return args


def test(super_params):
    # Make sure results directory exists
    os.makedirs(super_params.out_dir, exist_ok=True)
    print(f"Created results directory at {super_params.out_dir}")

    wandb.init(config=super_params, mode="offline", project="MorphiNet-test", name=super_params.run_id.replace("sct", super_params.target) if super_params.run_id else f"{super_params.target}-test")
    pipeline = TrainPipeline(
        super_params=super_params,
        seed=42, num_workers=0,
        is_training=False,
        target="acdc" if super_params.target == "acdc" else None
    )
    pipeline.prepare_all_dataloaders(data_types=["test"], validation_phase="network")
    pipeline.test(super_params.save_on)

    # Note: The test() method now includes post-processing of subdiv_mesh
    # The post_process_subdiv_mesh() method:
    # 1) Selects LV-MYO (label 2) and LV-ENDO (label 0) nodes from the subdivided mesh
    # 2) Creates convex hulls from each group of nodes individually  
    # 3) Subtracts LV-ENDO convex hull from LV-MYO convex hull to create refined LV-MYO mesh
    # This results in a more anatomically accurate representation of the left ventricular myocardium


def ablation(super_params):
    # Make sure results directory exists
    os.makedirs(super_params.out_dir, exist_ok=True)
    print(f"Created results directory at {super_params.out_dir}")

    wandb.init(mode="disabled")
    pipeline = TrainPipeline(
        super_params=super_params,
        seed=42, num_workers=0,
        is_training=False
    )
    # Use the new unified function to prepare test dataloaders
    pipeline.prepare_all_dataloaders(data_types=["test"], validation_phase="network")
    pipeline.ablation_study(super_params.save_on)


if __name__ == '__main__':
    super_params = config()

    super_params._mr = False

    from run import *
    
    # Network architecture parameters
    super_params.filters = (8, 16, 32, 64, 128)
    super_params.kernel_size = (3, 3, 3, 3, 3)
    super_params.strides = (1, 2, 2, 2, 2)
    super_params.layers = (1, 2, 2, 4)
    super_params.hidden_features_gsn = 64
    super_params.pixdim = [4, 4, 4]
    
    # Training parameters
    super_params.max_epochs = 200
    super_params.pretrain_epochs = 100
    super_params.train_epochs = 150
    super_params.val_interval = 5
    super_params.lambda_0 = 0.86
    super_params.lambda_1 = 0.75
    super_params.iteration = 20
    super_params.sigmoid_scale_factor = 0.19
    super_params.lr = 0.001
    super_params.batch_size = 1
    super_params.ct_ratio = 1.0

    # Dataset used for training, 'cap' -> mr data and 'sct' -> ct data
    super_params.save_on = "sct"
    
    # Data paths
    super_params.mr_json_dir = f"./dataset/dataset_task11_f0.json"
    super_params.mr_data_dir = f"/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX"
    super_params.template_mesh_dir = f"./template/template_mesh-myo.obj"

    # Test-specific settings
    ckpt = "sct--myo--f0--2025-06-08-0420"
    super_params.best_epoch = "best"
    super_params.target = "sct"
    super_params.ct_json_dir = f"/home/yd21/Documents/MorphiNet/dataset/dataset_task20_f0.json"
    super_params.ct_data_dir = f"/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset020_SCOTHEART"

    # Output paths
    super_params.run_id = ckpt
    super_params.ckpt_dir = f"/mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/{ckpt}/trained_weights"
    super_params.out_dir = f"/mnt/data/Experiment/TMI_2025/{super_params.target}/MorphiNet/myo/f0/"
    test(super_params)

    # super_params.out_dir = f"/mnt/data/Experiment/TMI_2025/{super_params.target}/"
    # ablation(super_params)
