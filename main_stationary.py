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

def config():
    """
        This function is for parsing commandline arguments.
    """
    parser = argparse.ArgumentParser()
    # mode parameters
    parser.add_argument("--mode", type=str, default="train", help="the mode of the script, can be 'train' or 'test'")

    # data parameters
    parser.add_argument("--test_on", type=str, default="sct", help="the dataset for validation, can be 'cap' or 'sct'")
    parser.add_argument("--control_mesh_dir", type=str,
                        default="/home/yd21/Documents/Nasreddin/template/control_mesh-cap_myo.obj",
                        # default="/home/yd21/Documents/Nasreddin/template/control_mesh-sct_myo.obj",
                        help="the path to your initial meshes")

    parser.add_argument("--ct_json_dir", type=str,
                        default="/home/yd21/Documents/Nasreddin/dataset/dataset_task20_f0.json", 
                        help="the path to the json file with named list of CTA train/valid/test sets")
    parser.add_argument("--ct_data_dir", type=str, 
                        default="/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset020_SCOTHEART", 
                        help="the path to your processed images, must be in nifti format")
    parser.add_argument("--mr_json_dir", type=str,
                        default="/home/yd21/Documents/Nasreddin/dataset/dataset_task17_f0.json", 
                        help="the path to the json file with named list of CMR train/valid/test sets")
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
    parser.add_argument("--pre_trained_sdf_module_dir", type=str, default=None, help="the path to the pretrained sdf-predict module")
    parser.add_argument("--pre_trained_mr_module_dir", type=str, default=None, help="the path to the pretrained subdiv module")

    # training parameters
    parser.add_argument("--max_epochs", type=int, default=10, help="the maximum number of epochs for training")
    parser.add_argument("--delay_epochs", type=int, default=5, help="the number of epochs for pre-training")
    parser.add_argument("--val_interval", type=int, default=1, help="the interval of validation")

    parser.add_argument("--batch_size", type=int, default=16, help="the batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="the learning rate for training")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="the cache rate for training, see MONAI document for more details")
    parser.add_argument("--crop_window_size", type=int, nargs='+', default=[128, 128, 128], help="the size of the crop window for training")
    parser.add_argument("--pixdim", type=float, nargs='+', default=[8, 8, 8], help="the pixel dimension of downsampled images")
    parser.add_argument("--point_limit", type=int, default=10_000, help="the number limits of sampling points during deformation")
    parser.add_argument("--lambda_", type=float, nargs='+', default=[1.0, 0.1, 0.1], help="the coefficients of chamfer distance, normal consistence, and laplacian smooth loss")

    # structure parameters for sdf predict module
    parser.add_argument("--num_classes", type=int, default=4, help="the number of segmentation classes of foreground exclude background")
    parser.add_argument("--init_filters", type=int, default=8, help="the number of initial filters for the modality handel")
    parser.add_argument("--num_init_blocks", type=int, nargs='+', default=(1, 2, 2, 4), help="the number of residual blocks for the modality handel")
    parser.add_argument("--hidden_features_sdf", type=int, default=256, help="the number of hidden features for the SDFNet decoder")

    # structure parameters for subdiv module
    parser.add_argument("--subdiv_levels", type=int, default=2, help="the number of subdivision levels for the mesh")
    parser.add_argument("--hidden_features_gsn", type=int, default=256, help="the number of hidden features for the GSNNet decoder")

    # run_id for wandb, will create automatically if not specified for training
    parser.add_argument("--run_id", type=str, default=None, help="the run name for wandb and local machine")

    # the best epoch for testing
    parser.add_argument("--best_epoch", type=int, default=None, help="the best epoch for testing")

    args = parser.parse_args()

    return args

def train():
    # initialize the training pipeline
    run_id = f"{time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))}"
    super_params.run_id = run_id
    wandb.init(project="Nasreddin_Stationary", name=run_id, config=super_params, mode="disabled")
    pipeline = TrainPipeline(
        super_params=super_params,
        seed=2077, num_workers=0,
    )

    # train the network
    for epoch in range(super_params.max_epochs):
        torch.cuda.empty_cache()
        # 1. train the SDF Module
        if (super_params.pre_trained_sdf_module_dir is None and super_params.pre_trained_mr_module_dir is None) \
        and epoch < super_params.delay_epochs:
            pipeline.train_iter(epoch, "sdf_predict")
            torch.cuda.empty_cache()
            continue

        # 2. train the GSN Module
        # pipeline.lr_scheduler_sdf._reset()
        pipeline.train_iter(epoch, "subdiv")
        torch.cuda.empty_cache()

        # 3. fine-tune the GSN Module if validation is on CAP # TODO: add delay_epochs for fine-tuning
        if super_params.test_on == "cap":
            # pipeline.lr_scheduler_gsn._reset()
            pipeline.fine_tune(epoch)
            torch.cuda.empty_cache()

        # 4. validate the network
        if (epoch - super_params.delay_epochs) % super_params.val_interval == 0:
            pipeline.valid(epoch, super_params.test_on)

    wandb.finish()

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
    super_params = config()

    if super_params.mode == "train":
        # Start training
        train()
    else:
        # Start test
        pass
        # test(run_id="2023-08-22-1916", best_epoch=26)  # default control of run_id and best_epoch
