import os, sys
import time
import glob
import argparse
import gc
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
    parser.add_argument("--template_mesh_dir", type=str,
                        default="./template/template_mesh-myo.obj",
                        help="the path to your initial meshes")

    # training parameters
    parser.add_argument("--max_epochs", type=int, default=5, help="the maximum number of epochs for training")
    parser.add_argument("--pretrain_epochs", type=int, default=2, help="the number of epochs to train the segmentation UNet")
    parser.add_argument("--train_epochs", type=int, default=3, help="the number of epochs to train the distance field prediction ResNet")
    parser.add_argument("--reduce_count_down", type=int, default=-1, help="the count down for reduce the mesh face numbers.")
    parser.add_argument("--val_interval", type=int, default=1, help="the interval of validation")

    parser.add_argument("--lr", type=float, default=1e-3, help="the learning rate for training")
    parser.add_argument("--batch_size", type=int, default=1, help="the batch size for training")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="the cache rate for training, see MONAI document for more details")
    parser.add_argument("--crop_window_size", type=int, nargs='+', default=[128, 128, 128], help="the size of the crop window for training")
    parser.add_argument("--pixdim", type=float, nargs='+', default=[8, 8, 8], help="the pixel dimension of downsampled images")
    parser.add_argument("--lambda_0", type=float, default=0.86, help="the loss coefficients for Chamfer verts distance term")
    parser.add_argument("--lambda_1", type=float, default=0.75, help="the loss coefficients for point to mesh distance term")
    parser.add_argument("--iteration", type=int, default=20, help="the iterations for the distance field warping")
    parser.add_argument("--sigmoid_scale_factor", type=float, default=0.19, help="the scale factor for the sigmoid mask transition")
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
                        # default=None, 
                        default="/mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/sct--myo--f0--2025-06-18-1019", 
                        help="path to pretrained models ('n' for no checkpoint, or specify a path)")

    # structure parameters for df-predict module
    parser.add_argument("--num_classes", type=int, default=4, help="the number of segmentation classes (background + 4 anatomical structures: LV, LV/RV-MYO, RV)")
    parser.add_argument("--filters", type=int, default=(8, 16, 32, 64, 128), nargs='+', help="the number of output channels in each layer of the encoder")
    parser.add_argument("--kernel_size", type=int, default=(3, 3, 3, 3, 3), nargs='+', help="the kernel size of the convolutional layer in the encoder")
    parser.add_argument("--strides", type=int, default=(1, 2, 2, 2, 2), nargs='+', help="the stride of the convolutional layer in the encoder")
    parser.add_argument("--upscale_ratio", type=int, default=2, help="the upscale ratio for the decoder output compared to input segmentation")
    parser.add_argument("--layers", type=int, default=(1, 2, 2, 4), nargs='+', help="the number of layers in each residual block of the decoder")

    # structure parameters for subdiv module
    parser.add_argument("--subdiv_levels", type=int, default=2, help="the number of subdivision levels for the mesh (should be an integer larger than 0, where 0 means no subdivision)")
    parser.add_argument("--hidden_features_gsn", type=int, default=64, help="the number of hidden features for the graph subdivide network")

    # run_id for wandb, will create automatically if not specified for training
    parser.add_argument("--run_id", type=str, default=None, help="the run name for wandb and local machine")

    # the best epoch (relevant for resuming training or knowing context)
    parser.add_argument("--best_epoch", type=int, default=None, help="The best epoch, usually for identifying a checkpoint to load")

    args = parser.parse_args()

    return args

def train(super_params):
    run_id = f"{time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))}"
    if not super_params.run_id or not wandb.run.resumed:
        super_params.run_id = f"{super_params.save_on}--" + \
            f"{os.path.basename(super_params.template_mesh_dir).split('-')[-1][:-4]}--" + \
                f"{os.path.basename(super_params.ct_json_dir).split('_')[-1][:-5]}--{run_id}"

    with wandb.init(config=super_params, mode=super_params.mode, project="MorphiNet", name=super_params.run_id, resume="allow"):
        pipeline = TrainPipeline(
            super_params=super_params,
            seed=8, num_workers=16,
            is_training=True
            )

        has_unet_ckpt = False
        has_resnet_ckpt = False
        if super_params.use_ckpt is not None:
            print(f"Loading pretrained weights from {super_params.use_ckpt}")
            base_ckpt_path = super_params.use_ckpt
            potential_trained_weights_path = os.path.join(base_ckpt_path, "trained_weights")
            
            if os.path.isfile(os.path.join(potential_trained_weights_path, "best_UNet_MR.pth")):
                actual_ckpt_dir = potential_trained_weights_path
            elif os.path.isfile(os.path.join(base_ckpt_path, "best_UNet_MR.pth")):
                actual_ckpt_dir = base_ckpt_path
            else:
                print(f"Warning: Could not find UNet/ResNet weights in {base_ckpt_path} or {potential_trained_weights_path}. Check --use_ckpt path.")
                actual_ckpt_dir = None

            if actual_ckpt_dir:
                unet_mr_path = glob.glob(f"{actual_ckpt_dir}/best_UNet_MR.pth")
                unet_ct_path = glob.glob(f"{actual_ckpt_dir}/best_UNet_CT.pth")
                resnet_path = glob.glob(f"{actual_ckpt_dir}/best_ResNet.pth")
                
                # has_unet_ckpt = bool(unet_mr_path and unet_ct_path)
                # has_resnet_ckpt = bool(resnet_path)
            
            pipeline.load_pretrained_weight("all")

        current_training_phase = None
        
        for epoch in range(super_params.max_epochs):
            torch.cuda.empty_cache()
            
            if epoch < super_params.pretrain_epochs:
                new_phase = "unet"
                will_validate_this_epoch = epoch % super_params.val_interval == 0
                
                if not has_unet_ckpt:
                    if current_training_phase != new_phase:
                        current_training_phase = new_phase
                        # Load both MR and CT training data for UNet phase
                        pipeline.prepare_all_dataloaders(data_types=["train"], training_phase=current_training_phase)
                    # UNet training - commit=False if validation follows, commit=True if no validation
                    pipeline.train_iter(epoch, "unet", commit_log=not will_validate_this_epoch)
                
                    # Validate segmentation after UNet training (both CT and MR encoders)
                    if will_validate_this_epoch:
                        # Prepare both CT and MR validation dataloaders for UNet phase
                        pipeline.prepare_all_dataloaders(data_types=["valid"], validation_phase="unet")
                        # This is the last log call for UNet phase, so commit=True
                        pipeline.validate_segmentation(epoch, super_params.save_on, commit=True)
                        # Clear both CT and MR validation dataloaders
                        pipeline._clear_dataloader("ct", "valid")
                        pipeline._clear_dataloader("mr", "valid")
                else:
                    print(f"Skipping UNet training (epoch {epoch}) - using checkpoint")
            
            elif epoch < super_params.train_epochs:
                new_phase = "resnet"
                will_validate_this_epoch = epoch % super_params.val_interval == 0
                
                if not has_resnet_ckpt:
                    if current_training_phase != new_phase:
                        current_training_phase = new_phase
                        # Load only CT training data for ResNet phase (clears MR data to save memory)
                        pipeline.prepare_all_dataloaders(data_types=["train"], training_phase=current_training_phase)
                    # ResNet training - commit=False if validation follows, commit=True if no validation
                    pipeline.train_iter(epoch, "resnet", commit_log=not will_validate_this_epoch)
                    
                    # Validate segmentation after ResNet training
                    if will_validate_this_epoch:
                        pipeline.prepare_all_dataloaders(data_types=["valid"], validation_phase="resnet")
                        # This is the last log call for ResNet phase, so commit=True
                        pipeline.validate_segmentation(epoch, super_params.save_on, commit=True)
                        pipeline._clear_dataloader("ct" if super_params.save_on == "sct" else "mr", "valid")
                else:
                    print(f"Skipping ResNet training (epoch {epoch}) - using checkpoint")
            
            else: # GSN training phase
                new_phase = "gsn"
                if current_training_phase != new_phase:
                    current_training_phase = new_phase
                    # Aggressive memory cleanup before GSN phase
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Load only CT training data for GSN phase (clears MR data to save memory)
                    pipeline.prepare_all_dataloaders(data_types=["train"], training_phase=current_training_phase)
                
                will_validate_this_epoch = (epoch - super_params.train_epochs) % super_params.val_interval == 0
                pipeline.train_iter(epoch, "gsn", commit_log=not will_validate_this_epoch)
                
                if epoch - super_params.train_epochs == super_params.reduce_count_down:
                    pipeline.update_precomputed_faces()
                
                if will_validate_this_epoch:
                    pipeline.prepare_all_dataloaders(data_types=["valid"], validation_phase="network") # Prepare val_loaders with full transforms
                    # valid is the last call, so commit=True (already set in valid method)
                    pipeline.valid(epoch, super_params.save_on)
                    pipeline._clear_dataloader("ct" if super_params.save_on == "sct" else "mr", "valid")

if __name__ == '__main__':
    super_params = config()

    from run import * # type: ignore

    print("Running in Training Mode...")
    train(super_params)
