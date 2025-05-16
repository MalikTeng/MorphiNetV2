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
    parser.add_argument("--template_mesh_dir", type=str,
                        default="./template/template_mesh-myo.obj",
                        help="the path to your initial meshes")

    # training parameters
    parser.add_argument("--max_epochs", type=int, default=10, help="the maximum number of epochs for training")
    parser.add_argument("--pretrain_epochs", type=int, default=5, help="the number of epochs to train the segmentation UNet")
    parser.add_argument("--train_epochs", type=int, default=8, help="the number of epochs to train the distance field prediction ResNet")
    parser.add_argument("--reduce_count_down", type=int, default=-1, help="the count down for reduce the mesh face numbers.")
    parser.add_argument("--val_interval", type=int, default=1, help="the interval of validation")

    parser.add_argument("--lr", type=float, default=1e-3, help="the learning rate for training")
    parser.add_argument("--batch_size", type=int, default=1, help="the batch size for training")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="the cache rate for training, see MONAI document for more details")
    parser.add_argument("--crop_window_size", type=int, nargs='+', default=[128, 128, 128], help="the size of the crop window for training")
    parser.add_argument("--pixdim", type=float, nargs='+', default=[4, 4, 4], help="the pixel dimension of downsampled images")
    parser.add_argument("--lambda_0", type=float, default=1.0, help="the loss coefficients for Chamfer verts distance term")
    parser.add_argument("--lambda_1", type=float, default=0.1, help="the loss coefficients for point to mesh distance term")
    parser.add_argument("--iteration", type=int, default=10, help="the iterations for the distance field warping")
    parser.add_argument("--sigmoid_scale_factor", type=float, default=1.0, help="the scale factor for the sigmoid mask transition")
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
                        default="/mnt/data/Experiment/MorphiNet/Checkpoint/dynamic/sct--myo--f0--2025-05-01-0949", 
                        help="path to pretrained models ('n' for no checkpoint, or specify a path)")

    # structure parameters for df-predict module
    parser.add_argument("--num_classes", type=int, default=4, help="the number of segmentation classes including the background")
    parser.add_argument("--kernel_size", type=int, default=(3, 3, 3, 3), nargs='+', help="the kernel size of the convolutional layer in the encoder")
    parser.add_argument("--strides", type=int, default=(1, 2, 2, 2), nargs='+', help="the stride of the convolutional layer in the encoder")
    parser.add_argument("--filters", type=int, default=(8, 16, 32, 64), nargs='+', help="the number of output channels in each layer of the encoder")
    parser.add_argument("--layers", type=int, default=(1, 4, 4, 8), nargs='+', help="the number of layers in each residual block of the decoder")

    # structure parameters for subdiv module
    parser.add_argument("--subdiv_levels", type=int, default=2, help="the number of subdivision levels for the mesh (should be an integer larger than 0, where 0 means no subdivision)")
    parser.add_argument("--hidden_features_gsn", type=int, default=16, help="the number of hidden features for the graph subdivide network")

    # run_id for wandb, will create automatically if not specified for training
    parser.add_argument("--run_id", type=str, default=None, help="the run name for wandb and local machine")

    # the best epoch for testing
    parser.add_argument("--best_epoch", type=int, default=None, help="the best epoch for testing")

    args = parser.parse_args()

    return args

def train(super_params):
    # initialize the training pipeline
    run_id = f"{time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))}"
    super_params.run_id = f"{super_params.save_on}--" + \
        f"{os.path.basename(super_params.template_mesh_dir).split('-')[-1][:-4]}--" + \
            f"{os.path.basename(super_params.ct_json_dir).split('_')[-1][:-5]}--{run_id}"

    with wandb.init(config=super_params, mode=super_params.mode, project="MorphiNet", name=super_params.run_id):
        pipeline = TrainPipeline(
            super_params=super_params,
            seed=8, num_workers=12,
            )

        if super_params.save_on == "cap" and super_params._4d:
            # refine 4D mesh with NDF
            pipeline.load_pretrained_weight("all")
            # pipeline._data_warper(rotation=False)
            for epoch in range(super_params.max_epochs, super_params.max_epochs + 50):
                # 5. refine the 4D mesh with NDF
                pipeline.train_iter(epoch, "ndf")
                # 6. validate network
                if epoch % super_params.val_interval == 0:
                    pipeline.valid(epoch, super_params.save_on)

        else:
            # Simplified training workflow
            # Check for existing checkpoints
            has_unet_ckpt = False
            has_resnet_ckpt = False
            if super_params.use_ckpt is not None:
                print(f"Loading pretrained weights from {super_params.use_ckpt}")
                ckpt_dir = f"{super_params.use_ckpt}/trained_weights"
                unet_mr_path = glob.glob(f"{ckpt_dir}/best_UNet_MR.pth")
                unet_ct_path = glob.glob(f"{ckpt_dir}/best_UNet_CT.pth")
                # resnet_path = glob.glob(f"{ckpt_dir}/best_ResNet.pth")
                
                has_unet_ckpt = bool(unet_mr_path and unet_ct_path)
                # has_resnet_ckpt = bool(resnet_path)
                
                # Load all available checkpoints
                pipeline.load_pretrained_weight("all")
            
            # train the network
            for epoch in range(super_params.max_epochs):
                torch.cuda.empty_cache()
                
                # Phase 1: UNet training (segmentation encoder)
                if epoch < super_params.pretrain_epochs:
                    if not has_unet_ckpt:
                        pipeline.train_iter(epoch, "unet")
                    else:
                        print(f"Skipping UNet training (epoch {epoch}) - using checkpoint")
                        # Jump to next phase
                        epoch = super_params.pretrain_epochs - 1
                
                # Phase 2: ResNet training (distance field prediction)
                elif epoch < super_params.train_epochs:
                    # First epoch of ResNet phase - ensure UNet weights are loaded
                    if epoch == super_params.pretrain_epochs:
                        pipeline.load_pretrained_weight("unet")
                    
                    if not has_resnet_ckpt:
                        pipeline.train_iter(epoch, "resnet")
                    else:
                        print(f"Skipping ResNet training (epoch {epoch}) - using checkpoint")
                        # Jump to next phase
                        epoch = super_params.train_epochs - 1
                
                # Phase 3: GSN training (graph subdivision) - always train this phase
                else:
                    # First epoch of GSN phase - ensure all weights are loaded
                    if epoch == super_params.train_epochs:
                        pipeline.load_pretrained_weight("all")
                    
                    # Always train the GSN module
                    pipeline.train_iter(epoch, "gsn")
                    
                    # Reduce mesh face numbers if needed
                    if epoch - super_params.train_epochs == super_params.reduce_count_down:
                        pipeline.update_precomputed_faces()
                    
                    # Validate network
                    if (epoch - super_params.train_epochs) % super_params.val_interval == 0:
                        pipeline.valid(epoch, super_params.save_on)


if __name__ == '__main__':
    super_params = config()

    if super_params._mr:
        from run_mr import *
    else:
        from run import *

    train(super_params)
