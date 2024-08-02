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
                        default="./dataset/dataset_task11_f0.json",    # less data less burden
                        # default="./dataset/dataset_task10_f0.json",  # use only for 4d
                        help="the path to the json file with named list of MR train/valid/test sets")
    parser.add_argument("--ct_data_dir", type=str, 
                        default="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset020_SCOTHEART", 
                        help="the path to your processed images, must be in nifti format")
    parser.add_argument("--mr_data_dir", type=str, 
                        default="/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset011_CAP_SAX", 
                        # default="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX", 
                        # default="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset010_CAP_SAX_NRRD", 
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

def train(super_params):
    # initialize the training pipeline
    run_id = f"{time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))}"
    super_params.run_id = f"{super_params.save_on}--" + \
        f"{os.path.basename(super_params.control_mesh_dir).split('-')[-1][:-4]}--" + \
            f"{os.path.basename(super_params.ct_json_dir).split('_')[-1][:-5]}--{run_id}"

    with wandb.init(config=super_params, mode=super_params.mode, project="MorphiNet", name=super_params.run_id):
        pipeline = TrainPipeline(
            super_params=super_params,
            seed=8, num_workers=0,
            )

        if super_params.save_on == "cap" and super_params._4d:
            # refine 4D mesh with NDF
            pipeline.load_pretrained_weight("all")
            for epoch in range(super_params.max_epochs, super_params.max_epochs + 50):
                # 5. refine the 4D mesh with NDF
                pipeline.train_iter(epoch, "ndf")
                # 6. validate network
                if epoch % super_params.val_interval == 0:
                    pipeline.valid(epoch, super_params.save_on)

        else:
            # train the network
            CKPT = False
            for epoch in range(super_params.max_epochs):
                torch.cuda.empty_cache()
                if epoch < super_params.pretrain_epochs:
                    if super_params.use_ckpt is not None and CKPT is False:
                        pipeline.load_pretrained_weight("unet")
                        CKPT = True
                    elif CKPT is False:
                        # 1. train segmentation encoder
                        pipeline.train_iter(epoch, "unet")
                elif epoch < super_params.train_epochs:
                    # drop the rotation and flip augmentation
                    if epoch == super_params.pretrain_epochs:
                        pipeline._data_warper(rotation=False)
                    # 2. train distance field prediction module
                    pipeline.train_iter(epoch, "resnet")
                else:
                    # 3. fine-tune the subdiv module
                    pipeline.train_iter(epoch, "gsn")
                    # 3.1 reduce the mesh face numbers
                    if epoch - super_params.train_epochs == super_params.reduce_count_down:
                        pipeline.update_precomputed_faces()
                    # 4. validate network
                    if epoch >= super_params.train_epochs and \
                        (epoch - super_params.train_epochs) % super_params.val_interval == 0:
                        pipeline.valid(epoch, super_params.save_on)


if __name__ == '__main__':
    super_params = config()

    # if super_params._mr:
    #     from run_mr import *
    # else:
    #     from run import *
    from run_mr import *

    train(super_params)
