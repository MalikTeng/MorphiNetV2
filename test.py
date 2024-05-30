import os
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

def config():
    """
        This function is for parsing commandline arguments.
    """
    parser = argparse.ArgumentParser()
    # mode parameters
    parser.add_argument("--save_on", type=str, default="sct", help="the dataset for validation, can be 'cap' or 'sct'")
    parser.add_argument("--control_mesh_dir", type=str,
                        default="/home/yd21/Documents/MorphiNet/template/control_mesh-lv.obj",
                        help="the path to your initial meshes")

    # training parameters
    parser.add_argument("--max_epochs", type=int, default=200, help="the maximum number of epochs for training")
    parser.add_argument("--pretrain_epochs", type=int, default=120, help="the number of epochs to train the segmentation encoder")
    parser.add_argument("--delay_epochs", type=int, default=160, help="the number of epochs to delay the fine-tuning and validation from start of pretrain")
    parser.add_argument("--reduce_count_down", type=int, default=10, help="the count down for reduce the mesh face numbers.")
    parser.add_argument("--val_interval", type=int, default=10, help="the interval of validation")

    parser.add_argument("--lr", type=float, default=1e-3, help="the learning rate for training")
    parser.add_argument("--batch_size", type=int, default=8, help="the batch size for training")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="the cache rate for training, see MONAI document for more details")
    parser.add_argument("--crop_window_size", type=int, nargs='+', default=[128, 128, 128], help="the size of the crop window for training")
    parser.add_argument("--pixdim", type=float, nargs='+', default=[8, 8, 8], help="the pixel dimension of downsampled images")
    parser.add_argument("--lambda_", type=float, nargs='+', default=[0.1, 3.6, 6.3, 0.1], help="the loss coefficients for DF MSE, Chamfer verts distance, face squared distance, and laplacian smooth term")

    # data parameters
    parser.add_argument("--ct_json_dir", type=str,
                        default="/home/yd21/Documents/MorphiNet/dataset/dataset_task20_f0.json", 
                        help="the path to the json file with named list of CTA train/valid/test sets")
    parser.add_argument("--mr_json_dir", type=str,
                        default="/home/yd21/Documents/MorphiNet/dataset/dataset_task17_f0.json", 
                        help="the path to the json file with named list of CMR train/valid/test sets")
    parser.add_argument("--ct_data_dir", type=str, 
                        default="/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset020_SCOTHEART", 
                        help="the path to your processed images, must be in nifti format")
    parser.add_argument("--mr_data_dir", type=str, 
                        default="/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset017_CAP_COMBINED", 
                        help="the path to your processed images")
    parser.add_argument("--ckpt_dir", type=str, 
                        default="/mnt/data/Experiment/MorphiNet/Checkpoint", 
                        help="the path to your checkpoint directory, for holding trained models and wandb logs")
    parser.add_argument("--out_dir", type=str, 
                        default="/mnt/data/Experiment/MorphiNet/Result", 
                        help="the path to your output directory, for saving outputs")
     
    # path to the pretrained modules
    parser.add_argument("--pretrained_pretext_mr_dir", type=str, default=None, help="the path to the pretrained pretext-mr")
    parser.add_argument("--pretrained_ae_dir", type=str, default=None, help="the path to the pretrained autoencoder")

    # structure parameters for df-predict module
    parser.add_argument("--num_classes", type=int, default=4, help="the number of segmentation classes of foreground exclude background")
    parser.add_argument("--channels", type=int, default=(8, 16, 32, 64, 128), nargs='+', help="the number of output channels in each layer of the encoder")
    parser.add_argument("--strides", type=int, default=(2, 2, 2, 2), nargs='+', help="the stride of the convolutional layer in the encoder")
    parser.add_argument("--layers", type=int, default=(1, 2, 2, 2), nargs='+', help="the number of layers in each residual block of the decoder")
    parser.add_argument("--block_inplanes", type=int, default=(4, 8, 16, 32), nargs='+', help="the number of intermedium channels in each residual block")

    # structure parameters for subdiv module
    parser.add_argument("--subdiv_levels", type=int, default=0, help="the number of subdivision levels for the mesh")
    # parser.add_argument("--subdiv_levels", type=int, default=2, help="the number of subdivision levels for the mesh (should be an integer larger than 0, where 0 means no subdivision)")
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
        seed=42, num_workers=0,
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

    fold = 'x'
    super_params.save_on = "cap"
    part = "myo"
    test_on = "cap"
    super_params.ct_json_dir = f"/home/yd21/Documents/MorphiNet/dataset/dataset_task20_f{fold}.json"
    super_params.ct_data_dir = f"/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset020_SCOTHEART"
    super_params.mr_json_dir = f"/home/yd21/Documents/MorphiNet/dataset/dataset_task10_f{fold}.json"
    super_params.mr_data_dir = f"/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset010_Abdul_SAX"
    super_params.best_epoch = 201
    super_params.control_mesh_dir = f"/home/yd21/Documents/MorphiNet/template/initial_mesh-{part}.obj"
    super_params.run_id = f"{super_params.save_on}-{part}-f{fold}"
    super_params.ckpt_dir = f"/mnt/data/Experiment/MorphiNet/Result/{super_params.save_on}-{part}-f{fold}/trained_weights"
    super_params.out_dir = f"/mnt/data/Experiment/MorphiNet/Output/{test_on}/{part}/"
    # ablation(super_params)

    test(super_params)
