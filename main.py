"""
Updated main.py for MorphiNet using the modular architecture.

This file provides a cleaner interface to the MorphiNet training pipeline
using the new modular components.
"""

import os
import sys
import time
import glob
import argparse
import gc
import torch
import wandb

from utils.tools import draw_eval_score
wandb.login()

import warnings
warnings.filterwarnings('ignore')

torch.multiprocessing.set_sharing_strategy('file_system')


def config():
    """Parse command line arguments for MorphiNet training."""
    parser = argparse.ArgumentParser(description="MorphiNet Training Pipeline")
    
    # Mode parameters
    parser.add_argument("--mode", type=str, default="offline", 
                       help="Wandb mode: 'disabled', 'offline', 'online'")
    parser.add_argument("--save_on", type=str, default="sct", 
                       help="Dataset for validation: 'cap' or 'sct'")
    parser.add_argument("--template_mesh_dir", type=str,
                       default="./template/template_mesh-myo.obj",
                       help="Path to template mesh file")

    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=5, 
                       help="Maximum number of epochs")
    parser.add_argument("--pretrain_epochs", type=int, default=2, 
                       help="Number of epochs for UNet training")
    parser.add_argument("--train_epochs", type=int, default=3, 
                       help="Number of epochs for ResNet training")
    parser.add_argument("--reduce_count_down", type=int, default=-1, 
                       help="Countdown for mesh face reduction")
    parser.add_argument("--val_interval", type=int, default=1, 
                       help="Validation interval")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="Cache rate")
    parser.add_argument("--crop_window_size", type=int, nargs='+', 
                       default=[128, 128, 128], help="Crop window size")
    parser.add_argument("--pixdim", type=float, nargs='+', default=[8, 8, 8], 
                       help="Pixel dimensions")
    parser.add_argument("--lambda_0", type=float, default=0.86, 
                       help="Chamfer distance loss coefficient")
    parser.add_argument("--lambda_1", type=float, default=0.75, 
                       help="Laplacian smoothing loss coefficient")
    parser.add_argument("--iteration", type=int, default=20, 
                       help="Distance field warping iterations")
    parser.add_argument("--sigmoid_scale_factor", type=float, default=0.19, 
                       help="Sigmoid mask scale factor")
    parser.add_argument("--mask_threshold", type=float, default=0.1, 
                       help="Distance map mask threshold")

    # Data parameters
    parser.add_argument("--ct_ratio", type=float, default=1.0, 
                       help="Portion of CT data for training")
    parser.add_argument("--ct_json_dir", type=str, 
                       default="./dataset/dataset_task20_f0.json",
                       help="CT dataset JSON file")
    parser.add_argument("--ct_data_dir", type=str, 
                       default="/path/to/ct/data",
                       help="CT data directory")
    parser.add_argument("--mr_json_dir", type=str, 
                       default="./dataset/dataset_task11_f0.json",
                       help="MR dataset JSON file")
    parser.add_argument("--mr_data_dir", type=str, 
                       default="/path/to/mr/data",
                       help="MR data directory")

    # Model parameters
    parser.add_argument("--num_classes", type=int, default=5, 
                       help="Number of segmentation classes")
    parser.add_argument("--filters", type=int, nargs='+', 
                       default=[8, 16, 32, 64, 128], 
                       help="UNet filter sizes")
    parser.add_argument("--kernel_size", type=int, nargs='+', 
                       default=[3, 3, 3, 3, 3], 
                       help="UNet kernel sizes")
    parser.add_argument("--strides", type=int, nargs='+', 
                       default=[1, 2, 2, 2, 2], 
                       help="UNet strides")
    parser.add_argument("--layers", type=int, nargs='+', 
                       default=[1, 2, 2, 4], 
                       help="ResNet layer configuration")
    parser.add_argument("--upscale_ratio", type=int, default=2, 
                       help="ResNet upscaling ratio")
    parser.add_argument("--subdiv_levels", type=int, default=2, 
                       help="Graph subdivision levels")
    parser.add_argument("--hidden_features_gsn", type=int, default=64, 
                       help="GSN hidden features")

    # Checkpoint parameters
    parser.add_argument("--use_ckpt", type=str, default="n", 
                       help="Checkpoint directory to resume from")
    parser.add_argument("--ckpt_dir", type=str, default="./Checkpoint", 
                       help="Directory to save checkpoints")
    parser.add_argument("--run_id", type=str, default="", 
                       help="Run identifier")

    # Data modality flags
    parser.add_argument("--_4d", action="store_true", 
                       help="Enable 4D data processing")
    parser.add_argument("--_mr", action="store_true", 
                       help="Train exclusively on MR data")

    return parser.parse_args()


def train_modular(super_params):
    """
    Train MorphiNet using the modular architecture.
    
    Args:
        super_params: Parsed command line arguments
    """
    print("="*80)
    print("MORPHINET TRAINING - MODULAR ARCHITECTURE")
    print("="*80)
    
    # Generate run ID
    run_id = f"{time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))}"
    if not super_params.run_id:
        super_params.run_id = f"{super_params.save_on}--" + \
            f"{os.path.basename(super_params.template_mesh_dir).split('-')[-1][:-4]}--" + \
                f"{os.path.basename(super_params.ct_json_dir).split('_')[-1][:-5]}--{run_id}"

    # Initialize Weights & Biases
    with wandb.init(config=super_params, mode=super_params.mode, 
                   project="MorphiNet", name=super_params.run_id, resume="allow"):
        
        try:
            # Import the modular pipeline
            from run import create_training_pipeline
            
            # Create training pipeline
            pipeline = create_training_pipeline(
                super_params=super_params,
                seed=8,
                num_workers=16
            )
            
            # Load pretrained weights if specified
            if super_params.use_ckpt != "n" and super_params.use_ckpt is not None:
                print(f"Loading pretrained weights from {super_params.use_ckpt}")
                
                # Determine checkpoint directory
                base_ckpt_path = super_params.use_ckpt
                potential_trained_weights_path = os.path.join(base_ckpt_path, "trained_weights")
                
                if os.path.isfile(os.path.join(potential_trained_weights_path, "best_UNet_MR.pth")):
                    actual_ckpt_dir = potential_trained_weights_path
                elif os.path.isfile(os.path.join(base_ckpt_path, "best_UNet_MR.pth")):
                    actual_ckpt_dir = base_ckpt_path
                else:
                    print(f"Warning: Could not find pretrained weights in {base_ckpt_path}")
                    actual_ckpt_dir = None
                
                if actual_ckpt_dir:
                    pipeline.load_pretrained_weights(actual_ckpt_dir)
            
            # Execute full training pipeline
            pipeline.train_full_pipeline()
            
            print("\nTraining completed successfully!")
            
            # Print final metrics
            training_metrics = pipeline.get_training_metrics()
            validation_metrics = pipeline.get_validation_metrics()
            
            if validation_metrics.get('best_eval_score'):
                print(f"Best validation score: {validation_metrics['best_eval_score']:.4f}")
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise e
        
        finally:
            # Cleanup
            if 'pipeline' in locals():
                del pipeline
            torch.cuda.empty_cache()
            gc.collect()


def train_legacy(super_params):
    """
    Train MorphiNet using the legacy epoch-by-epoch approach.
    
    This function maintains backward compatibility with the original training loop
    while using the modular components.
    
    Args:
        super_params: Parsed command line arguments
    """
    print("="*80)
    print("MORPHINET TRAINING - LEGACY COMPATIBLE MODE")
    print("="*80)
    
    # Generate run ID
    run_id = f"{time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))}"
    if not super_params.run_id or not wandb.run.resumed:
        super_params.run_id = f"{super_params.save_on}--" + \
            f"{os.path.basename(super_params.template_mesh_dir).split('-')[-1][:-4]}--" + \
                f"{os.path.basename(super_params.ct_json_dir).split('_')[-1][:-5]}--{run_id}"

    # Initialize Weights & Biases
    with wandb.init(config=super_params, mode=super_params.mode, 
                   project="MorphiNet", name=super_params.run_id, resume="allow"):
        
        try:
            # Import the modular pipeline
            from run import create_training_pipeline
            
            # Create training pipeline
            pipeline = create_training_pipeline(
                super_params=super_params,
                seed=8,
                num_workers=16
            )
            
            # Load pretrained weights if specified
            has_unet_ckpt = False
            has_resnet_ckpt = False
            
            if super_params.use_ckpt != "n" and super_params.use_ckpt is not None:
                print(f"Loading pretrained weights from {super_params.use_ckpt}")
                base_ckpt_path = super_params.use_ckpt
                potential_trained_weights_path = os.path.join(base_ckpt_path, "trained_weights")
                
                if os.path.isfile(os.path.join(potential_trained_weights_path, "best_UNet_MR.pth")):
                    actual_ckpt_dir = potential_trained_weights_path
                elif os.path.isfile(os.path.join(base_ckpt_path, "best_UNet_MR.pth")):
                    actual_ckpt_dir = base_ckpt_path
                else:
                    print(f"Warning: Could not find pretrained weights in {base_ckpt_path}")
                    actual_ckpt_dir = None

                if actual_ckpt_dir:
                    unet_mr_path = glob.glob(f"{actual_ckpt_dir}/best_UNet_MR.pth")
                    unet_ct_path = glob.glob(f"{actual_ckpt_dir}/best_UNet_CT.pth")
                    resnet_path = glob.glob(f"{actual_ckpt_dir}/best_ResNet.pth")
                    
                    has_unet_ckpt = bool(unet_mr_path and unet_ct_path)
                    has_resnet_ckpt = bool(resnet_path)
                    
                    pipeline.load_pretrained_weights(actual_ckpt_dir)
            
            # Training phases using the orchestrator's phase-based training
            if not has_unet_ckpt:
                print("Training UNet phase...")
                pipeline.train_phase("unet", 0, super_params.pretrain_epochs)
            else:
                print("Skipping UNet training - using checkpoint")
            
            if not has_resnet_ckpt:
                print("Training ResNet phase...")
                pipeline.train_phase("resnet", super_params.pretrain_epochs, super_params.train_epochs)
            else:
                print("Skipping ResNet training - using checkpoint")
            
            print("Training GSN phase...")
            pipeline.train_phase("gsn", super_params.train_epochs, super_params.max_epochs)
            
            print("\nTraining completed successfully!")
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise e
        
        finally:
            # Cleanup
            if 'pipeline' in locals():
                del pipeline
            torch.cuda.empty_cache()
            gc.collect()


def main():
    """Main entry point for MorphiNet training."""
    super_params = config()
    
    print("MorphiNet Training Pipeline")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Mode: {super_params.mode}")
    print(f"Save on: {super_params.save_on}")
    print(f"Max epochs: {super_params.max_epochs}")
    
    # Choose training mode
    use_modular = True  # Set to False for legacy epoch-by-epoch training
    
    if use_modular:
        train_modular(super_params)
    else:
        train_legacy(super_params)


if __name__ == '__main__':
    main()