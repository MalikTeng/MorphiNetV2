"""
Updated main.py for MorphiNet using the modular architecture.

This file provides a cleaner interface to the MorphiNet training pipeline
using the new modular components.
"""

import os
import sys
import time
# glob import not needed for modular architecture
import argparse
import gc
import torch
import wandb

from utils.tools import draw_eval_score
wandb.login()

import warnings
warnings.filterwarnings('ignore')

torch.multiprocessing.set_sharing_strategy('file_system')

# UNet transform logging has been removed as per cleanup requirements


def config():
    """Parse command line arguments for MorphiNet training."""
    parser = argparse.ArgumentParser(description="MorphiNet Training Pipeline")
    
    # Mode parameters
    parser.add_argument("--mode", type=str, default="offline", 
                       help="Wandb mode: 'disabled', 'offline', 'online'")
    parser.add_argument("--template_mesh_dir", type=str,
                       default="./template/template_mesh-myo.obj",
                       help="Path to template mesh file")
    parser.add_argument("--inference_only", action="store_true",
                       help="Run inference only (no training)")
    parser.add_argument("--test_phase", type=str, default="unet",
                       choices=["unet", "resnet", "gsn", "both", "all"],
                       help="Which phase to test: 'unet', 'resnet', 'gsn', 'both' (unet+resnet), or 'all' (unet+resnet+gsn)")
    parser.add_argument("--test_dataset", type=str, default="both",
                       choices=["acdc", "mmwhs", "cap", "scotheart", "both"],
                       help="Which dataset to test: 'acdc', 'mmwhs', 'cap', 'scotheart', or 'both'")

    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=3, 
                       help="Maximum number of epochs")
    parser.add_argument("--pretrain_epochs", type=int, default=1, 
                       help="Number of epochs for UNet training")
    parser.add_argument("--train_epochs", type=int, default=2, 
                       help="Number of epochs for ResNet training")
    parser.add_argument("--reduce_count_down", type=int, default=-1, 
                       help="Countdown for mesh face reduction")
    parser.add_argument("--val_interval", type=int, default=1, 
                       help="Validation interval")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="Cache rate")
    parser.add_argument("--max_samples", type=int, default=1, 
                       help="Maximum number of samples per dataset (0 for full dataset)")
    parser.add_argument("--crop_window_size", type=int, nargs='+', 
                       default=[128, 128, 128], help="Crop window size")
    parser.add_argument("--pixdim", type=float, nargs='+', default=[8, 8, 8], 
                       help="Pixel dimensions")
    parser.add_argument("--lambda_0", type=float, default=1.0, 
                       help="Chamfer distance loss coefficient")
    parser.add_argument("--lambda_1", type=float, default=1.0, 
                       help="Laplacian smoothing loss coefficient")
    parser.add_argument("--iteration", type=int, default=10, 
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
                       default="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset020_SCOTHEART",
                       help="CT data directory")
    parser.add_argument("--mr_json_dir", type=str, 
                       default="./dataset/dataset_task11_f0.json",
                       help="MR dataset JSON file")
    parser.add_argument("--mr_data_dir", type=str, 
                       default="/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX",
                       help="MR data directory")

    # Model parameters
    parser.add_argument("--num_classes", type=int, default=4, 
                       help="Number of segmentation classes (after preprocessing: background, LV, MYO, RV)")
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
    parser.add_argument("--ckpt_dir", type=str, default="/mnt/data/Experiment/MorphiNet/Checkpoint/", 
                       help="Directory to save checkpoints")
    parser.add_argument("--run_id", type=str, default="", 
                       help="Run identifier")

    # Deprecated parameters removed: --_4d and --_mr
    
    # Add target parameter for testing dataset identification
    parser.add_argument("--target", type=str, default=None,
                       help="Target dataset for testing (exact dataset identifier)")
    
    # Note: Histogram matching parameters removed - functionality now handled automatically by HistogramMatchd transform
    
    return parser.parse_args()


def test_morphinet(super_params):
    """
    Test trained MorphiNet models using the modular architecture.
    
    Args:
        super_params: Parsed command line arguments
    """
    print("="*80)
    print("MORPHINET INFERENCE PIPELINE")
    print("="*80)
    print(f"Testing phase: {super_params.test_phase}")
    print(f"Testing dataset: {super_params.test_dataset}")
    
    # Generate run ID for testing
    run_id = f"test-{super_params.test_phase}-{super_params.test_dataset}-{time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))}"
    
    # Initialize Weights & Biases for logging test results
    with wandb.init(config=super_params, mode=super_params.mode, 
                   project="MorphiNet-Testing", name=run_id, resume="allow"):
        
        try:
            # Import the modular pipeline
            from run import create_testing_pipeline
            
            # Create testing pipeline
            pipeline = create_testing_pipeline(
                super_params=super_params,
                seed=42,
                num_workers=4,
                dataset=super_params.test_dataset
            )
            
            # Load best model weights
            checkpoint_path = None
            
            # Determine checkpoint directory
            if super_params.use_ckpt != "n" and super_params.use_ckpt is not None:
                checkpoint_path = super_params.use_ckpt
                print(f"Using specified checkpoint path: {checkpoint_path}")
            else:
                # Auto-detect latest checkpoint if no specific path provided
                dynamic_dir = os.path.join(super_params.ckpt_dir, "dynamic")
                if os.path.exists(dynamic_dir):
                    # Find the most recent checkpoint directory
                    checkpoint_dirs = [d for d in os.listdir(dynamic_dir) 
                                     if os.path.isdir(os.path.join(dynamic_dir, d))]
                    if checkpoint_dirs:
                        # Sort by modification time, get most recent
                        checkpoint_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(dynamic_dir, x)), reverse=True)
                        checkpoint_path = os.path.join(dynamic_dir, checkpoint_dirs[0])
                        print(f"Auto-detected latest checkpoint: {checkpoint_path}")
                    else:
                        print(f"Warning: No checkpoint directories found in {dynamic_dir}")
                        return
                else:
                    print(f"Warning: Checkpoint directory {dynamic_dir} does not exist")
                    return
            
            # Determine weights path (support both direct weights dir and checkpoint/trained_weights structure)
            if os.path.exists(os.path.join(checkpoint_path, "trained_weights")):
                weights_path = os.path.join(checkpoint_path, "trained_weights")
            elif os.path.exists(os.path.join(checkpoint_path, "best_UNet_CT.pth")):
                weights_path = checkpoint_path
            else:
                print(f"Warning: Could not find trained weights in {checkpoint_path}")
                print("Expected structure: {checkpoint_path}/trained_weights/ or weights directly in {checkpoint_path}/")
                return
            
            print(f"Loading trained weights from {weights_path}")
            pipeline.load_pretrained_weights(weights_path)
            
            # Execute inference testing using the new modular approach
            test_results = pipeline.test()
            
        except Exception as e:
            print(f"Testing failed with error: {e}")
            raise e
        
        finally:
            # Cleanup
            if 'pipeline' in locals():
                del pipeline
            torch.cuda.empty_cache()
            gc.collect()


def train_morphinet(super_params):
    """
    Train MorphiNet using the modular architecture.
    
    Args:
        super_params: Parsed command line arguments
    """
    print("="*80)
    print("MORPHINET TRAINING PIPELINE")
    print("="*80)
    
    # Generate run ID
    run_id = f"{time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))}"
    if not super_params.run_id:
        super_params.run_id = f"ct--" + \
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


# Only modular architecture supported


def main():
    """Main entry point for MorphiNet training and testing."""
    super_params = config()
    
    print("MorphiNet Pipeline")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Mode: {super_params.mode}")
    
    if super_params.inference_only:
        print("Running in INFERENCE mode")
        print(f"Test phase: {super_params.test_phase}")
        print(f"Test dataset: {super_params.test_dataset}")
        print(f"Max samples: {super_params.max_samples}")
        
        # Run inference testing
        test_morphinet(super_params)
    else:
        print("Running in TRAINING mode")
        print(f"Training validation: UNet(CT+MR) -> ResNet(CT) -> GSN(CT)")
        print(f"Max epochs: {super_params.max_epochs}")
        
        # Train using modular architecture
        train_morphinet(super_params)


if __name__ == '__main__':
    main()