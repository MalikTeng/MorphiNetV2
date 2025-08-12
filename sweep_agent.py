#!/usr/bin/env python3
"""
MorphiNet Hyperparameter Sweep Agent

This script orchestrates WandB sweeps for MorphiNet hyperparameter optimization.
It integrates with the existing training pipeline while allowing sweep parameter overrides.
"""

import os
import sys
import yaml
import wandb
import argparse
import torch
import gc
from main import train_morphinet, config

# def get_gpu_memory_usage():
#     """
#     Get current GPU memory usage in MB.
    
#     Returns:
#         float: Current GPU memory allocated in MB
#     """
#     if torch.cuda.is_available():
#         return torch.cuda.memory_allocated() / 1024 / 1024
#     return 0

# def check_gpu_memory_threshold(threshold_mb=10000):
#     """
#     Check if GPU memory usage exceeds threshold.
    
#     Args:
#         threshold_mb: Memory threshold in MB (default 10GB)
        
#     Returns:
#         bool: True if memory is below threshold, False otherwise
#     """
#     current_mb = get_gpu_memory_usage()
#     if current_mb > threshold_mb:
#         print(f"WARNING: GPU memory usage ({current_mb:.1f} MB) exceeds threshold ({threshold_mb} MB)")
#         if wandb.run:
#             wandb.log({"gpu_memory_warning": True, "gpu_memory_mb": current_mb})
#         return False
#     return True

def cleanup_gpu_memory():
    """Clean up GPU memory and garbage collect."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset memory stats to prevent accumulation
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    gc.collect()

def create_sweep_config(config_path="sweep_config.yaml"):
    """
    Load sweep configuration from YAML file.
    
    Args:
        config_path: Path to sweep configuration YAML file
        
    Returns:
        Dictionary containing sweep configuration
    """
    with open(config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    return sweep_config

def train_with_sweep_config():
    """
    Training function that integrates with WandB sweep.
    This function is called by the sweep agent with different hyperparameter combinations.
    """
    # Set memory optimization environment variables
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    
    # Initialize WandB run (will be managed by sweep)
    # Use reinit=True to avoid conflicts with existing runs
    wandb.init(reinit=True)
    
    # Get sweep configuration from WandB
    sweep_config = wandb.config
    
    # Create base arguments using existing config parser
    # Save and restore sys.argv to avoid conflicts
    import sys
    original_argv = sys.argv
    sys.argv = ['sweep_agent.py']  # Set minimal argv for config()
    try:
        base_args = config()
    finally:
        sys.argv = original_argv
    
    # Override base arguments with sweep parameters
    for key, value in sweep_config.items():
        if hasattr(base_args, key):
            # Handle special cases for list parameters
            if key == 'layers' and isinstance(value, list):
                setattr(base_args, key, value)
            elif key == 'crop_window_size' and isinstance(value, list):
                setattr(base_args, key, value)
            elif key == 'pixdim' and isinstance(value, list):
                setattr(base_args, key, value)
            elif key == 'kernel_size' and isinstance(value, list):
                setattr(base_args, key, value)
            elif key == 'strides' and isinstance(value, list):
                setattr(base_args, key, value)
            else:
                setattr(base_args, key, value)
    
    # Generate unique run ID for sweep
    sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else "manual"
    run_name = f"sweep-{sweep_id}-{wandb.run.id}"
    base_args.run_id = run_name
    
    # Signal to training pipeline that this is a sweep run
    base_args.is_sweep_run = True
    
    # Ensure online logging for sweep visibility
    base_args.mode = "online"
    
    # Configure WandB metrics properly for sweep runs
    try:
        # Define epoch as the primary step metric (no auto-increment)
        wandb.define_metric("epoch")
        
        # Map all training metrics to use epoch as x-axis  
        wandb.define_metric("unet/*", step_metric="epoch")
        wandb.define_metric("resnet/*", step_metric="epoch") 
        wandb.define_metric("gsn/*", step_metric="epoch")
        
        print("WandB metrics configured for sweep run")
    except Exception as e:
        print(f"Warning: Could not configure WandB metrics: {e}")
        pass
    
    try:
        # Call the existing training function with modified arguments
        train_morphinet(base_args)
        
        cleanup_gpu_memory()
        
    except Exception as e:
        print(f"Training failed for sweep run {run_name}: {e}")
        # Log the failure to WandB
        if wandb.run:
            wandb.log({"training_failed": True, "error": str(e)})
        
        # Clean up even after failure
        print("Cleaning up after failed run...")
        cleanup_gpu_memory()
        
        # Mark run as failed
        if wandb.run:
            wandb.run.finish(exit_code=1)
        raise e
    
    finally:
        # Enhanced cleanup for sequential runs to prevent OOM
        print("Final cleanup...")
        
        # Properly close the WandB run first
        try:
            if wandb.run:
                wandb.finish()
        except Exception as e:
            print(f"Warning: WandB cleanup failed: {e}")
        
        # Enhanced GPU memory cleanup for sequential runs
        try:
            cleanup_gpu_memory()
            
            # Additional aggressive cleanup for sequential execution
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                # Clear cached memory more aggressively
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Clear IPC memory for multi-process safety
                for i in range(torch.cuda.device_count()):
                    try:
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                    except:
                        pass
                        
            print("Enhanced cleanup completed for sequential run")
        except Exception as e:
            print(f"Warning: Enhanced cleanup failed: {e}")
        
        # Final GPU memory cleanup
        cleanup_gpu_memory()
        
        # Additional cleanup for sweep context
        try:
            # Force garbage collection
            import gc
            gc.collect()
            # Clear any remaining CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass

def main():
    """Main entry point for sweep execution."""
    parser = argparse.ArgumentParser(description="MorphiNet Hyperparameter Sweep")
    parser.add_argument("--sweep_config", type=str, default="sweep_config.yaml",
                       help="Path to sweep configuration YAML file")
    parser.add_argument("--create_sweep", action="store_true",
                       help="Create a new sweep (returns sweep ID)")
    parser.add_argument("--sweep_id", type=str, default=None,
                       help="Join existing sweep with this ID")
    parser.add_argument("--count", type=int, default=1,
                       help="Number of sweep runs to execute")
    parser.add_argument("--project", type=str, default="MorphiNet-Sweep",
                       help="WandB project name for sweep")
    
    args = parser.parse_args()
    
    # Ensure WandB is logged in
    wandb.login()
    
    if args.create_sweep:
        # Create a new sweep
        sweep_config = create_sweep_config(args.sweep_config)
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print(f"Created sweep with ID: {sweep_id}")
        print(f"To run sweep agent: python sweep_agent.py --sweep_id {sweep_id}")
        return sweep_id
    
    elif args.sweep_id:
        # Join existing sweep
        print(f"Joining sweep: {args.sweep_id}")
        print(f"Will run {args.count} sweep runs")
        
        # Info about sequential execution
        if args.count > 1:
            print(f"INFO: Running {args.count} parameter combinations SEQUENTIALLY.")
            print("Each combination runs one after another to avoid OOM errors.")
            print("Bayesian optimization will suggest the best parameter combinations.")
            print(f"Expected total time: ~{args.count * 0.5:.1f} hours (assuming 30min per run)")
            print()
        
        # Start sweep agent - runs will be sequential within this agent
        wandb.agent(
            sweep_id=args.sweep_id,
            function=train_with_sweep_config,
            count=args.count,
            project=args.project
        )
        
    else:
        # Show usage
        print("Usage:")
        print("  Create sweep: python sweep_agent.py --create_sweep")
        print("  Join sweep:   python sweep_agent.py --sweep_id <sweep_id>")
        print("  Full example: python sweep_agent.py --create_sweep --sweep_config sweep_config.yaml")

if __name__ == "__main__":
    main()