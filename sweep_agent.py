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
from main import train_morphinet, config

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
    # Initialize WandB run (will be managed by sweep)
    wandb.init()
    
    # Get sweep configuration from WandB
    sweep_config = wandb.config
    
    # Create base arguments using existing config parser
    base_args = config()
    
    # Override base arguments with sweep parameters
    for key, value in sweep_config.items():
        if hasattr(base_args, key):
            # Handle special cases for list parameters
            if key == 'filters' and isinstance(value, list):
                setattr(base_args, key, value)
            elif key == 'layers' and isinstance(value, list):
                setattr(base_args, key, value)
            elif key == 'crop_window_size' and isinstance(value, list):
                setattr(base_args, key, value)
            elif key == 'pixdim' and isinstance(value, list):
                setattr(base_args, key, value)
            else:
                setattr(base_args, key, value)
    
    # Calculate GSN training epochs (remaining epochs after UNet and ResNet)
    total_epochs = base_args.max_epochs
    unet_epochs = base_args.pretrain_epochs
    resnet_epochs = base_args.train_epochs
    gsn_epochs = total_epochs - unet_epochs - resnet_epochs
    
    # Ensure positive GSN epochs
    if gsn_epochs < 10:
        gsn_epochs = 10
        # Adjust total epochs to accommodate minimum GSN training
        base_args.max_epochs = unet_epochs + resnet_epochs + gsn_epochs
    
    # Log the epoch distribution
    wandb.log({
        "epoch_distribution/unet_epochs": unet_epochs,
        "epoch_distribution/resnet_epochs": resnet_epochs, 
        "epoch_distribution/gsn_epochs": gsn_epochs,
        "epoch_distribution/total_epochs": base_args.max_epochs
    })
    
    # Generate unique run ID for sweep
    sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else "manual"
    run_name = f"sweep-{sweep_id}-{wandb.run.id}"
    base_args.run_id = run_name
    
    # Set validation modality consistently
    base_args.test_modality = "ct"
    
    # Ensure online logging for sweep visibility
    base_args.mode = "online"
    
    print(f"Starting sweep run: {run_name}")
    print(f"Hyperparameters: {dict(sweep_config)}")
    
    try:
        # Call the existing training function with modified arguments
        train_morphinet(base_args)
        
    except Exception as e:
        print(f"Training failed for sweep run {run_name}: {e}")
        # Log the failure to WandB
        wandb.log({"training_failed": True, "error": str(e)})
        # Mark run as failed
        wandb.run.finish(exit_code=1)
        raise e

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
        
        # Start sweep agent
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