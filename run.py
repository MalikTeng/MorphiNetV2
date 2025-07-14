"""
Refactored MorphiNet training pipeline using modular components.

This file replaces the monolithic run.py with a clean, modular implementation
that separates concerns into specialized modules.
"""

import os
import sys
import time
import wandb
import numpy as np
import torch
from collections import OrderedDict

# Import modular components
from pipeline.orchestrator import MorphiNetOrchestrator
from evaluation.metrics import MorphiNetMetrics
from pipeline.testing import run_full_test

# Configure device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configure PyTorch backends
torch.backends.cudnn.enabled = torch.backends.cudnn.is_available()
torch.backends.cudnn.benchmark = torch.backends.cudnn.is_available()


class MorphiNetPipeline:
    """Simplified MorphiNet pipeline using modular orchestrator."""
    
    def __init__(self, super_params, seed=42, num_workers=4, is_training=True, **kwargs):
        """
        Initialize the MorphiNet pipeline.
        
        Args:
            super_params: Configuration parameters
            seed: Random seed for reproducibility
            num_workers: Number of data loading workers
            is_training: Whether this is for training or inference
            **kwargs: Additional arguments
        """
        self.super_params = super_params
        self.is_training = is_training
        
        # Initialize the orchestrator
        self.orchestrator = MorphiNetOrchestrator(
            super_params=super_params,
            seed=seed,
            num_workers=num_workers,
            is_training=is_training,
            **kwargs
        )
        
        # Initialize metrics for evaluation
        self.metrics = MorphiNetMetrics(
            num_classes=super_params.num_classes,
            include_background=False
        )
        
        # Track training progress
        self.training_history = {
            'unet_loss': OrderedDict({k: np.asarray([]) for k in ["total", "seg"]}),
            'resnet_loss': OrderedDict({k: np.asarray([]) for k in ["total", "df"]}),
            'gsn_loss': OrderedDict({k: np.asarray([]) for k in ["total", "chmf", "smooth"]}),
            'eval_scores': OrderedDict({k: np.asarray([]) for k in ["myo"]}),
        }
    
    def train_full_pipeline(self):
        """
        Train the complete MorphiNet pipeline through all phases.
        
        This method orchestrates the three-stage training:
        1. UNet segmentation training
        2. ResNet distance field prediction training  
        3. GSN mesh refinement training
        """
        print("\n" + "="*80)
        print("STARTING MORPHINET TRAINING WITH MODULAR ARCHITECTURE")
        print("="*80)
        print(f"Device: {DEVICE}")
        print(f"Training phases: UNet -> ResNet -> GSN")
        print(f"Total epochs: {self.super_params.max_epochs}")
        print("="*80)
        
        # Record start time
        pipeline_start_time = time.time()
        
        try:
            # Execute full training pipeline
            self.orchestrator.train_full_pipeline()
            
            # Final evaluation and visualization
            self._generate_final_evaluation()
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise e
        finally:
            # Cleanup resources
            self.orchestrator.cleanup()
        
        # Record completion
        total_time = time.time() - pipeline_start_time
        print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total training time: {total_time/3600:.2f} hours")
    
    def train_phase(self, phase, start_epoch=0, end_epoch=None):
        """
        Train a specific phase of the model.
        
        Args:
            phase: Training phase ('unet', 'resnet', 'gsn')
            start_epoch: Starting epoch
            end_epoch: Ending epoch
        """
        self.orchestrator.train_phase(phase, start_epoch, end_epoch)
    
    def validate(self, epoch, save_on):
        """
        Perform validation using the orchestrator's validator.
        
        Args:
            epoch: Current epoch number
            save_on: Dataset to validate on ('sct' for CT, 'cap' for MR)
        
        Returns:
            Validation metrics
        """
        if not hasattr(self.orchestrator, 'validator'):
            raise RuntimeError("Validator not available. Ensure training mode is enabled.")
        
        return self.orchestrator.validator.validate_gsn(epoch, save_on)
    
    def test(self, test_data_dir=None, output_dir=None):
        """
        Run inference on test data using the modular testing pipeline.
        
        Args:
            test_data_dir: Directory containing test data (optional, uses super_params)
            output_dir: Directory to save results (optional, uses checkpoint dir)
        
        Returns:
            Dictionary containing test results
        """
        print(f"\n--- TESTING PHASE ---")
        
        # Run comprehensive testing using the testing module
        test_results = run_full_test(self, self.super_params)
        
        return test_results
    
    def load_pretrained_weights(self, weights_dir, phase=None):
        """
        Load pretrained weights.
        
        Args:
            weights_dir: Directory containing pretrained weights
            phase: Specific phase to load weights for
        """
        self.orchestrator.load_pretrained_weights(weights_dir, phase)
    
    def _generate_final_evaluation(self):
        """Generate final evaluation plots and summaries."""
        if not self.is_training:
            return
        
        try:
            # Generate training curve plots
            if hasattr(self.orchestrator, 'trainer'):
                self._plot_training_curves()
            
            # Generate final validation report
            if hasattr(self.orchestrator, 'validator'):
                self._generate_validation_report()
            
        except Exception as e:
            print(f"Warning: Could not generate final evaluation: {e}")
    
    def _plot_training_curves(self):
        """Plot training loss curves."""
        # Training curves now tracked in WandB - no need for local plots
        print("Training curves are tracked in WandB dashboard")
    
    def _generate_validation_report(self):
        """Generate final validation report."""
        try:
            validator = self.orchestrator.validator
            
            report = f"""
MORPHINET TRAINING SUMMARY
{'='*50}

Best Validation Score: {validator.best_eval_score:.4f}

Training Configuration:
- UNet Epochs: {self.super_params.pretrain_epochs}
- ResNet Epochs: {self.super_params.train_epochs - self.super_params.pretrain_epochs}
- GSN Epochs: {self.super_params.max_epochs - self.super_params.train_epochs}
- Total Epochs: {self.super_params.max_epochs}
- Learning Rate: {self.super_params.lr}
- Batch Size: {self.super_params.batch_size}

Mesh Parameters:
- Subdivision Levels: {self.super_params.subdiv_levels}
- GSN Hidden Features: {self.super_params.hidden_features_gsn}
- Warping Iterations: {self.super_params.iteration}

Loss Coefficients:
- Chamfer Distance (λ₀): {self.super_params.lambda_0}
- Laplacian Smoothing (λ₁): {self.super_params.lambda_1}

Output Directory: {self.orchestrator.ckpt_dir}
{'='*50}
"""
            
            # Save report
            report_path = os.path.join(self.orchestrator.ckpt_dir, "training_summary.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            
            print(report)
            print(f"Training summary saved to: {report_path}")
            
        except Exception as e:
            print(f"Warning: Could not generate validation report: {e}")
    
    def get_training_metrics(self):
        """
        Get current training metrics.
        
        Returns:
            Dictionary containing training metrics
        """
        if not hasattr(self.orchestrator, 'trainer'):
            return {}
        
        trainer = self.orchestrator.trainer
        
        return {
            'unet_loss': trainer.unet_loss,
            'resnet_loss': trainer.resnet_loss,
            'gsn_loss': trainer.gsn_loss,
        }
    
    def get_validation_metrics(self):
        """
        Get current validation metrics.
        
        Returns:
            Dictionary containing validation metrics
        """
        if not hasattr(self.orchestrator, 'validator'):
            return {}
        
        validator = self.orchestrator.validator
        
        return {
            'best_eval_score': validator.best_eval_score,
            'eval_df_score': validator.eval_df_score,
            'eval_msh_score': validator.eval_msh_score,
        }


def create_training_pipeline(super_params, **kwargs):
    """
    Factory function to create a MorphiNet training pipeline.
    
    Args:
        super_params: Configuration parameters
        **kwargs: Additional arguments for pipeline initialization
    
    Returns:
        Configured MorphiNetPipeline instance
    """
    return MorphiNetPipeline(
        super_params=super_params,
        is_training=True,
        **kwargs
    )


def create_inference_pipeline(super_params, **kwargs):
    """
    Factory function to create a MorphiNet inference pipeline.
    
    Args:
        super_params: Configuration parameters
        **kwargs: Additional arguments for pipeline initialization
    
    Returns:
        Configured MorphiNetPipeline instance for inference
    """
    return MorphiNetPipeline(
        super_params=super_params,
        is_training=False,
        **kwargs
    )


def create_testing_pipeline(super_params, **kwargs):
    """
    Factory function to create a MorphiNet testing pipeline.
    
    This factory creates a MorphiNetPipeline instance specifically configured
    for testing/inference operations. It mirrors the pattern used by
    create_training_pipeline and create_inference_pipeline.
    
    Args:
        super_params: Configuration parameters
        **kwargs: Additional arguments for pipeline initialization
    
    Returns:
        Configured MorphiNetPipeline instance for testing
    """
    return MorphiNetPipeline(
        super_params=super_params,
        is_training=False,
        **kwargs
    )


# Backward compatibility removed - use MorphiNetPipeline only


if __name__ == "__main__":
    # Example usage
    print("MorphiNet Modular Pipeline")
    print("This module provides a refactored, modular implementation of MorphiNet")
    print("Use create_training_pipeline() or create_inference_pipeline() to initialize")