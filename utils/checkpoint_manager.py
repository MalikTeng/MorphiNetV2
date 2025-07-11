import os
import torch
import glob
from collections import OrderedDict


class CheckpointManager:
    """Manages model checkpoints for MorphiNet."""
    
    def __init__(self, checkpoint_dir, models=None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save/load checkpoints
            models: Dictionary of models to manage
        """
        self.checkpoint_dir = checkpoint_dir
        self.models = models or {}
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, epoch, models=None, optimizers=None, schedulers=None, 
                       additional_data=None, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            models: Dictionary of models to save
            optimizers: Dictionary of optimizers to save
            schedulers: Dictionary of schedulers to save
            additional_data: Additional data to save (metrics, etc.)
            is_best: Whether this is the best model so far
        """
        models = models or self.models
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'models': {},
            'optimizers': {},
            'schedulers': {},
        }
        
        # Save model state dicts
        for name, model in models.items():
            checkpoint_data['models'][name] = model.state_dict()
        
        # Save optimizer state dicts
        if optimizers:
            for name, optimizer in optimizers.items():
                checkpoint_data['optimizers'][name] = optimizer.state_dict()
        
        # Save scheduler state dicts
        if schedulers:
            for name, scheduler in schedulers.items():
                checkpoint_data['schedulers'][name] = scheduler.state_dict()
        
        # Add additional data
        if additional_data:
            checkpoint_data.update(additional_data)
        
        # Save checkpoint
        if is_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path=None, load_best=False, epoch=None):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint path to load
            load_best: Whether to load the best checkpoint
            epoch: Specific epoch to load
        
        Returns:
            Loaded checkpoint data
        """
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            elif epoch is not None:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            else:
                # Load latest checkpoint
                checkpoint_path = self.get_latest_checkpoint()
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint_data
    
    def load_models(self, checkpoint_data, models=None, strict=True):
        """
        Load model states from checkpoint data.
        
        Args:
            checkpoint_data: Checkpoint data containing model states
            models: Dictionary of models to load states into
            strict: Whether to strictly enforce state dict keys match
        """
        models = models or self.models
        
        if 'models' not in checkpoint_data:
            raise ValueError("Checkpoint data does not contain model states")
        
        for name, model in models.items():
            if name in checkpoint_data['models']:
                model.load_state_dict(checkpoint_data['models'][name], strict=strict)
                print(f"Loaded state for model: {name}")
            else:
                print(f"Warning: Model '{name}' not found in checkpoint")
    
    def load_optimizers(self, checkpoint_data, optimizers):
        """
        Load optimizer states from checkpoint data.
        
        Args:
            checkpoint_data: Checkpoint data containing optimizer states
            optimizers: Dictionary of optimizers to load states into
        """
        if 'optimizers' not in checkpoint_data:
            print("Warning: Checkpoint data does not contain optimizer states")
            return
        
        for name, optimizer in optimizers.items():
            if name in checkpoint_data['optimizers']:
                optimizer.load_state_dict(checkpoint_data['optimizers'][name])
                print(f"Loaded state for optimizer: {name}")
            else:
                print(f"Warning: Optimizer '{name}' not found in checkpoint")
    
    def load_schedulers(self, checkpoint_data, schedulers):
        """
        Load scheduler states from checkpoint data.
        
        Args:
            checkpoint_data: Checkpoint data containing scheduler states
            schedulers: Dictionary of schedulers to load states into
        """
        if 'schedulers' not in checkpoint_data:
            print("Warning: Checkpoint data does not contain scheduler states")
            return
        
        for name, scheduler in schedulers.items():
            if name in checkpoint_data['schedulers']:
                scheduler.load_state_dict(checkpoint_data['schedulers'][name])
                print(f"Loaded state for scheduler: {name}")
            else:
                print(f"Warning: Scheduler '{name}' not found in checkpoint")
    
    def get_latest_checkpoint(self):
        """
        Get path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint
        """
        checkpoint_pattern = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoints found")
        
        # Extract epoch numbers and find the latest
        epochs = []
        for file in checkpoint_files:
            try:
                epoch = int(os.path.basename(file).split('_')[-1].split('.')[0])
                epochs.append((epoch, file))
            except ValueError:
                continue
        
        if not epochs:
            raise FileNotFoundError("No valid checkpoints found")
        
        # Return path to checkpoint with highest epoch number
        latest_epoch, latest_path = max(epochs, key=lambda x: x[0])
        return latest_path
    
    def list_checkpoints(self):
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint paths
        """
        checkpoint_pattern = os.path.join(self.checkpoint_dir, '*.pth')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        return sorted(checkpoint_files)
    
    def load_pretrained_weights(self, weights_dir, models=None, phase=None):
        """
        Load pretrained weights from individual model files.
        
        Args:
            weights_dir: Directory containing pretrained weight files
            models: Dictionary of models to load weights into
            phase: Specific training phase ('unet', 'resnet', 'gsn', or None for all)
        """
        models = models or self.models
        
        if not os.path.exists(weights_dir):
            print(f"Warning: Weights directory not found: {weights_dir}")
            return
        
        # Define weight file mappings
        weight_mappings = {
            'encoder_ct': ['best_UNet_CT.pth', 'UNet_CT.pth'],
            'encoder_mr': ['best_UNet_MR.pth', 'UNet_MR.pth'],
            'decoder': ['best_ResNet.pth', 'ResNet.pth'],
            'GSN': ['best_GSN.pth', 'GSN.pth'],
        }
        
        # Phase-specific loading
        if phase == 'unet':
            models_to_load = ['encoder_ct', 'encoder_mr']
        elif phase == 'resnet':
            models_to_load = ['encoder_ct', 'encoder_mr', 'decoder']
        elif phase == 'gsn':
            models_to_load = ['encoder_ct', 'encoder_mr', 'decoder', 'GSN']
        else:
            models_to_load = list(models.keys())
        
        for model_name in models_to_load:
            if model_name not in models:
                continue
            
            model = models[model_name]
            weight_files = weight_mappings.get(model_name, [f'{model_name}.pth'])
            
            # Try to load weights (prefer 'best' weights if available)
            loaded = False
            for weight_file in weight_files:
                weight_path = os.path.join(weights_dir, weight_file)
                if os.path.exists(weight_path):
                    try:
                        state_dict = torch.load(weight_path, map_location='cpu')
                        
                        # Check if keys match between model and checkpoint
                        model_keys = set(model.state_dict().keys())
                        checkpoint_keys = set(state_dict.keys())
                        
                        missing_keys = model_keys - checkpoint_keys
                        unexpected_keys = checkpoint_keys - model_keys
                        
                        if missing_keys:
                            print(f"WARNING: Missing keys in checkpoint for {model_name}: {len(missing_keys)} keys")
                        if unexpected_keys:
                            print(f"WARNING: Unexpected keys in checkpoint for {model_name}: {len(unexpected_keys)} keys")
                        
                        # Load with strict=False to handle architecture mismatches
                        missing_keys_actual, unexpected_keys_actual = model.load_state_dict(state_dict, strict=False)
                        
                        if missing_keys_actual:
                            print(f"INFO: {model_name} missing keys: {len(missing_keys_actual)}")
                        if unexpected_keys_actual:
                            print(f"INFO: {model_name} unexpected keys: {len(unexpected_keys_actual)}")
                        
                        print(f"Loaded pretrained weights for {model_name}: {weight_file}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load {weight_path}: {e}")
            
            if not loaded:
                print(f"Warning: No pretrained weights found for {model_name}")
    
    def cleanup_old_checkpoints(self, keep_last_n=5):
        """
        Remove old checkpoints, keeping only the last N and the best checkpoint.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoint_pattern = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) <= keep_last_n:
            return
        
        # Extract epoch numbers and sort
        epochs_files = []
        for file in checkpoint_files:
            try:
                epoch = int(os.path.basename(file).split('_')[-1].split('.')[0])
                epochs_files.append((epoch, file))
            except ValueError:
                continue
        
        epochs_files.sort(key=lambda x: x[0])
        
        # Remove old checkpoints (keep last N)
        to_remove = epochs_files[:-keep_last_n]
        
        for epoch, file_path in to_remove:
            try:
                os.remove(file_path)
                print(f"Removed old checkpoint: {os.path.basename(file_path)}")
            except OSError as e:
                print(f"Failed to remove {file_path}: {e}")