import torch
import torch.nn as nn
from monai.losses import DiceCELoss, MaskedDiceLoss


class LossManager:
    """Manages loss functions and optimizers for MorphiNet training."""
    
    def __init__(self, models, super_params):
        """
        Initialize loss functions, optimizers, and schedulers.
        
        Args:
            models: Dictionary containing model instances
            super_params: Configuration parameters containing learning rate, etc.
        """
        self.super_params = super_params
        self.models = models
        
        # Initialize loss functions
        self._setup_loss_functions()
        
        # Initialize optimizers
        self._setup_optimizers()
        
        # Initialize learning rate schedulers
        self._setup_schedulers()
        
        # Initialize gradient scalers
        self._setup_scalers()
    
    def _setup_loss_functions(self):
        """Initialize all loss functions."""
        # Separate loss functions for CT and MR to avoid shared state
        self.dice_loss_fn_ct = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
        )
        
        self.dice_loss_fn_mr = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
        )
        
        # Additional loss functions
        self.mse_loss_fn = nn.MSELoss()
        self.l1_loss_fn = nn.L1Loss()
        
        # Masked dice loss for ResNet training
        self.msk_dice_loss_fn = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
        )
    
    def _setup_optimizers(self):
        """Initialize optimizers for all models."""
        # UNet optimizers
        self.optimzer_mr_unet = torch.optim.Adam(
            self.models['encoder_mr'].parameters(), 
            lr=self.super_params.lr
        )
        
        self.optimzer_ct_unet = torch.optim.AdamW(
            self.models['encoder_ct'].parameters(), 
            lr=self.super_params.lr
        )
        
        # ResNet optimizer
        self.optimizer_resnet = torch.optim.AdamW(
            self.models['decoder'].parameters(), 
            lr=self.super_params.lr
        )
        
        # GSN optimizer
        self.optimizer_gsn = torch.optim.AdamW(
            self.models['GSN'].parameters(), 
            lr=self.super_params.lr
        )
    
    def _setup_schedulers(self):
        """Initialize learning rate schedulers."""
        scheduler_params = {
            "mode": "min",
            "factor": 0.1,
            "patience": 5,
            "verbose": False,
            "threshold": 1e-2,
            "threshold_mode": "rel",
            "cooldown": int(self.super_params.max_epochs // 50),
            "min_lr": 1e-6,
            "eps": 1e-8
        }
        
        # UNet schedulers
        self.lr_scheduler_mr_unet = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimzer_mr_unet, **scheduler_params
        )
        
        self.lr_scheduler_ct_unet = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimzer_ct_unet, **scheduler_params
        )
        
        # ResNet scheduler
        self.lr_scheduler_resnet = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_resnet, **scheduler_params
        )
        
        # GSN scheduler
        self.lr_scheduler_gsn = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gsn, **scheduler_params
        )
    
    def _setup_scalers(self):
        """Initialize gradient scalers for mixed precision training."""
        self.scaler_mr_unet = torch.cuda.amp.GradScaler()
        self.scaler_ct_unet = torch.cuda.amp.GradScaler()
        self.scaler_resnet = torch.cuda.amp.GradScaler()
        self.scaler_gsn = torch.cuda.amp.GradScaler()
    
    def get_loss_functions(self):
        """
        Get dictionary of loss functions.
        
        Returns:
            Dictionary containing all loss functions
        """
        return {
            'dice_ct': self.dice_loss_fn_ct,
            'dice_mr': self.dice_loss_fn_mr,
            'mse': self.mse_loss_fn,
            'l1': self.l1_loss_fn,
            'masked_dice': self.msk_dice_loss_fn,
        }
    
    def get_optimizers(self):
        """
        Get dictionary of optimizers.
        
        Returns:
            Dictionary containing all optimizers
        """
        return {
            'mr_unet': self.optimzer_mr_unet,
            'ct_unet': self.optimzer_ct_unet,
            'resnet': self.optimizer_resnet,
            'gsn': self.optimizer_gsn,
        }
    
    def get_schedulers(self):
        """
        Get dictionary of learning rate schedulers.
        
        Returns:
            Dictionary containing all schedulers
        """
        return {
            'mr_unet': self.lr_scheduler_mr_unet,
            'ct_unet': self.lr_scheduler_ct_unet,
            'resnet': self.lr_scheduler_resnet,
            'gsn': self.lr_scheduler_gsn,
        }
    
    def get_scalers(self):
        """
        Get dictionary of gradient scalers.
        
        Returns:
            Dictionary containing all scalers
        """
        return {
            'mr_unet': self.scaler_mr_unet,
            'ct_unet': self.scaler_ct_unet,
            'resnet': self.scaler_resnet,
            'gsn': self.scaler_gsn,
        }
    
    def get_current_lr(self, optimizer_name):
        """
        Get current learning rate for specified optimizer.
        
        Args:
            optimizer_name: Name of optimizer ('mr_unet', 'ct_unet', 'resnet', 'gsn')
        
        Returns:
            Current learning rate
        """
        optimizer_map = {
            'mr_unet': self.optimzer_mr_unet,
            'ct_unet': self.optimzer_ct_unet,
            'resnet': self.optimizer_resnet,
            'gsn': self.optimizer_gsn,
        }
        
        if optimizer_name not in optimizer_map:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer_map[optimizer_name].param_groups[0]['lr']