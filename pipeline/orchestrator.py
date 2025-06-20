import os
import torch
import wandb
import time
import gc
from collections import OrderedDict
from trimesh import load
from monai.utils import set_determinism
from monai.networks.nets import DynUNet

# Import modular components
try:
    from data.loaders import DataLoaderManager
    from data.preprocessors import DataPreprocessor
    from model.mesh_operations import MeshOperations
    from model.inference import ModelInference
    from model.networks import UpscalingResNet, GSN, Subdivision
    from training.trainer import MorphiNetTrainer
    from training.validators import MorphiNetValidator
    from training.losses import LossManager
    from utils.checkpoint_manager import CheckpointManager
    from utils.rasterize.rasterize import Rasterize
except ImportError as e:
    print(f"Import error in orchestrator: {e}")
    print("Make sure all modules are properly installed and accessible")
    raise


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MorphiNetOrchestrator:
    """Main orchestrator for MorphiNet training pipeline."""
    
    def __init__(self, super_params, seed=42, num_workers=4, is_training=True, **kwargs):
        """
        Initialize the MorphiNet orchestrator.
        
        Args:
            super_params: Configuration parameters
            seed: Random seed
            num_workers: Number of data loading workers
            is_training: Whether this is for training or inference
            **kwargs: Additional arguments
        """
        self.super_params = super_params
        self.seed = seed
        self.num_workers = num_workers
        self.is_training = is_training
        self.target = kwargs.get("target")
        
        # Set deterministic behavior
        set_determinism(seed=seed)
        
        # Initialize directory structure
        if is_training:
            self.ckpt_dir = os.path.join(super_params.ckpt_dir, "dynamic", super_params.run_id)
            os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        print("Initializing MorphiNet components...")
        
        # Initialize data components
        self.dataloader_manager = DataLoaderManager(
            self.super_params, self.num_workers, self.target
        )
        self.preprocessor = DataPreprocessor(self.super_params)
        
        # Initialize model components
        self._initialize_models()
        self.mesh_ops = MeshOperations(self.super_params, self.vert_label)
        self.inference = ModelInference(self.super_params)
        
        # Initialize training components (only if training)
        if self.is_training:
            self._initialize_training_components()
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            self.ckpt_dir if self.is_training else "./checkpoints",
            self.models
        )
        
        print("Component initialization complete!")
    
    def _initialize_models(self):
        """Initialize neural network models."""
        print("Initializing neural network models...")
        
        # Prepare kernel sizes for different spatial dimensions
        mr_kernel_size = [(k, k) for k in self.super_params.kernel_size]
        mr_strides = [(s, s) for s in self.super_params.strides]
        mr_upsample_kernel_size = [(s, s) for s in self.super_params.strides[1:]]
        
        ct_kernel_size = [(k, k, k) for k in self.super_params.kernel_size]
        ct_strides = [(s, s, s) for s in self.super_params.strides]
        ct_upsample_kernel_size = [(s, s, s) for s in self.super_params.strides[1:]]
        
        # Initialize UNet encoders
        self.encoder_mr = DynUNet(
            spatial_dims=2, in_channels=1,
            out_channels=self.super_params.num_classes,
            kernel_size=mr_kernel_size, 
            strides=mr_strides,
            upsample_kernel_size=mr_upsample_kernel_size, 
            filters=self.super_params.filters, 
            dropout=False,
            deep_supervision=False,
            res_block=True
        ).to(DEVICE)
        
        self.encoder_ct = DynUNet(
            spatial_dims=3, in_channels=1,
            out_channels=self.super_params.num_classes,
            kernel_size=ct_kernel_size, 
            strides=ct_strides,
            upsample_kernel_size=ct_upsample_kernel_size, 
            filters=self.super_params.filters, 
            dropout=False,
            deep_supervision=False,
            res_block=True
        ).to(DEVICE)
        
        # Initialize ResNet decoder
        self.decoder = UpscalingResNet(
            spatial_dims=3,
            in_channels=self.super_params.num_classes,
            out_channels=self.super_params.num_classes,
            upscale_ratio=self.super_params.upscale_ratio,
            layers=self.super_params.layers,
            act=("leakyrelu", {"inplace": True, "negative_slope": 0.1}),
            norm=("INSTANCE", {"affine": True}),
        ).to(DEVICE)
        
        # Initialize template mesh and subdivision
        self._initialize_mesh_components()
        
        # Initialize GSN
        self.GSN = GSN(
            hidden_features=self.super_params.hidden_features_gsn, 
            num_layers=self.super_params.subdiv_levels if self.super_params.subdiv_levels > 0 else 2,
            num_iterations=self.super_params.iteration,
        ).to(DEVICE)
        
        # Store models in dictionary
        self.models = {
            'encoder_mr': self.encoder_mr,
            'encoder_ct': self.encoder_ct,
            'decoder': self.decoder,
            'GSN': self.GSN,
        }
        
        print("Neural network models initialized successfully!")
    
    def _initialize_mesh_components(self):
        """Initialize mesh-related components."""
        # Load and process template mesh
        template_mesh = load(self.super_params.template_mesh_dir)
        self.mesh_ops = MeshOperations(self.super_params)
        self.mesh_ops._mesh_label(template_mesh)
        self.vert_label = self.mesh_ops.vert_label
        
        # Initialize subdivision
        self.subdivided_faces = Subdivision(
            template_mesh, self.super_params.subdiv_levels, mesh_label=self.vert_label
        )
        
        # Initialize rasterizer
        raster_size = [int(i // self.super_params.pixdim[0] * self.super_params.upscale_ratio) 
                      for i in self.super_params.crop_window_size]
        self.rasterizer = Rasterize(raster_size)
        
        # Store subdivision in mesh_ops for access by other components
        self.mesh_ops.subdivided_faces = self.subdivided_faces
        self.mesh_ops.rasterizer = self.rasterizer
    
    def _initialize_training_components(self):
        """Initialize training-specific components."""
        print("Initializing training components...")
        
        # Initialize loss manager
        self.loss_manager = LossManager(self.models, self.super_params)
        
        # Get components from loss manager
        loss_functions = self.loss_manager.get_loss_functions()
        optimizers = self.loss_manager.get_optimizers()
        schedulers = self.loss_manager.get_schedulers()
        scalers = self.loss_manager.get_scalers()
        
        # Initialize trainer
        self.trainer = MorphiNetTrainer(
            super_params=self.super_params,
            models=self.models,
            optimizers=optimizers,
            schedulers=schedulers,
            scalers=scalers,
            loss_functions=loss_functions,
            dataloaders=self.dataloader_manager.__dict__,
            preprocessor=self.preprocessor,
            mesh_ops=self.mesh_ops,
            inference=self.inference,
            target=self.target
        )
        
        # Initialize validator
        self.validator = MorphiNetValidator(
            super_params=self.super_params,
            models=self.models,
            dataloaders=self.dataloader_manager.__dict__,
            preprocessor=self.preprocessor,
            mesh_ops=self.mesh_ops,
            inference=self.inference,
            ckpt_dir=self.ckpt_dir
        )
        
        print("Training components initialized successfully!")
    
    def prepare_dataloaders(self, data_types=["train"], training_phase="unet", 
                          validation_phase="network", include_test=False):
        """
        Prepare data loaders for training/validation/testing.
        
        Args:
            data_types: Types of data loaders to prepare
            training_phase: Training phase for data preparation
            validation_phase: Validation phase for data preparation
            include_test: Whether to include test data loaders
        """
        self.dataloader_manager.prepare_all_dataloaders(
            data_types=data_types,
            training_phase=training_phase,
            validation_phase=validation_phase,
            include_test=include_test
        )
    
    def train_phase(self, phase, start_epoch=0, end_epoch=None):
        """
        Train a specific phase of the model.
        
        Args:
            phase: Training phase ('unet', 'resnet', 'gsn')
            start_epoch: Starting epoch
            end_epoch: Ending epoch (if None, uses phase-specific defaults)
        """
        if not self.is_training:
            raise RuntimeError("Orchestrator not initialized for training")
        
        # Determine epoch range based on phase
        if end_epoch is None:
            if phase == "unet":
                end_epoch = self.super_params.pretrain_epochs
            elif phase == "resnet":
                end_epoch = self.super_params.train_epochs
            elif phase == "gsn":
                end_epoch = self.super_params.max_epochs
            else:
                raise ValueError(f"Unknown phase: {phase}")
        
        print(f"\n{'='*80}")
        print(f"STARTING {phase.upper()} TRAINING")
        print(f"Epochs: {start_epoch} to {end_epoch}")
        print(f"{'='*80}")
        
        # Prepare appropriate data loaders
        if phase == "unet":
            self.prepare_dataloaders(["train", "valid"], training_phase="unet", validation_phase="unet")
        else:
            self.prepare_dataloaders(["train", "valid"], training_phase=phase, validation_phase="network")
        
        # Training loop
        for epoch in range(start_epoch, end_epoch):
            epoch_start_time = time.time()
            
            # Training step
            self.trainer.train_iter(epoch, phase, commit_log=False)
            
            # Validation step (every val_interval epochs)
            if (epoch + 1) % self.super_params.val_interval == 0:
                if phase == "unet":
                    # For UNet phase, validate segmentation only
                    self.validator.validate_segmentation(epoch, self.super_params.save_on)
                else:
                    # For ResNet/GSN phases, validate full pipeline
                    self.validator.validate(epoch, self.super_params.save_on)
            
            # Epoch timing
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            
            # Memory cleanup
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        
        print(f"\n{phase.upper()} TRAINING COMPLETED!")
    
    def train_full_pipeline(self):
        """Train the complete MorphiNet pipeline through all phases."""
        print("\n" + "="*80)
        print("STARTING FULL MORPHINET TRAINING PIPELINE")
        print("="*80)
        
        # Phase 1: UNet Training
        self.train_phase("unet", 0, self.super_params.pretrain_epochs)
        
        # Phase 2: ResNet Training
        self.train_phase("resnet", self.super_params.pretrain_epochs, self.super_params.train_epochs)
        
        # Phase 3: GSN Training
        self.train_phase("gsn", self.super_params.train_epochs, self.super_params.max_epochs)
        
        print("\n" + "="*80)
        print("FULL MORPHINET TRAINING PIPELINE COMPLETED!")
        print("="*80)
    
    def load_pretrained_weights(self, weights_dir, phase=None):
        """
        Load pretrained weights for models.
        
        Args:
            weights_dir: Directory containing pretrained weights
            phase: Specific phase to load weights for
        """
        self.checkpoint_manager.load_pretrained_weights(weights_dir, self.models, phase)
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save current model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model
        """
        if not self.is_training:
            return
        
        additional_data = {
            'best_eval_score': getattr(self.validator, 'best_eval_score', 0.0),
            'super_params': self.super_params.__dict__,
        }
        
        self.checkpoint_manager.save_checkpoint(
            epoch=epoch,
            models=self.models,
            optimizers=self.loss_manager.get_optimizers() if hasattr(self, 'loss_manager') else None,
            schedulers=self.loss_manager.get_schedulers() if hasattr(self, 'loss_manager') else None,
            additional_data=additional_data,
            is_best=is_best
        )
    
    def cleanup(self):
        """Clean up resources."""
        # Clear data loaders
        if hasattr(self, 'dataloader_manager'):
            del self.dataloader_manager
        
        # Clear models
        if hasattr(self, 'models'):
            for model in self.models.values():
                del model
        
        # Force garbage collection
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        print("Pipeline cleanup completed!")