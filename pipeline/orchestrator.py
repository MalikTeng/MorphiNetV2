import os
import torch
import wandb
import time
import gc
from collections import OrderedDict
from trimesh import load
from pytorch3d.structures import Meshes
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
    # Import Open3D rasterizer for voxelization
    from utils.rasterize.voxelize_open3d import VoxelizeOpen3D
except ImportError as e:
    print(f"Import error in orchestrator: {e}")
    print("Make sure all modules are properly installed and accessible")
    raise


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MorphiNetOrchestrator:
    """
    Centralized orchestrator for MorphiNet training pipeline.
    
    Coordinates all training components including data loading, model training,
    validation, and checkpoint management in a clean, modular architecture.
    """
    
    def __init__(self, super_params, seed=42, num_workers=4, is_training=True, **kwargs):
        """
        Initialize the MorphiNet orchestrator.
        
        Args:
            super_params: Configuration parameters
            seed: Random seed for reproducibility  
            num_workers: Number of data loading workers
            is_training: Whether this is for training or inference
            **kwargs: Additional arguments
        """
        self.super_params = super_params
        self.is_training = is_training
        self.seed = seed
        self.num_workers = num_workers
        # Handle backward compatibility for target parameter
        self.dataset = kwargs.get('dataset', kwargs.get('target', None))
        
        # Auto-detect dataset from data paths if not explicitly provided
        if self.dataset is None:
            self.dataset = self._auto_detect_dataset()
        
        # Global step counter for wandb logging consistency
        self.global_step = 0
        
        # Continuous epoch counter for sweep runs (across all phases)
        self.continuous_epoch = 0
        
        # Set deterministic behavior
        set_determinism(seed=seed)
        
        # Initialize directory structure
        if is_training:
            self.ckpt_dir = os.path.join(super_params.ckpt_dir, "dynamic", super_params.run_id)
            os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        if is_training:
            self._initialize_training_components()
        
        print("MorphiNet Orchestrator initialized successfully!")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        print("Initializing MorphiNet components...")
        
        # Initialize data components
        self.dataloader_manager = DataLoaderManager(
            self.super_params, self.num_workers, dataset=self.dataset
        )
        self.preprocessor = DataPreprocessor(self.super_params, dataset=self.dataset)
        
        # Initialize model components
        self._initialize_models()
        self.inference = ModelInference(self.super_params)
        
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
            deep_supervision=True,  # Enable deep supervision for MR
            deep_supr_num=2,       # Number of deep supervision levels
        ).to(DEVICE)
    
        self.encoder_ct = DynUNet(
            spatial_dims=3, in_channels=1,
            out_channels=self.super_params.num_classes,
            kernel_size=ct_kernel_size, 
            strides=ct_strides,
            upsample_kernel_size=ct_upsample_kernel_size, 
            deep_supervision=False,
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
        # Load and process template mesh using trimesh
        template_mesh_trimesh = load(self.super_params.template_mesh_dir)
        
        # Convert Trimesh object to PyTorch3D Meshes object
        template_mesh = Meshes(
            verts=[torch.tensor(template_mesh_trimesh.vertices, dtype=torch.float32)],
            faces=[torch.tensor(template_mesh_trimesh.faces, dtype=torch.int32)]
        ).to(DEVICE)
        
        self.mesh_ops = MeshOperations(self.super_params)
        self.mesh_ops._mesh_label(template_mesh_trimesh)  # Use trimesh object for mesh operations
        self.vert_label = self.mesh_ops.vert_label
        
        # Initialize subdivision with PyTorch3D Meshes object
        self.subdivided_faces = Subdivision(
            template_mesh, self.super_params.subdiv_levels, mesh_label=self.vert_label
        )
        
        # Initialize rasterizer using Trimesh signed-distance field approach
        # raster_size = [int(i // self.super_params.pixdim[0] * self.super_params.upscale_ratio) 
        #                  for i in self.super_params.crop_window_size]
        raster_size = [128, 128, 128]
        
        # Use VoxelizeOpen3D with occupancy method for solid voxelization
        self.rasterizer = VoxelizeOpen3D(shape=raster_size, method='occupancy')
        print("Rasterizer: Open3D raycasting occupancy voxelization")
        
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
            dataloaders=self.dataloader_manager,
            preprocessor=self.preprocessor,
            mesh_ops=self.mesh_ops,
            inference=self.inference,
            orchestrator=self,
            dataset=self.dataset
        )
        
        # Initialize validator
        self.validator = MorphiNetValidator(
            super_params=self.super_params,
            models=self.models,
            dataloaders=self.dataloader_manager,
            preprocessor=self.preprocessor,
            mesh_ops=self.mesh_ops,
            inference=self.inference,
            ckpt_dir=self.ckpt_dir,
            orchestrator=self
        )
        
        print("Training components initialized successfully!")
    
    def prepare_dataloaders(self, data_types=["train"], phase="unet", test_modal="ct"):
        """
        Prepare data loaders for training/validation/testing.
        
        Args:
            data_types: Types of data loaders to prepare
            phase: Training phase for data preparation
            include_test: Whether to include test data loaders
        """
        self.dataloader_manager.prepare_all_dataloaders(
            data_types=data_types,
            phase=phase,
            test_modal=test_modal
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
            self.prepare_dataloaders(["train", "valid"], phase="unet")
        elif phase == "resnet":
            self.prepare_dataloaders(["train", "valid"], phase=phase)
        else:
            self.prepare_dataloaders(["train", "valid"], phase=phase)
        
        # Training loop
        for epoch in range(start_epoch, end_epoch):
            epoch_start_time = time.time()
            
            # Training step
            self.trainer.train_iter(epoch, phase, commit_log=False)
            
            # Validation step (every val_interval epochs)
            if (epoch + 1) % self.super_params.val_interval == 0:
                if phase == "unet":
                    # For UNet phase, validate CT & MR segmentation
                    self.validator.validate_unet(epoch, "ct")
                    self.validator.validate_unet(epoch, "mr")
                elif phase == "resnet":
                    # For ResNet phase, validate UNet + ResNet pipeline on CT data only
                    self.validator.validate_resnet(epoch)
                else:
                    # For GSN phase, validate full pipeline on CT data (GSN is CT-trained)
                    self.validator.validate_gsn(epoch)
            
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
        
        # Phase 1: UNet Training (only if pretrain_epochs > 0)
        if self.super_params.pretrain_epochs > 0:
            print(f"UNet phase: epochs 0 to {self.super_params.pretrain_epochs}")
            self.train_phase("unet", 0, self.super_params.pretrain_epochs)
        else:
            print("UNet phase: SKIPPED (pretrain_epochs = 0)")
        
        # Phase 2: ResNet Training (only if train_epochs > pretrain_epochs)
        if self.super_params.train_epochs > self.super_params.pretrain_epochs:
            print(f"ResNet phase: epochs {self.super_params.pretrain_epochs} to {self.super_params.train_epochs}")
            self.train_phase("resnet", self.super_params.pretrain_epochs, self.super_params.train_epochs)
        else:
            print(f"ResNet phase: SKIPPED (train_epochs={self.super_params.train_epochs} <= pretrain_epochs={self.super_params.pretrain_epochs})")
        
        # Phase 3: GSN Training (only if max_epochs > train_epochs
        if self.super_params.max_epochs > self.super_params.train_epochs:
            print(f"GSN phase: epochs {self.super_params.train_epochs} to {self.super_params.max_epochs}")
            self.train_phase("gsn", self.super_params.train_epochs, self.super_params.max_epochs)
        else:
            if self.super_params.train_epochs <= self.super_params.pretrain_epochs:
                print(f"GSN phase: SKIPPED (train_epochs={self.super_params.train_epochs} <= pretrain_epochs={self.super_params.pretrain_epochs})")
            else:
                print(f"GSN phase: SKIPPED (max_epochs={self.super_params.max_epochs} <= train_epochs={self.super_params.train_epochs})")
        
        # Save final checkpoint after all training phases
        print("\nSaving final checkpoint...")
        self._save_final_checkpoint()
        
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
    
    def _save_final_checkpoint(self):
        """Save final model checkpoint after all training phases complete."""
        ckpt_weight_path = os.path.join(self.ckpt_dir, "trained_weights")
        os.makedirs(ckpt_weight_path, exist_ok=True)
        
        # Determine the last trained epoch based on which phases were executed
        last_epoch = 0
        if self.super_params.max_epochs > self.super_params.train_epochs and self.super_params.train_epochs > self.super_params.pretrain_epochs:
            last_epoch = self.super_params.max_epochs - 1
        elif self.super_params.train_epochs > self.super_params.pretrain_epochs:
            last_epoch = self.super_params.train_epochs - 1
        elif self.super_params.pretrain_epochs > 0:
            last_epoch = self.super_params.pretrain_epochs - 1
        
        # Save final models
        if hasattr(self.models['encoder_ct'], 'state_dict'):
            torch.save(self.models['encoder_ct'].state_dict(), os.path.join(ckpt_weight_path, f"final_UNet_CT.pth"))
        if hasattr(self.models['encoder_mr'], 'state_dict'):
            torch.save(self.models['encoder_mr'].state_dict(), os.path.join(ckpt_weight_path, f"final_UNet_MR.pth"))
        if hasattr(self.models['decoder'], 'state_dict'):
            torch.save(self.models['decoder'].state_dict(), os.path.join(ckpt_weight_path, f"final_ResNet.pth"))
        if hasattr(self.models['GSN'], 'state_dict'):
            torch.save(self.models['GSN'].state_dict(), os.path.join(ckpt_weight_path, f"final_GSN.pth"))
        
        # Save subdivision faces if available
        if hasattr(self, 'mesh_ops') and hasattr(self.mesh_ops, 'subdivided_faces'):
            for level, faces in enumerate(self.mesh_ops.subdivided_faces.faces_levels):
                torch.save(faces, os.path.join(ckpt_weight_path, f"final_subdivided_faces_l{level}.pth"))
        
        print(f"Final checkpoints saved to {ckpt_weight_path}")
    
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
    
    def get_next_step(self):
        """Get the next global step for wandb logging."""
        self.global_step += 1
        return self.global_step
    
    def get_current_step(self):
        """Get the current global step."""
        return self.global_step
    
    def set_step(self, step):
        """Set the global step counter (for synchronization)."""
        self.global_step = step
    
    def get_next_continuous_epoch(self):
        """Get the next continuous epoch for sweep runs (across all phases)."""
        self.continuous_epoch += 1
        return self.continuous_epoch
    
    def get_current_continuous_epoch(self):
        """Get the current continuous epoch."""
        return self.continuous_epoch
    
    def _auto_detect_dataset(self):
        """
        Auto-detect dataset from data directory paths.
        
        Returns:
            str: Dataset identifier ('acdc', 'cap', 'scotheart', 'mmwhs') or None
        """
        print("Auto-detecting dataset from data paths...")
        
        # Check MR data directory for dataset identifiers
        if hasattr(self.super_params, 'mr_data_dir') and self.super_params.mr_data_dir:
            mr_path = self.super_params.mr_data_dir.lower()
            if 'acdc' in mr_path or 'dataset021' in mr_path:
                print(f"Detected ACDC dataset from MR path: {self.super_params.mr_data_dir}")
                return "acdc"
            elif 'cap' in mr_path or 'dataset011' in mr_path:
                print(f"Detected CAP dataset from MR path: {self.super_params.mr_data_dir}")
                return "cap"
        
        # Check CT data directory for dataset identifiers
        if hasattr(self.super_params, 'ct_data_dir') and self.super_params.ct_data_dir:
            ct_path = self.super_params.ct_data_dir.lower()
            if 'scotheart' in ct_path or 'dataset020' in ct_path:
                print(f"Detected SCOTHEART dataset from CT path: {self.super_params.ct_data_dir}")
                return "scotheart"
            elif 'mmwhs' in ct_path or 'dataset022' in ct_path:
                print(f"Detected MMWHS dataset from CT path: {self.super_params.ct_data_dir}")
                return "mmwhs"
        
        # Check JSON file paths as backup
        if hasattr(self.super_params, 'mr_json_dir') and self.super_params.mr_json_dir:
            mr_json = self.super_params.mr_json_dir.lower()
            if 'task21' in mr_json:
                print(f"Detected ACDC dataset from MR JSON: {self.super_params.mr_json_dir}")
                return "acdc"
            elif 'task11' in mr_json:
                print(f"Detected CAP dataset from MR JSON: {self.super_params.mr_json_dir}")
                return "cap"
        
        if hasattr(self.super_params, 'ct_json_dir') and self.super_params.ct_json_dir:
            ct_json = self.super_params.ct_json_dir.lower()
            if 'task20' in ct_json:
                print(f"Detected SCOTHEART dataset from CT JSON: {self.super_params.ct_json_dir}")
                return "scotheart"
            elif 'task22' in ct_json:
                print(f"Detected MMWHS dataset from CT JSON: {self.super_params.ct_json_dir}")
                return "mmwhs"
        
        print("Warning: Could not auto-detect dataset from paths")
        return None