import os
import json
import gc
import torch
from monai.data import DataLoader, CacheDataset as Dataset
from monai.transforms import Compose
from data.transforms import pre_transform
from data.dataset_utils import collate_4D_batch


class DataLoaderManager:
    """Manages data loading for MorphiNet training, validation, and testing."""
    
    def __init__(self, super_params, num_workers=4, target=None, dataset=None):
        """
        Initialize the DataLoader Manager.
        
        Args:
            super_params: Configuration parameters
            num_workers: Number of workers for data loading
            target: Target for transformation (deprecated, use dataset)
            dataset: Dataset name for transformation
        """
        self.super_params = super_params
        self.num_workers = num_workers
        # Handle backward compatibility
        self.dataset = dataset if dataset is not None else target
        
        # Initialize dataloader attributes
        self.mr_train_loader, self.mr_valid_loader, self.mr_test_loader = None, None, None
        self.ct_train_loader, self.ct_valid_loader, self.ct_test_loader = None, None, None
        self.mr_train_ds, self.mr_valid_ds, self.mr_test_ds = None, None, None
        self.ct_train_ds, self.ct_valid_ds, self.ct_test_ds = None, None, None
    
    def _prepare_transform(self, keys, modal, training_phase: str, **kwargs):
        """Prepare transforms for training and validation data."""
        train_transform = pre_transform(
            keys, modal, "train",
            self.super_params.crop_window_size,
            self.super_params.pixdim, phase=training_phase, 
            upscale_ratio=self.super_params.upscale_ratio, **kwargs
        )
        valid_transform = pre_transform(
            keys, modal, "valid",
            self.super_params.crop_window_size,
            self.super_params.pixdim, phase=training_phase, 
            upscale_ratio=self.super_params.upscale_ratio, **kwargs
        )
        return train_transform, valid_transform
    
    def _remap_abs_path(self, data_list, modal, phase):
        """Remap relative paths to absolute paths."""
        if modal == "mr":
            return [{
                "mr_image": os.path.join(self.super_params.mr_data_dir, f"images{phase}", os.path.basename(d["image"])),
                "mr_label": os.path.join(self.super_params.mr_data_dir, f"labels{phase}", os.path.basename(d["label"])),
            } for d in data_list]
        elif modal == "ct":
            return [{
                "ct_image": os.path.join(self.super_params.ct_data_dir, f"images{phase}", os.path.split(d["image"])[-1]),
                "ct_label": os.path.join(self.super_params.ct_data_dir, f"labels{phase}", os.path.split(d["label"])[-1]),
            } for d in data_list]
    
    def _prepare_training_dataloaders(self, training_phase: str):
        """Prepare training dataloaders based on training phase."""
        # UNet phase: load both MR and CT data for training encoders
        # ResNet/GSN phases: load only CT data
        prepare_mr_train = (training_phase == "unet")
        prepare_ct_train = True  # CT data always needed for training
        
        self._prepare_modal_dataloader("mr", "train", prepare_mr_train, training_phase)
        self._prepare_modal_dataloader("ct", "train", prepare_ct_train, training_phase)
    
    def _prepare_validation_dataloaders(self, validation_phase: str):
        """Prepare validation dataloaders based on validation phase."""
        # UNet phase: validate both MR and CT encoders
        # ResNet/GSN phases: validate based on save_on parameter
        prepare_mr_valid = (validation_phase == "unet") or self.super_params.validation_modality == "mr"
        prepare_ct_valid = (validation_phase == "unet") or self.super_params.validation_modality == "ct"
        
        transform_phase = "validation" if validation_phase == "network" else validation_phase
        
        self._prepare_modal_dataloader("mr", "valid", prepare_mr_valid, transform_phase)
        self._prepare_modal_dataloader("ct", "valid", prepare_ct_valid, transform_phase)
    
    def _prepare_test_dataloaders(self):
        """Prepare test dataloaders."""
        # Test data loading based on save_on parameter
        prepare_mr_test = self.super_params.validation_modality == "mr"
        prepare_ct_test = self.super_params.validation_modality == "ct"
        
        self._prepare_modal_dataloader("mr", "test", prepare_mr_test, "validation")
        self._prepare_modal_dataloader("ct", "test", prepare_ct_test, "validation")
    
    def _prepare_modal_dataloader(self, modal: str, data_type: str, should_prepare: bool, transform_phase: str = "validation"):
        """
        Prepare a specific modal dataloader for any data type.
        
        Args:
            modal: "mr" or "ct"
            data_type: "train", "valid", or "test"
            should_prepare: Whether to prepare this dataloader
            transform_phase: Phase to use for transforms
        """
        if not should_prepare:
            self._clear_dataloader(modal, data_type)
            return
            
        # Clear existing dataloader
        self._clear_dataloader(modal, data_type)
        
        # Determine data split and phase suffix based on data type
        if data_type == "train":
            data_split = "train_fold0"
            phase_suffix = "Tr"
            batch_size = self.super_params.batch_size
            shuffle = True
            print_msg = f"Preparing {modal.upper()} training data for phase {transform_phase}..."
        elif data_type == "valid":
            data_split = "validation_fold0"
            phase_suffix = "Tr"
            batch_size = 1
            shuffle = False
            print_msg = f"Preparing {modal.upper()} validation data..."
        else:  # test
            data_split = "test"
            phase_suffix = "Ts"
            batch_size = 1
            shuffle = False
            print_msg = f"Preparing {modal.upper()} test data..."
        
        # Get JSON file path
        json_path = self.super_params.mr_json_dir if modal == "mr" else self.super_params.ct_json_dir
        
        with open(json_path, "r") as f:
            # Choose appropriate transform based on data type
            if data_type == "train":
                transform, _ = self._prepare_transform(
                    [f"{modal}_image", f"{modal}_label"], modal, 
                    dataset=self.dataset, training_phase=transform_phase
                )
            else:  # valid or test
                _, transform = self._prepare_transform(
                    [f"{modal}_image", f"{modal}_label"], modal, 
                    dataset=self.dataset, training_phase=transform_phase
                )
            
            data_json = json.load(f)
            data_list = self._remap_abs_path(data_json[data_split], modal, phase_suffix)
            
            # Apply sample limiting if specified
            if self.super_params.max_samples > 0:
                data_list = data_list[:self.super_params.max_samples]
            
            # Create dataset
            dataset = Dataset(
                data=data_list, transform=transform,
                cache_rate=self.super_params.cache_rate, num_workers=self.num_workers
            )
            
            # Create and assign dataloader
            if dataset.__len__() > 0:
                dataloader = DataLoader(
                    dataset, batch_size=batch_size, shuffle=shuffle, 
                    num_workers=self.num_workers, collate_fn=collate_4D_batch
                )
                self._assign_dataloader(modal, data_type, dataloader, dataset)
            else:
                self._assign_dataloader(modal, data_type, None, None)
    
    def _assign_dataloader(self, modal: str, data_type: str, dataloader, dataset):
        """Helper to assign dataloader and dataset to correct attributes."""
        if modal == "mr":
            if data_type == "train":
                self.mr_train_loader = dataloader
                self.mr_train_ds = dataset
            elif data_type == "valid":
                self.mr_valid_loader = dataloader
                self.mr_valid_ds = dataset
            elif data_type == "test":
                self.mr_test_loader = dataloader
                self.mr_test_ds = dataset
        elif modal == "ct":
            if data_type == "train":
                self.ct_train_loader = dataloader
                self.ct_train_ds = dataset
            elif data_type == "valid":
                self.ct_valid_loader = dataloader
                self.ct_valid_ds = dataset
            elif data_type == "test":
                self.ct_test_loader = dataloader
                self.ct_test_ds = dataset
    
    def _clear_dataloader(self, modal, type_):
        """Clear specific dataloaders and datasets."""
        if modal == "mr":
            if type_ == "train" and self.mr_train_loader:
                del self.mr_train_loader
                del self.mr_train_ds
                self.mr_train_loader, self.mr_train_ds = None, None
            elif type_ == "valid" and self.mr_valid_loader:
                del self.mr_valid_loader
                del self.mr_valid_ds
                self.mr_valid_loader, self.mr_valid_ds = None, None
            elif type_ == "test" and self.mr_test_loader:
                del self.mr_test_loader
                del self.mr_test_ds
                self.mr_test_loader, self.mr_test_ds = None, None
        elif modal == "ct":
            if type_ == "train" and self.ct_train_loader:
                del self.ct_train_loader
                del self.ct_train_ds
                self.ct_train_loader, self.ct_train_ds = None, None
            elif type_ == "valid" and self.ct_valid_loader:
                del self.ct_valid_loader
                del self.ct_valid_ds
                self.ct_valid_loader, self.ct_valid_ds = None, None
            elif type_ == "test" and self.ct_test_loader:
                del self.ct_test_loader
                del self.ct_test_ds
                self.ct_test_loader, self.ct_test_ds = None, None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def prepare_all_dataloaders(self, data_types=["train"], training_phase="unet", validation_phase="network", include_test=False):
        """
        Unified function to prepare training, validation, and/or test dataloaders.
        
        Args:
            data_types: List of data types to prepare ["train", "valid", "test"]
            training_phase: Phase for training data ("unet", "resnet", "gsn")
            validation_phase: Phase for validation ("unet", "resnet", "gsn", "network")
            include_test: Whether to also prepare test dataloaders
        """
        # Handle include_test parameter
        if include_test and "test" not in data_types:
            data_types = data_types + ["test"]
        
        # Prepare each requested data type
        for data_type in data_types:
            if data_type == "train":
                self._prepare_training_dataloaders(training_phase)
            elif data_type == "valid":
                self._prepare_validation_dataloaders(validation_phase)
            elif data_type == "test":
                self._prepare_test_dataloaders()
            else:
                raise ValueError(f"Unknown data_type: {data_type}")