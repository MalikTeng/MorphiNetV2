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
    
    def _prepare_transform(self, keys, modal, section: str, phase: str, **kwargs):
        """Prepare transforms for training and validation data."""
        return pre_transform(
            keys, modal, section,
            self.super_params.crop_window_size,
            self.super_params.pixdim, 
            phase=phase, 
            upscale_ratio=self.super_params.upscale_ratio,
            dataset=self.dataset,
            **kwargs
        )
    
    def _remap_abs_path(self, data_list, modal, section):
        """Remap relative paths to absolute paths."""
        if modal == "mr":
            remapped = []
            for d in data_list:
                img_name = os.path.basename(d["image"])  # e.g., patient001_frame12.nii.gz
                stem = os.path.splitext(os.path.splitext(img_name)[0])[0]
                remapped.append({
                    "mr_image": os.path.join(self.super_params.mr_data_dir, f"images{section}", img_name),
                    "mr_label": os.path.join(self.super_params.mr_data_dir, f"labels{section}", os.path.basename(d["label"])),
                    "mr_case_id": stem,
                })
            return remapped
        elif modal == "ct":
            remapped = []
            for d in data_list:
                img_name = os.path.basename(d["image"])  # e.g., ct_train_1007.nii.gz
                stem = os.path.splitext(os.path.splitext(img_name)[0])[0]
                remapped.append({
                    "ct_image": os.path.join(self.super_params.ct_data_dir, f"images{section}", img_name),
                    "ct_label": os.path.join(self.super_params.ct_data_dir, f"labels{section}", os.path.basename(d["label"])),
                    "ct_case_id": stem,
                })
            return remapped
    
    def _prepare_training_dataloaders(self, phase: str):
        """Prepare training dataloaders based on training phase."""
        if phase == "unet":
            prepare_mr_train = True
            prepare_ct_train = True
        else:  # resnet or gsn
            prepare_mr_train = False
            prepare_ct_train = True
        
        self._prepare_modal_dataloader("mr", "train", prepare_mr_train, phase)
        self._prepare_modal_dataloader("ct", "train", prepare_ct_train, phase)
    
    def _prepare_validation_dataloaders(self, phase: str):
        """Prepare validation dataloaders based on validation phase."""
        if phase == "unet":
            prepare_mr_valid = True
            prepare_ct_valid = True
        else: # resnet or gsn
            prepare_mr_valid = False
            prepare_ct_valid = True
        
        self._prepare_modal_dataloader("mr", "valid", prepare_mr_valid, phase)
        self._prepare_modal_dataloader("ct", "valid", prepare_ct_valid, phase)
    
    def _prepare_test_dataloaders(self, test_modal: str, phase: str):
        """Prepare test dataloaders."""
        self._prepare_modal_dataloader(test_modal, "test", True, phase)
    
    def _prepare_modal_dataloader(self, modal: str, data_type: str, should_prepare: bool, phase: str):
        """
        Prepare a specific modal dataloader for any data type.
        
        Args:
            modal: "mr" or "ct"
            data_type: "train", "valid", or "test"
            should_prepare: Whether to prepare this dataloader
            phase: Phase to use for transforms
        """
        if not should_prepare:
            self._clear_dataloader(modal, data_type)
            return
            
        self._clear_dataloader(modal, data_type)
        
        if data_type == "train":
            data_split, section, batch_size, shuffle = "train_fold0", "Tr", self.super_params.batch_size, True
        elif data_type == "valid":
            data_split, section, batch_size, shuffle = "validation_fold0", "Tr", 1, False
        else:  # test
            data_split, section, batch_size, shuffle = "test", "Ts", 1, False
        
        json_path = self.super_params.mr_json_dir if modal == "mr" else self.super_params.ct_json_dir
        
        with open(json_path, "r") as f:
            transform = self._prepare_transform(
                [f"{modal}_image", f"{modal}_label"], modal, data_type, phase
            )
            
            data_json = json.load(f)
            data_list = self._remap_abs_path(data_json[data_split], modal, section)

            if self.super_params.max_samples > 0:
                data_list = data_list[:self.super_params.max_samples]
            
            dataset = Dataset(
                data=data_list, transform=transform,
                cache_rate=self.super_params.cache_rate, num_workers=self.num_workers
            )
            
            if len(dataset) > 0:
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
                self.mr_train_loader, self.mr_train_ds = dataloader, dataset
            elif data_type == "valid":
                self.mr_valid_loader, self.mr_valid_ds = dataloader, dataset
            elif data_type == "test":
                self.mr_test_loader, self.mr_test_ds = dataloader, dataset
        elif modal == "ct":
            if data_type == "train":
                self.ct_train_loader, self.ct_train_ds = dataloader, dataset
            elif data_type == "valid":
                self.ct_valid_loader, self.ct_valid_ds = dataloader, dataset
            elif data_type == "test":
                self.ct_test_loader, self.ct_test_ds = dataloader, dataset
    
    def _clear_dataloader(self, modal, type_):
        """Clear specific dataloaders and datasets."""
        loaders = {
            "mr": {"train": ("mr_train_loader", "mr_train_ds"), 
                   "valid": ("mr_valid_loader", "mr_valid_ds"), 
                   "test": ("mr_test_loader", "mr_test_ds")},
            "ct": {"train": ("ct_train_loader", "ct_train_ds"), 
                   "valid": ("ct_valid_loader", "ct_valid_ds"), 
                   "test": ("ct_test_loader", "ct_test_ds")}
        }
        if modal in loaders and type_ in loaders[modal]:
            loader_attr, ds_attr = loaders[modal][type_]
            if getattr(self, loader_attr, None):
                delattr(self, loader_attr)
                delattr(self, ds_attr)
                setattr(self, loader_attr, None)
                setattr(self, ds_attr, None)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def prepare_all_dataloaders(self, data_types=["train"], phase="unet", test_modal="ct"):
        """
        Unified function to prepare training, validation, and/or test dataloaders.
        
        Args:
            data_types: List of data types to prepare ["train", "valid", "test"]
            phase: Phase for training data ("unet", "resnet", "gsn")
            test_modal: Modality for test data ("ct", "mr")
        """
        for data_type in data_types:
            if data_type == "train":
                self._prepare_training_dataloaders(phase)
            elif data_type == "valid":
                self._prepare_validation_dataloaders(phase)
            elif data_type == "test":
                self._prepare_test_dataloaders(test_modal, phase)
            else:
                raise ValueError(f"Unknown data_type: {data_type}")