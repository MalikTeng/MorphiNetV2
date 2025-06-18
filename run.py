import os, sys, json, glob, tqdm, time
from collections import OrderedDict
from itertools import chain
from trimesh import Trimesh, load
from trimesh.convex import convex_hull
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import save_obj
from pytorch3d.ops import sample_points_from_meshes, taubin_smoothing
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops.marching_cubes import marching_cubes
from monai.data import DataLoader, CacheDataset as Dataset
from monai.losses import DiceCELoss, MaskedDiceLoss
from monai.metrics import DiceMetric, MSEMetric
from monai.networks.nets import DynUNet, SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, 
    AsDiscrete,
    KeepLargestConnectedComponent,
    RemoveSmallObjects,
    CropForegroundd,
    Resized,
    Spacingd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    EnsureTyped, 
)
from monai.transforms.utils import distance_transform_edt, generate_spatial_bounding_box
from monai.utils import set_determinism
# from einops import rearrange
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
import wandb
import plotly.figure_factory as ff
import gc # Import garbage collector
from scipy.ndimage import binary_dilation

from data.transform import pre_transform
from data.components import Maskd, FlexResized
from data.dataset import collate_4D_batch
from utils import *
from model.networks import *

from utils.rasterize.rasterize import Rasterize


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TrainPipeline:
    def __init__(
            self,
            super_params,
            seed, num_workers,
            is_training=True,
            **kwargs
        ):
        """
        :param 
            super_params: parameters for setting up dataset, network structure, training, etc.
            seed: random seed to shuffle data during augmentation.
            num_workers: tourch.utils.data.DataLoader num_workers.
            is_training: switcher for training (True, default) or testing (False).
        """
            
        self.super_params = super_params
        self.seed = seed
        self.num_workers = num_workers
        self.is_training = is_training
        self.target = kwargs.get("target")
        self.sigmoid_scale_factor = super_params.sigmoid_scale_factor # Store the new parameter
        self.mask_threshold = super_params.mask_threshold # Store the mask threshold
        set_determinism(seed=self.seed)

        # Initialize dataloader attributes to None
        self.mr_train_loader, self.mr_valid_loader, self.mr_test_loader = None, None, None
        self.ct_train_loader, self.ct_valid_loader, self.ct_test_loader = None, None, None
        self.mr_train_ds, self.mr_valid_ds, self.mr_test_ds = None, None, None
        self.ct_train_ds, self.ct_valid_ds, self.ct_test_ds = None, None, None

        if is_training:
            self.ckpt_dir = os.path.join(super_params.ckpt_dir, "dynamic", super_params.run_id)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.unet_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "seg"]}
            )
            self.resnet_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "df"]}
            )
            self.gsn_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "chmf", "smooth"]}
            )
            self.ndf_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "ndf"]}
            )
            self.eval_df_score = OrderedDict(
                {k: np.asarray([]) for k in ["myo"]}
            )
            self.eval_msh_score = self.eval_df_score.copy()
            self.best_eval_score = 0
        else:
            self.ckpt_dir = super_params.ckpt_dir
            self.out_dir = super_params.out_dir
            os.makedirs(self.out_dir, exist_ok=True)

        # data augmentation for resizing the segmentation prediction into crop window size
        self.pred_transform = Compose([
            AsDiscrete(argmax=True),
            # RemoveSmallObjects(min_size=8),
            KeepLargestConnectedComponent(independent=True, connectivity=3),
        ])
        # CPU-optimized post_transform pipeline for memory efficiency
        self.post_transform = Compose([
            Spacingd(["pred", "label"], [2.0, 2.0, 2.0], mode=("bilinear", "nearest"), allow_missing_keys=True),
            CropForegroundd(["pred", "label"], source_key="label", allow_missing_keys=True),
            Maskd(["pred", "label", "modal"], allow_missing_keys=True),
            FlexResized(
                ["pred", "label"], 
                (-1, self.super_params.crop_window_size[0], -1), 
                allow_missing_keys=True
                ),
            Resized(
                ["pred", "label"], 
                int(self.super_params.crop_window_size[0] // self.super_params.pixdim[0]), 
                size_mode="longest", mode=("bilinear", "nearest-exact"), 
                allow_missing_keys=True
                ),
            ResizeWithPadOrCropd(
                ["pred", "label"],
                int(self.super_params.crop_window_size[0] // self.super_params.pixdim[0]), 
                mode="constant", value=0,
                allow_missing_keys=True
                ),
            EnsureTyped(["pred", "label"], device="cpu", allow_missing_keys=True),  # Keep on CPU
        ])
        
        # GPU version for when we specifically need GPU output
        self.post_transform_gpu = Compose([
            Spacingd(["pred", "label"], [2.0, 2.0, 2.0], mode=("bilinear", "nearest"), allow_missing_keys=True),
            CropForegroundd(["pred", "label"], source_key="label", allow_missing_keys=True),
            Maskd(["pred", "label", "modal"], allow_missing_keys=True),
            FlexResized(
                ["pred", "label"], 
                (-1, self.super_params.crop_window_size[0], -1), 
                allow_missing_keys=True
                ),
            Resized(
                ["pred", "label"], 
                int(self.super_params.crop_window_size[0] // self.super_params.pixdim[0]), 
                size_mode="longest", mode=("bilinear", "nearest-exact"), 
                allow_missing_keys=True
                ),
            ResizeWithPadOrCropd(
                ["pred", "label"],
                int(self.super_params.crop_window_size[0] // self.super_params.pixdim[0]), 
                mode="constant", value=0,
                allow_missing_keys=True
                ),
            EnsureTyped(["pred", "label"], device=DEVICE, allow_missing_keys=True),
        ])

        # if super_params.use_ckpt is None:
        #     self._data_warper(rotation=True)
        # Do not call _data_warper here, it will be called by main.py per phase

        # import control mesh (NDC space, [-1, 1]) to compute the subdivision matrix
        template_mesh = load(super_params.template_mesh_dir)
        # centroid = template_mesh.bounds.mean(axis=0)
        # extent = template_mesh.bounds.ptp(axis=0)
        # template_mesh.apply_translation(-centroid)
        # template_mesh.apply_scale(2 / extent)
        self._mesh_label(template_mesh)
        self.template_mesh = Meshes(
            verts=[torch.tensor(template_mesh.vertices, dtype=torch.float32)], 
            faces=[torch.tensor(template_mesh.faces, dtype=torch.int64)]
            ).to(DEVICE)

        self._prepare_modules()
        self._prepare_optimiser()

        self.rasterizer = Rasterize([int(i // self.super_params.pixdim[0]) for i in self.super_params.crop_window_size]) # tool for rasterizing mesh

    def _memory_efficient_post_transform(self, seg_pred_list, seg_true_list, modal, to_gpu=True):
        """
        Memory-efficient post-transform processing that handles tensors individually
        and optionally keeps processing on CPU to save GPU memory.
        """
        processed_preds = []
        
        # Process each tensor individually to avoid large batch processing
        for i, (pred, true) in enumerate(zip(seg_pred_list, seg_true_list)):
            # Move to CPU if not already there
            if pred.is_cuda:
                pred = pred.cpu()
            if true.is_cuda:
                true = true.cpu()
            
            # Apply post-transform on CPU
            result = self.post_transform({"pred": pred, "label": true, "modal": modal})
            processed_pred = result["pred"]
            
            # Move to GPU only when needed and one at a time
            if to_gpu:
                processed_pred = processed_pred.to(DEVICE)
            
            processed_preds.append(processed_pred)
            
            # Clear intermediate results to free memory
            del pred, true, result
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        
        # Stack on GPU
        if to_gpu:
            return torch.stack(processed_preds, dim=0)
        else:
            return processed_preds

    def _prepare_slice_for_wandb(self, slice_tensor, is_segmentation, num_classes=None):
        """
        Prepares a 2D tensor slice for logging to Weights & Biases as an image.
        Handles normalization for input images and scaling for segmentation masks.
        """
        slice_np = slice_tensor.cpu().numpy().astype(np.float32)
        
        if is_segmentation:
            if num_classes is None:
                raise ValueError("num_classes must be provided for segmentation masks.")
            scale_factor = 255.0 / (num_classes - 1) if num_classes > 1 else 255.0
            slice_viz = (slice_np * scale_factor).astype(np.uint8)
        else: # Input image
            min_val = slice_np.min()
            max_val = slice_np.max()
            if max_val - min_val > 1e-6:
                slice_norm = (slice_np - min_val) / (max_val - min_val)
            else:
                slice_norm = np.zeros_like(slice_np)
            slice_viz = (slice_norm * 255.0).astype(np.uint8)
            
        return slice_viz

    def _mesh_label(self, mesh):
        COLOR_MAPPING = {
            (1, 0, 0): 0,   # LV-ENDO
            (0, 1, 0): 1,   # RV-ENDO
            (0, 0, 1): 2,   # LV-EPI
            (1, 1, 0): 3,   # RV-EPI
            (1, 0, 1): 4,   # MV & AAV
            (0, 1, 1): 5,   # TV
            (1, 1, 1): 6,   # PV
            (0, 0, 0): 7,   # LEAVE OUT
        }
        vert_label = mesh.visual.vertex_colors[:, :3]
        vert_label = np.where(vert_label <= 85, 0, 1)
        vert_label = np.array([COLOR_MAPPING[tuple(c)] for c in vert_label])
        self.vert_label = torch.tensor(vert_label, dtype=torch.long, device=DEVICE)
        mesh_lv = convex_hull(mesh.vertices[np.any(np.stack([vert_label == i for i in [0, 2]]), axis=0)])  # LV-ENDO and LV-EPI
        mesh_rv = convex_hull(mesh.vertices[np.any(np.stack([vert_label == i for i in [1, 3]]), axis=0)])  # RV-ENDO and RV-EPI
        # Stack centers of both LV and RV
        self.mesh_c = torch.tensor([mesh_lv.center_mass, mesh_rv.center_mass], device=DEVICE)

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
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    def prepare_all_dataloaders(self, data_types=["train"], training_phase="unet", validation_phase="network", include_test=False):
        """
        Unified function to prepare training, validation, and/or test dataloaders.
        
        Args:
            data_types: List of data types to prepare ["train", "valid", "test"]
            training_phase: Phase for training data ("unet", "resnet", "gsn") - affects which training data is loaded
            validation_phase: Phase for validation ("unet", "resnet", "gsn", "network") - affects which validation data is loaded
            include_test: Whether to also prepare test dataloaders (only used if "test" not in data_types)
        """
        # Handle legacy include_test parameter
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

    def _prepare_training_dataloaders(self, training_phase: str):
        """Prepare training dataloaders based on training phase."""
        # Load MR training data only for UNet phase OR if save_on == "cap" (for testing/validation)
        prepare_mr_train = (self.is_training and training_phase == "unet") or self.super_params.save_on == "cap"
        prepare_ct_train = self.is_training or self.super_params.save_on == "sct"
        
        self._prepare_modal_dataloader("mr", "train", prepare_mr_train, training_phase)
        self._prepare_modal_dataloader("ct", "train", prepare_ct_train, training_phase)

    def _prepare_validation_dataloaders(self, validation_phase: str):
        """Prepare validation dataloaders based on validation phase."""
        # For UNet phase, prepare both CT and MR validation data regardless of save_on
        # For other phases, only prepare validation data for the specific modal
        prepare_mr_valid = (validation_phase == "unet") or self.super_params.save_on == "cap"
        prepare_ct_valid = (validation_phase == "unet") or self.super_params.save_on == "sct"
        
        # Use the appropriate transform phase: "validation" for full pipeline, or specific phase for partial validation
        transform_phase = "validation" if validation_phase == "network" else validation_phase
        
        self._prepare_modal_dataloader("mr", "valid", prepare_mr_valid, transform_phase)
        self._prepare_modal_dataloader("ct", "valid", prepare_ct_valid, transform_phase)

    def _prepare_test_dataloaders(self):
        """Prepare test dataloaders."""
        prepare_mr_test = self.super_params.save_on == "cap"
        prepare_ct_test = self.super_params.save_on == "sct"
        
        self._prepare_modal_dataloader("mr", "test", prepare_mr_test, "validation")
        self._prepare_modal_dataloader("ct", "test", prepare_ct_test, "validation")

    def _prepare_modal_dataloader(self, modal: str, data_type: str, should_prepare: bool, transform_phase: str = "validation"):
        """
        Enhanced helper function to prepare a specific modal dataloader for any data type.
        
        Args:
            modal: "mr" or "ct"
            data_type: "train", "valid", or "test"
            should_prepare: Whether to prepare this dataloader
            transform_phase: Phase to use for transforms (training_phase for train data, "validation" for valid/test)
        """
        if not should_prepare:
            # Clear the dataloader if we're not preparing it
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
                    target=self.target, training_phase=transform_phase
                )
            else:  # valid or test
                _, transform = self._prepare_transform(
                    [f"{modal}_image", f"{modal}_label"], modal, 
                    target=self.target, training_phase=transform_phase
                )
            
            data_json = json.load(f)
            data_list = self._remap_abs_path(data_json[data_split], modal, phase_suffix)
            
            # Limit training data for testing (remove in production)
            if data_type == "train":
                data_list = data_list[:5]
            
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
                # Assign to appropriate attribute
                self._assign_dataloader(modal, data_type, dataloader, dataset)
            else:
                # Set to None if empty
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

    # # Legacy wrapper functions for backward compatibility
    # def _data_warper(self, rotation: bool, training_phase: str):
    #     """Legacy wrapper for backward compatibility."""
    #     self.prepare_all_dataloaders(data_types=["train"], training_phase=training_phase, rotation=rotation)

    # def prepare_dataloaders(self, rotation: bool = False, validation_phase: str = "network", include_test: bool = False):
    #     """Legacy wrapper for backward compatibility."""
    #     data_types = ["valid"]
    #     if include_test:
    #         data_types.append("test")
    #     self.prepare_all_dataloaders(data_types=data_types, validation_phase=validation_phase, rotation=rotation)

    # def prepare_validation_specific_dataloaders(self, rotation: bool = False, validation_phase: str = "network"):
    #     """Legacy wrapper for backward compatibility."""
    #     self.prepare_all_dataloaders(data_types=["valid"], validation_phase=validation_phase, rotation=rotation)

    # def prepare_test_specific_dataloaders(self, rotation: bool = False):
    #     """Legacy wrapper for backward compatibility."""
    #     self.prepare_all_dataloaders(data_types=["test"], validation_phase="network", rotation=rotation)

    def _prepare_transform(self, keys, modal, training_phase: str, **kwargs):
        # For training data, use the specified training_phase
        train_transform = pre_transform(
            keys, modal, "train",
            self.super_params.crop_window_size,
            self.super_params.pixdim, phase=training_phase, **kwargs
            )
        # For validation/test data transform preparation, use the specified phase.
        valid_transform = pre_transform(
            keys, modal, "valid", # "valid" section for MONAI transforms like Rand*
            self.super_params.crop_window_size,
            self.super_params.pixdim, phase=training_phase, **kwargs # Use training_phase here by default
            )
        
        return train_transform, valid_transform

    def _remap_abs_path(self, data_list, modal, phase):
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
        
    def _prepare_modules(self):
        # initialise the df-predict module
        # Convert 1D parameter lists to the appropriate dimension based on spatial_dims
        mr_kernel_size = [(k, k) for k in self.super_params.kernel_size]
        mr_strides = [(s, s) for s in self.super_params.strides]
        mr_upsample_kernel_size = [(s, s) for s in self.super_params.strides[1:]]
        
        # Kernel sizes for 3D CT UNet
        ct_kernel_size = [(k, k, k) for k in self.super_params.kernel_size]
        ct_strides = [(s, s, s) for s in self.super_params.strides]
        ct_upsample_kernel_size = [(s, s, s) for s in self.super_params.strides[1:]]
        
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
            spatial_dims=3, in_channels=1, # Changed to 3D
            out_channels=self.super_params.num_classes,
            kernel_size=ct_kernel_size, 
            strides=ct_strides,
            upsample_kernel_size=ct_upsample_kernel_size, 
            filters=self.super_params.filters, 
            dropout=False,
            deep_supervision=False,
            res_block=True
        ).to(DEVICE)
        self.decoder = SegResNet(
            act=("leakyrelu", {"inplace": True, "negative_slope": 0.1}),
            norm=("INSTANCE", {"affine": True}),
            in_channels=self.super_params.num_classes,
            out_channels=self.super_params.num_classes,
            blocks_down=self.super_params.layers,
            blocks_up=tuple([1 for _ in range(len(self.super_params.layers)-1)]),
        ).to(DEVICE)

        # initialise the subdiv module
        self.subdivided_faces = Subdivision(self.template_mesh, self.super_params.subdiv_levels, mesh_label=self.vert_label) # create pre-computed subdivision matrix
        self.GSN = GSN(
            hidden_features=self.super_params.hidden_features_gsn, 
            num_layers=self.super_params.subdiv_levels if self.super_params.subdiv_levels > 0 else 2,
            num_iterations=self.super_params.iteration,
        ).to(DEVICE)
        
        # Initialize local mesh warper for use in warp_template_mesh
        self.local_mesh_warper = LocalMeshWarper(self.super_params.iteration).to(DEVICE)

        # initialise th NDF module
        # self.NDF = NODEBlock(
        #     hidden_size=16, atol=1, rtol=1e-2,
        # ).to(DEVICE)

    def _prepare_optimiser(self):
        # Create separate loss functions for CT and MR training to avoid shared state
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
            
        self.mse_loss_fn = nn.MSELoss()

        self.msk_dice_loss_fn = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
        )
        self.l1_loss_fn = nn.L1Loss()

        # initialise the optimiser for unet
        self.optimzer_mr_unet = torch.optim.Adam(
            self.encoder_mr.parameters(), lr=self.super_params.lr
            )
        self.lr_scheduler_mr_unet = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimzer_mr_unet, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=int(self.super_params.max_epochs//50), min_lr=1e-6, eps=1e-8
            )
        self.optimzer_ct_unet = torch.optim.AdamW(
            self.encoder_ct.parameters(), lr=self.super_params.lr
            )
        self.lr_scheduler_ct_unet = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimzer_ct_unet, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=int(self.super_params.max_epochs//50), min_lr=1e-6, eps=1e-8
            )
        
        # initialise the optimiser for resnet
        self.optimizer_resnet = torch.optim.AdamW(
            self.decoder.parameters(), 
            lr=self.super_params.lr
            )
        self.lr_scheduler_resnet = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_resnet, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=int(self.super_params.max_epochs//50), min_lr=1e-6, eps=1e-8
            )
        
        # initialise the optimiser for gsn
        self.optimizer_gsn = torch.optim.AdamW(
            self.GSN.parameters(), 
            lr=self.super_params.lr
            )
        self.lr_scheduler_gsn = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gsn, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=int(self.super_params.max_epochs//50), min_lr=1e-6, eps=1e-8
            )
        
        # initialise the optimiser for ndf
        # self.optimizer_ndf = torch.optim.Adam(
        #     self.NDF.parameters(),
        #     lr=self.super_params.lr
        # )
        # self.lr_scheduler_ndf = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer_ndf, mode="min", factor=0.1, patience=5, verbose=False,
        #     threshold=1e-2, threshold_mode="rel",
        #     cooldown=int(self.super_params.max_epochs//50), min_lr=1e-6, eps=1e-8
        # )

        # initialise the gradient scaler
        self.scaler_mr_unet = torch.cuda.amp.GradScaler()
        self.scaler_ct_unet = torch.cuda.amp.GradScaler()
        self.scaler_resnet = torch.cuda.amp.GradScaler()
        self.scaler_gsn = torch.cuda.amp.GradScaler()
        # self.scaler_ndf = torch.cuda.amp.GradScaler()
        
        torch.backends.cudnn.enabled = torch.backends.cudnn.is_available()
        torch.backends.cudnn.benchmark = torch.backends.cudnn.is_available()

    def surface_extractor(self, seg_true, labels=None):
        """
            WARNING: this operation is non-differentiable.
            input:
                seg_true: ground truth segmentation.
                labels: integer or list of integers/lists specifying which labels to extract.
                       If None, uses default [[1], [1, 2, 3]] for backwards compatibility.
                       If integer, extracts only that label.
                       If list, each element can be an integer or list of integers to combine.
            return:
                surface mesh with vertices and faces in NDC space [-1, 1].
        """
        # Handle labels parameter
        if labels is None:
            # For GSN phase: extract only LV and MYO surfaces as originally designed
            seg_idx_list = [[2]]   # myocardium
        elif isinstance(labels, int):
            # Single label
            seg_idx_list = [[labels]]
        elif isinstance(labels, list):
            # List of labels or lists of labels
            seg_idx_list = []
            for label_group in labels:
                if isinstance(label_group, int):
                    seg_idx_list.append([label_group])
                elif isinstance(label_group, list):
                    seg_idx_list.append(label_group)
                else:
                    raise ValueError(f"Invalid label type in labels list: {type(label_group)}")
        else:
            raise ValueError(f"Invalid labels type: {type(labels)}. Must be None, int, or list.")
        
        seg_true_multi = [torch.any(torch.stack([seg_true == i for i in seg_idx]), dim=0) for seg_idx in seg_idx_list]

        mesh_true = []
        for seg_true_ in seg_true_multi:
            verts, faces = marching_cubes(
                seg_true_.squeeze(1).permute(0, 3, 1, 2).to(torch.float32), 
                isolevel=0.1,
                return_local_coords=True,
            )
            mesh_true.append(taubin_smoothing(Meshes(verts, faces), 0.77, -0.34, 30))

        return mesh_true

    @torch.no_grad()
    def warp_template_mesh(self, df_preds):
        """
            input:
                df preds: the predicted df.
            return:
                warped control mesh with vertices and faces in NDC space.
        """

        b, *_, d = df_preds.shape

        def find_rotation_matrix_xz(vector_msh, vector_df):
            # Project vectors onto xz-plane
            vector_msh_xz = torch.stack([vector_msh[:, 0], vector_msh[:, 2]], dim=1)
            vector_df_xz = torch.stack([vector_df[:, 0], vector_df[:, 2]], dim=1)

            # Normalize the projected vectors
            vector_msh_xz = vector_msh_xz / torch.norm(vector_msh_xz, dim=1, keepdim=True)
            vector_df_xz = vector_df_xz / torch.norm(vector_df_xz, dim=1, keepdim=True)

            # Calculate the cosine of the angle between the projected vectors
            cos_theta = torch.sum(vector_msh_xz * vector_df_xz, dim=1)

            # Calculate the sine of the angle using the determinant of 2x2 matrix
            sin_theta = vector_msh_xz[:, 0] * vector_df_xz[:, 1] - vector_msh_xz[:, 1] * vector_df_xz[:, 0]

            # Create rotation matrices
            R = torch.zeros(vector_msh.shape[0], 3, 3, device=vector_msh.device, dtype=torch.float64)
            R[:, 0, 0] = cos_theta
            R[:, 0, 2] = sin_theta
            R[:, 1, 1] = 1
            R[:, 2, 0] = -sin_theta
            R[:, 2, 2] = cos_theta

            return R

        template_mesh = load(self.super_params.template_mesh_dir)
        # extent = template_mesh.bounds.ptp(axis=0)
        # template_mesh.apply_scale(2 / extent)
        template_mesh = Meshes(
            verts=[torch.tensor(template_mesh.vertices, dtype=torch.float64)], 
            faces=[torch.tensor(template_mesh.faces, dtype=torch.int64)]
            ).to(DEVICE).extend(b)
        
        # sample and apply offset in two-stage manner
        # stage 1: smooth global offset
        verts = template_mesh.verts_padded()
        # find the rotation matrix that makes the centroid vector are in the same direction
        # Use RV distance field (channel 2) for RV center calculation
        df_c = torch.stack([2 * (torch.nonzero(df <= 1).to(torch.float64).mean(0) / d - 0.5) 
                            for df in df_preds[:, 2]])[:, [1, 0, 2]]   # reorder dimensions, using RV channel
        # Use only the RV center as the reference center
        mesh_c = self.mesh_c[1].unsqueeze(0).expand(b, -1).to(torch.float64)
        R = find_rotation_matrix_xz(mesh_c, df_c)
        # Ensure verts are in double precision before matrix multiplication
        verts = verts.to(torch.float64)
        verts = torch.bmm(R, verts.transpose(1, 2)).transpose(1, 2)

        template_mesh = template_mesh.update_padded(verts)

        # stage 2: local offset using LocalMeshWarper
        template_mesh = self.local_mesh_warper(template_mesh, df_preds, self.vert_label)

        return template_mesh

    def load_pretrained_weight(self, phase):
        # Determine which checkpoint directory to use
        if self.super_params.use_ckpt is None:
            # Using weights from current training session, no need to load checkpoints
            print("Using model weights from current training session")
            return
        else:
            # Load pretrained checkpoints from the specified directory
            ckpt_dir = os.path.join(self.super_params.use_ckpt, "trained_weights")
            print(f"Loading pretrained checkpoints from: {ckpt_dir}")
            
            if phase == "unet" or phase == "all":
                try:
                    encoder_mr_path = glob.glob(f"{ckpt_dir}/best_UNet_MR.pth")
                    encoder_ct_path = glob.glob(f"{ckpt_dir}/best_UNet_CT.pth")
                    
                    if encoder_mr_path and encoder_ct_path:
                        encoder_mr_ckpt = torch.load(encoder_mr_path[0], map_location=DEVICE)
                        encoder_ct_ckpt = torch.load(encoder_ct_path[0], map_location=DEVICE)
                        self.encoder_mr.load_state_dict(encoder_mr_ckpt)
                        self.encoder_ct.load_state_dict(encoder_ct_ckpt)
                        print("UNet weights loaded successfully.")
                except Exception as e:
                    print(f"Note: {e}")
                    print("Proceeding with current model weights.")

            if phase == "resnet" or phase == "all":
                try:
                    decoder_path = glob.glob(f"{ckpt_dir}/best_ResNet.pth")
                    if decoder_path:
                        decoder_ckpt = torch.load(decoder_path[0], map_location=DEVICE)
                        self.decoder.load_state_dict(decoder_ckpt)
                        print("ResNet weights loaded successfully.")
                except Exception as e:
                    print(f"Note: {e}")
                    print("Proceeding with current model weights.")

            # GSN checkpoint loading is intentionally skipped
            # The GSN will always use its current weights

    def _filter_unlabeled_slices(self, img, seg):
        """
        Filter out slices without labels between the first and last labeled slice.
        Maintains corresponding slices in both halves of the data.
        
        Args:
            img: Input image tensor
            seg: Input segmentation tensor
            
        Returns:
            Filtered image and segmentation tensors
        """
        half_size = seg.shape[0]//2
        mask = seg[:half_size] > 0
        has_label = mask.any(dim=1).any(dim=1).any(dim=1)
        
        if has_label.sum() > 0:  # Only process if at least one slice has a label
            start_idx = torch.where(has_label)[0].min()
            end_idx = torch.where(has_label)[0].max()
            
            # Create a full mask: keep slices outside [start_idx, end_idx] and labeled slices within range
            full_mask = torch.ones_like(has_label, device=DEVICE, dtype=torch.bool)
            full_mask[start_idx:end_idx+1] = has_label[start_idx:end_idx+1]  # Only filter unlabeled slices within range
            
            # Get valid indices from first half
            first_half_indices = torch.where(full_mask)[0]
            
            # Create corresponding indices for the second half
            second_half_indices = first_half_indices + half_size
            
            # Combine indices from both halves
            valid_indices = torch.cat([first_half_indices, second_half_indices])
            
            # Apply the mask
            img = img[valid_indices]
            seg = seg[valid_indices]
            
        return img, seg

    def _apply_resnet_padding(self, tensor):
        """
        Apply padding to ensure ResNet compatibility with skip connections.
        
        Args:
            tensor: Input tensor with shape (B, C, H, W, D)
        
        Returns:
            Tuple of (padded_tensor, pad_info) where pad_info contains padding information
        """
        # Calculate stride factor based on ResNet layers
        # SegResNet typically has 2^(number of layers) stride factor
        num_layers = len(self.super_params.layers)
        stride_factor = 2 ** num_layers
        
        original_shape = tensor.shape
        b, c, h, w, d = original_shape
        
        # Calculate padding needed for each spatial dimension
        pad_h = (stride_factor - h % stride_factor) % stride_factor
        pad_w = (stride_factor - w % stride_factor) % stride_factor
        pad_d = (stride_factor - d % stride_factor) % stride_factor
        
        # Apply padding if needed
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            # PyTorch pad format: (D_left, D_right, W_left, W_right, H_left, H_right)
            padding = (0, pad_d, 0, pad_w, 0, pad_h)
            padded_tensor = F.pad(tensor, padding, mode="constant", value=0)
            
            # Store padding info for removal later
            pad_info = {
                'original_shape': original_shape,
                'pad_h': pad_h,
                'pad_w': pad_w,
                'pad_d': pad_d
            }
            
            # Silent padding - no logging
            # Store padding info for debugging if needed
            if not hasattr(self, '_resnet_padding_logged'):
                self._resnet_padding_logged = True
        else:
            padded_tensor = tensor
            pad_info = {'original_shape': original_shape, 'pad_h': 0, 'pad_w': 0, 'pad_d': 0}
        
        return padded_tensor, pad_info

    def _remove_resnet_padding(self, tensor, pad_info):
        """
        Remove padding applied for ResNet compatibility.
        
        Args:
            tensor: Padded tensor
            pad_info: Dictionary containing padding information
        
        Returns:
            Tensor with padding removed
        """
        # Extract padding information
        original_shape = pad_info['original_shape']
        pad_h = pad_info['pad_h']
        pad_w = pad_info['pad_w']
        pad_d = pad_info['pad_d']
        
        # Remove padding by slicing to original dimensions
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            _, _, orig_h, orig_w, orig_d = original_shape
            tensor = tensor[:, :, :orig_h, :orig_w, :orig_d]
        
        return tensor

    def _convert_to_onehot(self, tensor, num_classes, is_prediction=True):
        """
        Streamlined function to convert tensors to one-hot format for DiceMetric.
        
        Args:
            tensor: Input tensor to convert
            num_classes: Number of classes
            is_prediction: If True, applies argmax first (for predictions), 
                          if False, treats as labels (for ground truth)
        
        Returns:
            One-hot encoded tensor with shape (B, C, H, W, D) or (B, C, H, W)
        """
        if is_prediction:
            # For predictions: apply argmax first, then one-hot
            tensor_discrete = torch.argmax(tensor, dim=1)
        else:
            # For ground truth: squeeze channel dimension if present, then convert to long
            if tensor.dim() > 3 and tensor.shape[1] == 1:
                tensor_discrete = tensor.squeeze(1).long()
            else:
                tensor_discrete = tensor.long()
        
        # Convert to one-hot and permute to (B, C, spatial_dims...)
        onehot = torch.nn.functional.one_hot(tensor_discrete, num_classes=num_classes)
        
        # Permute based on tensor dimensions
        if tensor.dim() == 5:  # 3D case: (B, C, H, W, D)
            onehot = onehot.permute(0, 4, 1, 2, 3).float()
        elif tensor.dim() == 4:  # 2D case: (B, C, H, W)
            onehot = onehot.permute(0, 3, 1, 2).float()
        else:
            raise ValueError(f"Unsupported tensor dimensions: {tensor.dim()}")
        
        return onehot

    def train_iter(self, epoch, phase, commit_log=True):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1} - {phase.upper()} TRAINING")
        print(f"{'='*60}")
        if phase == "unet":
            self.encoder_mr.train()
            self.encoder_ct.train()

            train_loss_epoch = dict(total=0.0, ct=0.0, mr=0.0)
            log_data_unet = {} # Initialize dict to collect all unet phase logs for this epoch
            
            # train the CT segmentation encoder
            log_ct_step = np.random.randint(0, len(self.ct_train_loader)) if self.ct_train_loader is not None and len(self.ct_train_loader) > 0 else -1
            if self.ct_train_loader is not None:
                for step, data_ct in enumerate(self.ct_train_loader):
                    img_ct, seg_true_ct = (
                        data_ct["ct_image"].as_tensor().to(DEVICE),
                        data_ct["ct_label"].as_tensor().to(DEVICE),
                        )

                    self.optimzer_ct_unet.zero_grad()
                    with torch.autocast(device_type=DEVICE):
                        seg_pred_ct = sliding_window_inference(
                            img_ct, 
                            roi_size=self.super_params.crop_window_size, # Use full 3D roi_size for CT
                            sw_batch_size=8, 
                            predictor=self.encoder_ct,
                            overlap=0.5, 
                            mode="gaussian",
                            # device=torch.device('cpu'),  # Move output stitching to CPU to save GPU memory
                            # buffer_steps=4,  # Buffer multiple steps before writing to CPU
                            # buffer_dim=-1,   # Buffer along last spatial dimension
                        ) 
                        loss = self.dice_loss_fn_ct(seg_pred_ct.to(DEVICE), seg_true_ct)

                    self.scaler_ct_unet.scale(loss).backward()
                    self.scaler_ct_unet.step(self.optimzer_ct_unet)
                    self.scaler_ct_unet.update()
                    
                    train_loss_epoch["ct"] += loss.item()

                    if step == log_ct_step:
                        # Extract case ID for logging
                        case_id_ct = os.path.basename(self.ct_train_loader.dataset.data[step]["ct_label"]).replace(".nii.gz", '').replace(".seg.nrrd", '')

                        # Log a slice of the ground truth and prediction
                        # For CT, assuming img_ct is (B, C, D, H, W), typically B=1 for logging
                        # Select a slice from the Depth dimension (dim 2)
                        if img_ct.dim() == 5 and img_ct.shape[2] > 0: # Ensure it's 5D and has depth
                            depth_slice_idx_ct = img_ct.shape[2] // 2
                            
                            # Prepare input image slice (B=0, C=0)
                            input_img_ct_slice = img_ct[0, 0, depth_slice_idx_ct, :, :]
                            input_img_ct_viz = self._prepare_slice_for_wandb(input_img_ct_slice, is_segmentation=False)
                            h_in_ct, w_in_ct = input_img_ct_viz.shape[:2]

                            # Prepare ground truth segmentation slice
                            gt_slice_ct = seg_true_ct[0, 0, depth_slice_idx_ct, :, :]
                            gt_slice_ct_viz = self._prepare_slice_for_wandb(gt_slice_ct, is_segmentation=True, num_classes=self.super_params.num_classes)
                            h_gt_ct, w_gt_ct = gt_slice_ct_viz.shape[:2]

                            # Prepare predicted segmentation slice (move to GPU first for argmax)
                            pred_slice_ct = torch.argmax(seg_pred_ct[0, :, depth_slice_idx_ct, :, :].to(DEVICE), dim=0)
                            pred_slice_ct_viz = self._prepare_slice_for_wandb(pred_slice_ct, is_segmentation=True, num_classes=self.super_params.num_classes)
                            h_pred_ct, w_pred_ct = pred_slice_ct_viz.shape[:2]
                            
                            log_data_unet["CT/Input Image"] = wandb.Image(input_img_ct_viz, caption=f"Case ID: {case_id_ct}")
                            log_data_unet["CT/Ground Truth Segmentation"] = wandb.Image(gt_slice_ct_viz, caption=f"Case ID: {case_id_ct}")
                            log_data_unet["CT/Predicted Segmentation"] = wandb.Image(pred_slice_ct_viz, caption=f"Case ID: {case_id_ct}")
                        else:
                            print(f"Warning: CT image tensor for logging has unexpected shape: {img_ct.shape}")

            train_loss_epoch["ct"] = train_loss_epoch["ct"] / (step + 1) if self.ct_train_loader is not None and len(self.ct_train_loader) > 0 else 0.0
            
            if self.ct_train_loader is not None and len(self.ct_train_loader) > 0:
                print(f"CT UNet Training - Loss: {train_loss_epoch['ct']:.4f}, LR: {self.optimzer_ct_unet.param_groups[0]['lr']:.6f}")
                
            self.lr_scheduler_ct_unet.step(train_loss_epoch["ct"])

            # train the CMR segmentation encoder
            log_mr_step = np.random.randint(0, len(self.mr_train_loader)) if self.mr_train_loader is not None and len(self.mr_train_loader) > 0 else -1
            if self.mr_train_loader is not None:
                for step, data_mr in enumerate(self.mr_train_loader):
                    img_mr, seg_true_mr = (
                        data_mr["mr_image"].as_tensor().to(DEVICE),
                        data_mr["mr_label"].as_tensor().to(DEVICE),
                        )

                    # Filter out slices without labels
                    img_mr, seg_true_mr = self._filter_unlabeled_slices(img_mr, seg_true_mr)

                    self.optimzer_mr_unet.zero_grad()
                    with torch.autocast(device_type=DEVICE):
                        seg_pred_mr = sliding_window_inference(
                            img_mr,
                            roi_size=self.super_params.crop_window_size[:2],
                            sw_batch_size=8,
                            predictor=self.encoder_mr,
                            overlap=0.5,
                            mode="gaussian",
                            # device=torch.device('cpu'),  # Move output stitching to CPU to save GPU memory
                            # buffer_steps=4,  # Buffer multiple steps before writing to CPU
                            # buffer_dim=-1,   # Buffer along last spatial dimension
                        )
                        loss = self.dice_loss_fn_mr(seg_pred_mr.to(DEVICE), seg_true_mr)

                    self.scaler_mr_unet.scale(loss).backward()
                    self.scaler_mr_unet.step(self.optimzer_mr_unet)
                    self.scaler_mr_unet.update()
                    
                    train_loss_epoch["mr"] += loss.item()

                    if step == log_mr_step:
                        # Extract case ID for logging
                        case_id_mr = os.path.basename(self.mr_train_loader.dataset.data[step]["mr_label"]).replace(".nii.gz", '').replace(".seg.nrrd", '')
                        case_id_mr = case_id_mr.split('-')[0]
                        
                        # Log a slice of the ground truth and prediction
                        slice_idx_mr = seg_true_mr.shape[0] // 4 # Select a slice from the Slices dimension

                        # Prepare input image slice
                        input_img_mr_slice = img_mr[slice_idx_mr, 0]
                        input_img_mr_viz = self._prepare_slice_for_wandb(input_img_mr_slice, is_segmentation=False)
                        h_in_mr, w_in_mr = input_img_mr_viz.shape[:2]

                        # Prepare ground truth segmentation slice
                        gt_slice_mr = seg_true_mr[slice_idx_mr, 0]
                        gt_slice_mr_viz = self._prepare_slice_for_wandb(gt_slice_mr, is_segmentation=True, num_classes=self.super_params.num_classes)
                        h_gt_mr, w_gt_mr = gt_slice_mr_viz.shape[:2]

                        # Prepare predicted segmentation slice (move to GPU first for argmax)
                        pred_slice_mr = torch.argmax(seg_pred_mr[slice_idx_mr].to(DEVICE), dim=0)
                        pred_slice_mr_viz = self._prepare_slice_for_wandb(pred_slice_mr, is_segmentation=True, num_classes=self.super_params.num_classes)
                        h_pred_mr, w_pred_mr = pred_slice_mr_viz.shape[:2]

                        # Add to log_data_unet for grouped, epoch-indexed viewing
                        log_data_unet["MR/Input Image"] = wandb.Image(input_img_mr_viz, caption=f"Case ID: {case_id_mr}")
                        log_data_unet["MR/Ground Truth Segmentation"] = wandb.Image(gt_slice_mr_viz, caption=f"Case ID: {case_id_mr}")
                        log_data_unet["MR/Predicted Segmentation"] = wandb.Image(pred_slice_mr_viz, caption=f"Case ID: {case_id_mr}")


            train_loss_epoch["mr"] = train_loss_epoch["mr"] / (step + 1) if self.mr_train_loader is not None and len(self.mr_train_loader) > 0 else 0.0
            
            if self.mr_train_loader is not None and len(self.mr_train_loader) > 0:
                print(f"MR UNet Training - Loss: {train_loss_epoch['mr']:.4f}, LR: {self.optimzer_mr_unet.param_groups[0]['lr']:.6f}")
            
            self.lr_scheduler_mr_unet.step(train_loss_epoch["mr"])

            train_loss_epoch["total"] = train_loss_epoch["ct"] + train_loss_epoch["mr"]
            train_loss_epoch["seg"] = train_loss_epoch["total"]

            for k, v in self.unet_loss.items():
                self.unet_loss[k] = np.append(self.unet_loss[k], train_loss_epoch[k])

            # Add losses to the dictionary
            log_data_unet["unet_loss_ct"] = train_loss_epoch["ct"]
            log_data_unet["unet_loss_mr"] = train_loss_epoch["mr"]
            
            print(f"UNet Total Loss: {train_loss_epoch['total']:.4f} (CT: {train_loss_epoch['ct']:.4f}, MR: {train_loss_epoch['mr']:.4f})")
            print(f"{'='*60}")

            # Single log call for the unet phase
            if log_data_unet: # Ensure there's something to log
                wandb.log(log_data_unet, step=epoch + 1, commit=commit_log)
            
        elif phase == "resnet":
            self.encoder_ct.eval()
            self.decoder.train()

            train_loss_epoch = dict(total=0.0, df=0.0)
            if self.ct_train_loader is not None:
                for step, data_ct in enumerate(self.ct_train_loader):
                    img_ct, seg_true_ct, seg_true_ct_ds = (
                        data_ct["ct_image"].to(DEVICE),
                        data_ct["ct_label"].to(DEVICE),
                        data_ct["ct_label_ds"].to(DEVICE),
                    )
                    
                    self.optimizer_resnet.zero_grad()
                    with torch.autocast(device_type=DEVICE):
                        seg_pred_ct = sliding_window_inference(
                            img_ct,
                            roi_size=self.super_params.crop_window_size, # Use full 3D roi_size for CT
                            sw_batch_size=8,
                            predictor=self.encoder_ct,
                            overlap=0.5,
                            mode="gaussian",
                            # device=torch.device('cpu'),  # Move output stitching to CPU to save GPU memory
                            # buffer_steps=4,  # Buffer multiple steps before writing to CPU
                            # buffer_dim=-1,   # Buffer along last spatial dimension
                        )
                        # Use memory-efficient post-transform processing for resnet phase
                        seg_pred_ct_ds = self._memory_efficient_post_transform(seg_pred_ct, seg_true_ct, "ct", to_gpu=True)
                        
                        # Calculate binary mask and compute distance map
                        binary_mask_pred = (torch.argmax(seg_pred_ct_ds, dim=1, keepdim=True) == 0)
                        dist_map_pred = (-distance_transform_edt(binary_mask_pred.squeeze(1)) + distance_transform_edt(~binary_mask_pred.squeeze(1))).unsqueeze(1)
                        mask = torch.sigmoid(dist_map_pred * self.sigmoid_scale_factor + 1).detach()
                        mask = mask * binary_mask_pred
                        mask[mask < self.mask_threshold] = 0
                        
                        # Apply padding for ResNet compatibility
                        seg_pred_ct_ds_padded, pad_info = self._apply_resnet_padding(seg_pred_ct_ds)
                        
                        # Apply ResNet with padding
                        resnet_output_padded = self.decoder(seg_pred_ct_ds_padded)
                        
                        # Remove padding from ResNet output
                        resnet_output = self._remove_resnet_padding(resnet_output_padded, pad_info)
                        
                        seg_pred_ct_ds = seg_pred_ct_ds + mask * resnet_output

                        loss = self.msk_dice_loss_fn(seg_pred_ct_ds, seg_true_ct_ds)

                    self.scaler_resnet.scale(loss).backward()
                    self.scaler_resnet.step(self.optimizer_resnet)
                    self.scaler_resnet.update()
                    
                    train_loss_epoch["total"] += loss.item()
                    train_loss_epoch["df"] += loss.item()

            for k, v in train_loss_epoch.items():
                train_loss_epoch[k] = v / (step + 1) if self.ct_train_loader is not None and len(self.ct_train_loader) > 0 else 0.0
                self.resnet_loss[k] = np.append(self.resnet_loss[k], train_loss_epoch[k])

            print(f"ResNet Training - Loss: {train_loss_epoch['total']:.4f}, LR: {self.optimizer_resnet.param_groups[0]['lr']:.6f}")
            print(f"{'='*60}")

            wandb.log(
                {f"{phase}_loss": train_loss_epoch["total"]},
                step=epoch + 1, commit=commit_log
                )

            self.lr_scheduler_resnet.step(train_loss_epoch["total"])

        elif phase == "gsn":
            self.encoder_ct.eval()
            self.decoder.eval()
            self.GSN.train()

            finetune_loss_epoch = dict(total=0.0, chmf=0.0, smooth=0.0)
            if self.ct_train_loader is not None:
                for step, data_ct in enumerate(self.ct_train_loader):
                    img_ct, seg_true_ct = (
                        data_ct["ct_image"].to(DEVICE),
                        data_ct["ct_label"].to(DEVICE)
                    )
                    
                    seg_true_ct_ = torch.stack([self.post_transform({"label": i, "modal": "ct"})["label"] for i in seg_true_ct], dim=0)
                    # Generate ground truth mesh for myocardium only (label=2)
                    mesh_true_ct = self.surface_extractor(seg_true_ct_.to(DEVICE), labels=2)

                    self.optimizer_gsn.zero_grad()
                    with torch.autocast(device_type=DEVICE):
                        seg_pred_ct = sliding_window_inference(
                            img_ct,
                            roi_size=self.super_params.crop_window_size, # Use full 3D roi_size for CT
                            sw_batch_size=4,  # Reduced from 8 to 4 for GSN phase to save memory
                            predictor=self.encoder_ct,
                            overlap=0.5,
                            mode="gaussian",
                            device=torch.device('cpu'),  # Move output stitching to CPU to save GPU memory
                            buffer_steps=4,  # Buffer multiple steps before writing to CPU
                            buffer_dim=-1,   # Buffer along last spatial dimension
                        )
                        # Use memory-efficient post-transform processing
                        seg_pred_ct_ds = self._memory_efficient_post_transform(seg_pred_ct, seg_true_ct, "ct", to_gpu=True)
                        
                        binary_mask_pred = (torch.argmax(seg_pred_ct_ds, dim=1, keepdim=True) == 0)
                        dist_map_pred = (-distance_transform_edt(binary_mask_pred.squeeze(1)) + distance_transform_edt(~binary_mask_pred.squeeze(1))).unsqueeze(1)
                        mask = torch.sigmoid(dist_map_pred * self.sigmoid_scale_factor + 1).detach()
                        mask = mask * binary_mask_pred
                        mask[mask < self.mask_threshold] = 0

                        # Apply padding for ResNet compatibility
                        seg_pred_ct_ds_padded, pad_info = self._apply_resnet_padding(seg_pred_ct_ds)
                        
                        # Apply ResNet with padding
                        resnet_output_padded = self.decoder(seg_pred_ct_ds_padded)
                        
                        # Remove padding from ResNet output
                        resnet_output = self._remove_resnet_padding(resnet_output_padded, pad_info)
                        
                        seg_pred_ct_ds = seg_pred_ct_ds + mask * resnet_output
                        seg_pred_ct_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ct_ds])
                        
                        foreground = seg_pred_ct_ds > 0  # Include RV in foreground
                        lv = (seg_pred_ct_ds == 1)
                        rv = (seg_pred_ct_ds == 3)
                        myo = (seg_pred_ct_ds == 2)  # Now contains combined LV-MYO + RV-MYO
                        df_pred_ct = torch.stack([
                            distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                            for i in [foreground, lv, rv, myo]], dim=1)
                        
                        template_mesh = self.warp_template_mesh(df_pred_ct.detach())
                        
                        # Convert template mesh to half precision for compatibility with AMP training
                        template_mesh = template_mesh.update_padded(template_mesh.verts_padded().to(torch.float16))
                        
                        level_outs = self.GSN(template_mesh, self.subdivided_faces.faces_levels)

                        loss_chmf, loss_smooth = 0.0, 0.0
                        for l, subdiv_mesh in enumerate(level_outs):
                            verts_label = self.subdivided_faces.labels_levels[l]
                            # Filter to only surface nodes: LV-ENDO (0), RV-ENDO (1), LV-EPI (2), RV-EPI (3)
                            surface_mask = torch.any(torch.stack([verts_label == i for i in [0, 1, 2, 3]]), dim=0)
                            surface_verts = subdiv_mesh.verts_padded()[:, surface_mask]
                            
                            # Calculate chamfer loss between surface vertices and myocardium ground truth
                            loss_chmf += chamfer_distance(
                                surface_verts, 
                                mesh_true_ct[0].verts_padded(),
                                point_reduction="mean", batch_reduction="mean"
                                )[0] 
                            loss_smooth += mesh_laplacian_smoothing(subdiv_mesh, method="cot")
                        
                        loss = self.super_params.lambda_0 * loss_chmf +\
                            self.super_params.lambda_1 * loss_smooth

                    self.scaler_gsn.scale(loss).backward()
                    self.scaler_gsn.step(self.optimizer_gsn)
                    self.scaler_gsn.update()
                    
                    finetune_loss_epoch["total"] += loss.item()
                    finetune_loss_epoch["chmf"] += loss_chmf.item()
                    finetune_loss_epoch["smooth"] += loss_smooth.item()
                    
                    # Aggressive memory cleanup during GSN training
                    del seg_pred_ct, seg_pred_ct_ds, binary_mask_pred, dist_map_pred, mask
                    del seg_pred_ct_ds_padded, resnet_output_padded, resnet_output
                    del foreground, lv, myo, df_pred_ct, template_mesh, level_outs
                    del loss_chmf, loss_smooth, loss
                    torch.cuda.empty_cache()

                for k, v in finetune_loss_epoch.items():
                    finetune_loss_epoch[k] = v / (step + 1) if self.ct_train_loader is not None and len(self.ct_train_loader) > 0 else 0.0
                    self.gsn_loss[k] = np.append(self.gsn_loss[k], finetune_loss_epoch[k])


                print(f"GSN Training - Total Loss: {finetune_loss_epoch['total']:.4f} "
                      f"(Chamfer: {finetune_loss_epoch['chmf']:.4f}, Smooth: {finetune_loss_epoch['smooth']:.4f})")
                print(f"GSN Training - LR: {self.optimizer_gsn.param_groups[0]['lr']:.6f}")
                print(f"{'='*60}")

                wandb.log(
                    {f"{phase}_loss": finetune_loss_epoch["total"]},
                    step=epoch + 1, commit=commit_log
                )

                self.lr_scheduler_gsn.step(finetune_loss_epoch["total"])

    def valid(self, epoch, save_on):
        print(f"\n--- FULL NETWORK VALIDATION ---")
        print(f"Phase: Full Pipeline (UNet + ResNet + GSN), Modal: {save_on.upper()}")
        
        self.decoder.eval()
        self.GSN.eval()
        # if self.super_params._4d:
        #     self.NDF.eval()
        
        # save model
        ckpt_weight_path = os.path.join(self.ckpt_dir, "trained_weights")
        os.makedirs(ckpt_weight_path, exist_ok=True)
        torch.save(self.encoder_ct.state_dict(), os.path.join(ckpt_weight_path, f"{epoch + 1}_UNet_CT.pth"))
        torch.save(self.encoder_mr.state_dict(), os.path.join(ckpt_weight_path, f"{epoch + 1}_UNet_MR.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(ckpt_weight_path, f"{epoch + 1}_ResNet.pth"))
        torch.save(self.GSN.state_dict(), os.path.join(ckpt_weight_path, f"{epoch + 1}_GSN.pth"))
        # if self.super_params._4d:
        #     torch.save(self.NDF.state_dict(), os.path.join(ckpt_weight_path, f"{epoch + 1}_NDF.pth"))
        # save the subdivided_faces.faces_levels as pth file
        for level, faces in enumerate(self.subdivided_faces.faces_levels):
            torch.save(faces, os.path.join(ckpt_weight_path, f"{epoch+1}_subdivided_faces_l{level}.pth"))
        
        # choose the validation loader
        if save_on == "sct":
            modal = "ct"
            encoder = self.encoder_ct
            valid_loader = self.ct_valid_loader
            roi_size = self.super_params.crop_window_size # Always use 3D roi_size for CT
        elif save_on == "cap":
            modal = "mr"
            encoder = self.encoder_mr
            valid_loader = self.mr_valid_loader
            roi_size = self.super_params.crop_window_size[:2]
        else:
            raise ValueError("Invalid dataset name")
        encoder.eval()

        df_metric_batch_decoder = MSEMetric(reduction="mean_batch")
        msh_metric_batch_decoder = DiceMetric(include_background=False, reduction="mean_batch")

        cached_data = dict()
        choice_case = np.random.choice(len(valid_loader), 1)[0]
        with torch.no_grad():
            for step, data in enumerate(valid_loader):
                img, seg_true, seg_true_ds, df_true = (
                    data[f"{modal}_image"].to(DEVICE),
                    data[f"{modal}_label"].to(DEVICE),
                    data[f"{modal}_label_ds"].to(DEVICE),
                    data[f"{modal}_df"].as_tensor().to(DEVICE),
                )
                num_items_for_unflatten = 1 if modal == 'ct' else 2
                
                # Apply unflatten only for MR
                if modal == 'mr':
                    seg_true = seg_true.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
                    seg_true_ds = seg_true_ds.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
                # For CT, seg_true and seg_true_ds are assumed to be (B, C, D, H, W)

                # evaluate the error between predicted df and the true df
                seg_pred = sliding_window_inference(
                    img, 
                    roi_size=roi_size, 
                    sw_batch_size=8, 
                    predictor=encoder,
                    overlap=0.5, 
                    mode="gaussian",
                    device=torch.device('cpu'),  # Move output stitching to CPU to save GPU memory
                    buffer_steps=4,  # Buffer multiple steps before writing to CPU
                    buffer_dim=-1,   # Buffer along last spatial dimension
                )
                # Apply unflatten only for MR
                if modal == 'mr':
                    seg_pred = seg_pred.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
                # For CT, seg_pred is assumed to be (B, NumClasses, D, H, W)
                
                # Use memory-efficient post-transform processing
                seg_pred_ds = self._memory_efficient_post_transform(seg_pred, seg_true, modal, to_gpu=True)
                
                binary_mask_pred = (torch.argmax(seg_pred_ds, dim=1, keepdim=True) == 0)
                dist_map_pred = (-distance_transform_edt(binary_mask_pred.squeeze(1)) + distance_transform_edt(~binary_mask_pred.squeeze(1))).unsqueeze(1)
                mask = torch.sigmoid(dist_map_pred * self.sigmoid_scale_factor + 1).detach()
                mask = mask * binary_mask_pred
                mask[mask < self.mask_threshold] = 0

                seg_pred_ds = seg_pred_ds + mask * self.decoder(seg_pred_ds)
                seg_pred_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ds])
                foreground = seg_pred_ds > 0  # Include RV in foreground
                lv = (seg_pred_ds == 1)
                rv = (seg_pred_ds == 3)
                myo = (seg_pred_ds == 2)  # Now contains combined LV-MYO + RV-MYO
                df_pred = torch.stack([
                    distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                    for i in [foreground, lv, rv, myo]], dim=1)
                
                df_metric_batch_decoder(df_pred, df_true)

                # evaluate the error between subdivided mesh and the true segmentation
                template_mesh = self.warp_template_mesh(df_pred)
                template_mesh = template_mesh.update_padded(template_mesh.verts_padded().to(torch.float16))
                
                subdiv_mesh = self.GSN(template_mesh, self.subdivided_faces.faces_levels, df_pred, self.subdivided_faces.labels_levels)[-1]
                
                # # Apply post-processing to create LV-MYO as difference between LV-EPI and LV-ENDO convex hulls
                # subdiv_mesh_post = self.post_process_subdiv_mesh(subdiv_mesh)
                
                voxeld_mesh = torch.cat([
                    self.rasterizer(
                        pred_mesh.verts_padded(), pred_mesh.faces_padded())
                    for pred_mesh in subdiv_mesh
                    ], dim=0)

                # Create 3-pixel thick dilation mask for seg_true_ds
                dilated_mask = torch.zeros_like(seg_true_ds)
                for batch_idx in range(seg_true_ds.shape[0]):
                    # Convert to numpy for dilation, then back to tensor
                    seg_np = seg_true_ds[batch_idx, 0].cpu().numpy().astype(bool)
                    dilated_np = binary_dilation(seg_np, iterations=2)
                    dilated_mask[batch_idx, 0] = torch.from_numpy(dilated_np.astype(np.float32)).to(seg_true_ds.device)

                # Apply dilated mask to voxeld_mesh
                voxeld_mesh_masked = voxeld_mesh * dilated_mask
                
                seg_true_ds = (seg_true_ds == 2).to(torch.float32)
                msh_metric_batch_decoder(voxeld_mesh_masked, seg_true_ds)

                if step == choice_case:
                    df_true = df_true
                    df_pred = df_pred

                    cached_data = {
                        "df_true": df_true[0].cpu(),
                        "df_pred": df_pred[0].cpu(),
                        "seg_pred_ds": seg_pred_ds[0].cpu(),
                        "seg_true_ds": seg_true_ds[0].cpu(),
                        "subdiv_mesh": subdiv_mesh[0].cpu(),
                        "template_mesh": template_mesh[0].cpu(),
                    }

        # log dice score
        self.eval_df_score["myo"] = np.append(self.eval_df_score["myo"], df_metric_batch_decoder.aggregate().cpu())
        self.eval_msh_score["myo"] = np.append(self.eval_msh_score["myo"], msh_metric_batch_decoder.aggregate().cpu())
        
        # Calculate evaluation score
        eval_score_epoch = msh_metric_batch_decoder.aggregate().mean()
        df_score_epoch = df_metric_batch_decoder.aggregate().mean()
        
        print(f"Mesh Dice Score: {eval_score_epoch:.4f}")
        print(f"Distance Field MSE: {df_score_epoch:.4f}")
        print(f"Current Best Score: {self.best_eval_score:.4f}")
        
        # Initialize dictionary to collect all validation logs for this epoch
        log_data_valid = {}
        
        log_data_valid["eval_score"] = eval_score_epoch
        # log_data_valid["epoch"] = epoch + 1  # Add epoch as a custom x-axis
        
        # Update summary for best values (helps in run comparison)
        if eval_score_epoch > self.best_eval_score:
            # save the best model
            torch.save(self.encoder_ct.state_dict(), os.path.join(ckpt_weight_path, f"best_UNet_CT.pth"))
            torch.save(self.encoder_mr.state_dict(), os.path.join(ckpt_weight_path, f"best_UNet_MR.pth"))
            torch.save(self.decoder.state_dict(), os.path.join(ckpt_weight_path, f"best_ResNet.pth"))
            torch.save(self.GSN.state_dict(), os.path.join(ckpt_weight_path, f"best_GSN.pth"))
            # if self.super_params._4d:
            #     torch.save(self.NDF.state_dict(), os.path.join(ckpt_weight_path, f"best_NDF.pth"))
            
            # save the subdivided_faces.faces_levels as pth file
            for level, faces in enumerate(self.subdivided_faces.faces_levels):
                torch.save(faces, os.path.join(ckpt_weight_path, f"best_subdivided_faces_l{level}.pth"))
            
            self.best_eval_score = eval_score_epoch
            wandb.run.summary["best_eval_score"] = eval_score_epoch
            
            print(f"*** NEW BEST VALIDATION SCORE: {eval_score_epoch:.4f} ***")
            print(f"Saving best model and generating visualizations...")

            # save visualization when the eval score is the best
            # Set up visualization directory
            visualization_dir = f"{self.ckpt_dir}/visualizations"
            os.makedirs(visualization_dir, exist_ok=True)
            
            # Create visualizations - save both HTML and static images
            draw_plotly(
                seg_true=cached_data["seg_true_ds"], 
                # seg_true=cached_data["seg_true"], 
                mesh_pred=cached_data["subdiv_mesh"],
                save_html=True,
                save_dir=visualization_dir,
                filename="seg_true_ds_vs_mesh_pred.html",
                # filename="seg_true_vs_mesh_pred.html",
                export_png_filename=f"seg_true_ds_vs_mesh_pred_epoch{epoch}.png"
                # export_png_filename=f"seg_true_vs_mesh_pred_epoch{epoch}.png"
            )
            log_data_valid["seg_true_ds vs mesh_pred"] = wandb.Image(f"{visualization_dir}/seg_true_ds_vs_mesh_pred_epoch{epoch}.png")
            
            draw_plotly(
                seg_true=cached_data["seg_true_ds"], 
                seg_pred=cached_data["seg_pred_ds"],
                save_html=True,
                save_dir=visualization_dir,
                filename="seg_true_ds_vs_seg_pred_ds.html",
                export_png_filename=f"seg_true_ds_vs_seg_pred_ds_epoch{epoch}.png"
            )
            log_data_valid["seg_true_ds vs seg_pred_ds"] = wandb.Image(f"{visualization_dir}/seg_true_ds_vs_seg_pred_ds_epoch{epoch}.png")
            
            draw_plotly(
                df_pred=cached_data["df_pred"],
                mesh_pred=cached_data["template_mesh"],
                mesh_c=self.mesh_c,
                save_html=True,
                save_dir=visualization_dir,
                filename="template_vs_df_pred.html",
                export_png_filename=f"template_vs_df_pred_epoch{epoch}.png"
            )
            log_data_valid["template vs df_pred"] = wandb.Image(f"{visualization_dir}/template_vs_df_pred_epoch{epoch}.png")
            
            draw_plotly(
                seg_true=cached_data["seg_true_ds"], 
                df_pred=cached_data["df_pred"],
                save_html=True,
                save_dir=visualization_dir,
                filename="seg_true_ds_vs_df_pred.html",
                export_png_filename=f"seg_true_ds_vs_df_pred_epoch{epoch}.png"
            )
            log_data_valid["seg_true_ds vs df_pred"] = wandb.Image(f"{visualization_dir}/seg_true_ds_vs_df_pred_epoch{epoch}.png")
            
            dist_fig = ff.create_distplot(
                [cached_data["df_true"][-1].flatten().cpu().numpy(), 
                cached_data["df_pred"][-1].flatten().cpu().numpy()],
                group_labels=["df_true", "df_pred"],
                colors=["#EF553B", "#3366CC"],
                bin_size=0.1
            )
            dist_fig.write_image(f"{visualization_dir}/df_true_vs_pred.png")
            log_data_valid["df true vs pred"] = wandb.Image(f"{visualization_dir}/df_true_vs_pred.png")
        
        # Single log call for the validation phase
        if log_data_valid:
            wandb.log(log_data_valid, step=epoch + 1, commit=True)
        
        print(f"--- END FULL NETWORK VALIDATION ---\n")
         
    @torch.no_grad()
    def test(self, save_on):
        # load networks
        self.encoder_ct.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_UNet_CT.pth")))
        self.encoder_mr.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_UNet_MR.pth")))
        self.decoder.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_ResNet.pth")))
        self.GSN.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_GSN.pth")))
        # if self.super_params._4d:
        #     self.NDF.load_state_dict(
        #         torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_NDF.pth")))
        # load the subdivided_faces.faces_levels
        self.subdivided_faces.faces_levels = [torch.load(
            glob.glob(f"{self.ckpt_dir}/*_subdivided_faces_l{level}.pth")[-1]
            ) for level in range(self.super_params.subdiv_levels)]
        self.decoder.eval()
        self.GSN.eval()

        if save_on in "sct":
            modal = "ct"
            encoder = self.encoder_ct
            valid_loader = self.ct_test_loader
            roi_size = self.super_params.crop_window_size  # Always use 3D roi_size for CT
        elif save_on == "cap":
            modal = "mr"
            encoder = self.encoder_mr
            valid_loader = self.mr_test_loader
            roi_size = self.super_params.crop_window_size[:2]
        else:
            raise ValueError("Invalid dataset name")
        encoder.eval()

        msh_metric_batch_decoder = DiceMetric(include_background=False, reduction="none")
        # actual_heart_size_in_pixel = []

        total_inference_time = 0.0  # Track total inference time
        choice_case = np.random.choice(len(valid_loader), 1)[0]
        visualization_data = None  # Store visualization data for later
        
        with torch.no_grad():
            for step, data in enumerate(valid_loader):
                id = os.path.basename(valid_loader.dataset.data[step][f"{modal}_label"]).replace(".nii.gz", '').replace(".seg.nrrd", '')
                if save_on == "cap":
                    id = id.split('-')[0]

                img, seg_true, seg_true_ds, df_true = (
                    data[f"{modal}_image"].to(DEVICE),
                    data[f"{modal}_label"].to(DEVICE),
                    data[f"{modal}_label_ds"].as_tensor().to(DEVICE),
                    data[f"{modal}_df"].as_tensor().to(DEVICE),
                )
                num_items_for_unflatten = 1 if modal == 'ct' else 2
                
                # Apply unflatten only for MR
                if modal == 'mr':
                    seg_true = seg_true.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
                    seg_true_ds = seg_true_ds.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
                # For CT, seg_true and seg_true_ds are assumed to be (B, C, D, H, W)

                # Start timing inference
                start_time = time.time()
                
                # Inference pipeline
                seg_pred = sliding_window_inference(
                    img, 
                    roi_size=roi_size,
                    sw_batch_size=8, 
                    predictor=encoder,
                    overlap=0.5, 
                    mode="gaussian",
                    device=torch.device('cpu'),  # Move output stitching to CPU to save GPU memory
                    buffer_steps=4,  # Buffer multiple steps before writing to CPU
                    buffer_dim=-1,   # Buffer along last spatial dimension
                )
                
                # Apply unflatten only for MR
                if modal == 'mr':
                    seg_pred = seg_pred.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
                # For CT, seg_pred is assumed to be (B, NumClasses, D, H, W)
                
                # Use memory-efficient post-transform processing for test function
                seg_pred_ds = self._memory_efficient_post_transform(seg_pred, seg_true, modal, to_gpu=True)
                
                binary_mask_pred = (torch.argmax(seg_pred_ds, dim=1, keepdim=True) == 0)
                dist_map_pred = (-distance_transform_edt(binary_mask_pred.squeeze(1)) + distance_transform_edt(~binary_mask_pred.squeeze(1))).unsqueeze(1)
                mask = torch.sigmoid(dist_map_pred * self.sigmoid_scale_factor + 3).detach()
                mask = mask * binary_mask_pred
                mask[mask < self.mask_threshold] = 0

                seg_pred_ds = seg_pred_ds + mask * self.decoder(seg_pred_ds)
                seg_pred_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ds])
                foreground = seg_pred_ds > 0  # Include RV in foreground
                lv = (seg_pred_ds == 1)
                rv = (seg_pred_ds == 3)
                myo = (seg_pred_ds == 2)  # Now contains combined LV-MYO + RV-MYO
                df_pred = torch.stack([
                    distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                    for i in [foreground, lv, rv, myo]], dim=1)
                
                template_mesh = self.warp_template_mesh(df_pred)
                template_mesh = template_mesh.update_padded(template_mesh.verts_padded().to(torch.float16))
                
                subdiv_mesh = self.GSN(template_mesh, self.subdivided_faces.faces_levels, df_pred, self.subdivided_faces.labels_levels)[-1]
                
                voxeld_mesh = torch.cat([
                    self.rasterizer(
                        pred_mesh.verts_padded(), pred_mesh.faces_padded())
                    for pred_mesh in subdiv_mesh
                    ], dim=0)
                
                # End timing inference
                end_time = time.time()
                total_inference_time += (end_time - start_time)

                # Create 3-pixel thick dilation mask for seg_true_ds
                dilated_mask = torch.zeros_like(seg_true_ds)
                for batch_idx in range(seg_true_ds.shape[0]):
                    # Convert to numpy for dilation, then back to tensor
                    seg_np = seg_true_ds[batch_idx, 0].cpu().numpy().astype(bool)
                    dilated_np = binary_dilation(seg_np, iterations=2)
                    dilated_mask[batch_idx, 0] = torch.from_numpy(dilated_np.astype(np.float32)).to(seg_true_ds.device)

                # Apply dilated mask to voxeld_mesh
                voxeld_mesh_masked = voxeld_mesh * dilated_mask

                # Generate ground truth mesh using surface_extractor
                try:
                    mesh_true_gt = self.surface_extractor(seg_true_ds, labels=2)  # Only extract myocardium (label 2)
                except Exception as e:
                    print(f"ERROR: Failed to generate ground truth mesh for id: {id}: {e}")
                    mesh_true_gt = None

                # Save every subdiv_mesh to the output directory
                try:
                    if self.super_params.save_on == 'cap': # Adjusted from elif to if
                        for i, phase in zip(range(len(subdiv_mesh)), ['ED', 'ES']):
                            current_mesh_to_save = subdiv_mesh[i].cpu()
                            save_obj_path = os.path.join(self.out_dir, f"{id}-{phase}.obj")
                            save_obj(save_obj_path, current_mesh_to_save.verts_packed(), current_mesh_to_save.faces_packed())
                    else:
                        current_mesh_to_save = subdiv_mesh[0].cpu()
                        save_obj_path = os.path.join(self.out_dir, f"{id}.obj")
                        save_obj(save_obj_path, current_mesh_to_save.verts_packed(), current_mesh_to_save.faces_packed())
                except Exception as e:
                    print(f"ERROR: Failed to save subdiv_mesh for id: {id}: {e}")

                # Save ground truth mesh to the output directory
                if mesh_true_gt is not None:
                    try:
                        if self.super_params.save_on == 'cap':
                            for i, phase in zip(range(len(mesh_true_gt)), ['ED', 'ES']):
                                # Now only extracting myocardium, so save only one mesh per phase
                                if i < len(mesh_true_gt):
                                    gt_mesh_to_save = mesh_true_gt[0].cpu()  # Only one mesh (myocardium)
                                    save_obj_path = os.path.join(self.out_dir, f"{id}-{phase}_gt_myo.obj")
                                    save_obj(save_obj_path, gt_mesh_to_save.verts_packed(), gt_mesh_to_save.faces_packed())
                        else:
                            # Now only extracting myocardium, so save only one mesh
                            if len(mesh_true_gt) > 0:
                                gt_mesh_to_save = mesh_true_gt[0].cpu()  # Only one mesh (myocardium)
                                save_obj_path = os.path.join(self.out_dir, f"{id}_gt_myo.obj")
                                save_obj(save_obj_path, gt_mesh_to_save.verts_packed(), gt_mesh_to_save.faces_packed())
                    except Exception as e:
                        print(f"ERROR: Failed to save ground truth mesh for id: {id}: {e}")

                # # Non-timed operations
                # actual_heart_size_in_pixel.append(list(data[f"{modal}_label_ds"].applied_operations[3 if self.super_params.target == "acdc" else 4]["orig_size"]))

                seg_true_ds = (seg_true_ds == 2).to(torch.float32)
                msh_metric_batch_decoder(voxeld_mesh_masked, seg_true_ds)

                # Store visualization data for later processing
                if step == choice_case:
                    # seg_pred = torch.stack([self.pred_transform(i) for i in seg_pred])
                    # seg_true_ds = F.interpolate(seg_true,
                    #                             scale_factor=1 / self.super_params.pixdim[-1], 
                    #                             mode="nearest-exact")[0].cpu()
                    # Store data for visualization after the loop
                    visualization_data = {
                        "id": id,
                        # "seg_true": seg_true[0].cpu(),
                        # "seg_pred": seg_pred[0].cpu(),
                        # "seg_true_ds": seg_true_ds,
                        "seg_true_ds": seg_true_ds[0].cpu(),
                        "seg_pred_ds": seg_pred_ds[0].cpu(),
                        "df_true": df_true[0].cpu(),
                        "df_pred": df_pred[0].cpu(),
                        "template_mesh": template_mesh[0].cpu(),
                        "subdiv_mesh": subdiv_mesh[0].cpu()
                    }
         
        # Calculate average inference time
        average_inference_time = total_inference_time / len(valid_loader)
        print(f"Average inference time: {average_inference_time:.6f} seconds")

        # Process visualization after timing measurements
        if visualization_data:
            try:
                # Create visualization directory under the output directory
                visualization_dir = os.path.join(self.out_dir, "visualizations")
                os.makedirs(visualization_dir, exist_ok=True)
                print(f"Saving visualizations to: {visualization_dir}")
                
                # Test directory write access
                test_file = os.path.join(visualization_dir, "test_write.tmp")
                try:
                    with open(test_file, "w") as f:
                        f.write("Testing write access")
                    os.remove(test_file)
                    print("Directory write access confirmed")
                except Exception as e:
                    print(f"WARNING: Directory write test failed: {e}")
                
                # Create visualizations - only save static images, no HTML
                id = visualization_data["id"]
                
                try:
                    draw_plotly(
                        seg_true=visualization_data["seg_true_ds"], 
                        mesh_pred=visualization_data["subdiv_mesh"],
                        save_html=True,
                        save_dir=visualization_dir,
                        export_static=True,
                        export_png_filename=f"seg_true_ds_vs_mesh_pred_{id}.png"
                    )
                    
                    wandb.log(
                        {
                            "seg_true_ds vs mesh_pred": wandb.Image(f"{visualization_dir}/seg_true_ds_vs_mesh_pred_{id}.png")
                        },
                        commit=False
                    )
                except Exception as e:
                    print(f"ERROR: Failed to generate seg_true vs mesh_pred plot: {e}")
                
                try:
                    draw_plotly(
                        seg_true=visualization_data["seg_true_ds"], 
                        seg_pred=visualization_data["seg_pred_ds"],
                        save_html=True,
                        save_dir=visualization_dir,
                        export_static=True,
                        export_png_filename=f"seg_true_ds_vs_seg_pred_ds_{id}.png"
                    )
                    
                    wandb.log(
                        {
                            "seg_true_ds vs seg_pred_ds": wandb.Image(f"{visualization_dir}/seg_true_ds_vs_seg_pred_ds_{id}.png")
                        },
                        commit=False
                    )
                except Exception as e:
                    print(f"ERROR: Failed to generate seg_true_ds vs seg_pred_ds plot: {e}")
                
                try:
                    draw_plotly(
                        df_pred=visualization_data["df_pred"],
                        mesh_pred=visualization_data["template_mesh"],
                        mesh_c=self.mesh_c,
                        save_html=True,
                        save_dir=visualization_dir,
                        export_static=True,
                        export_png_filename=f"template_vs_df_pred_{id}.png"
                    )
                    
                    wandb.log(
                        {
                            "template vs df_pred": wandb.Image(f"{visualization_dir}/template_vs_df_pred_{id}.png")
                        },
                        commit=False
                    )
                except Exception as e:
                    print(f"ERROR: Failed to generate template vs df_pred plot: {e}")
                
                try:
                    draw_plotly(
                        seg_true=visualization_data["seg_true_ds"], 
                        df_pred=visualization_data["df_pred"],
                        save_html=True,
                        save_dir=visualization_dir,
                        export_static=True,
                        export_png_filename=f"seg_true_ds_vs_df_pred_{id}.png"
                    )
                    
                    wandb.log(
                        {
                            "seg_true_ds vs df_pred": wandb.Image(f"{visualization_dir}/seg_true_ds_vs_df_pred_{id}.png")
                        },
                        commit=False
                    )
                except Exception as e:
                    print(f"ERROR: Failed to generate seg_true_ds vs df_pred plot: {e}")
                
                # For distribution plots, keep using Plotly directly as it works reliably
                try:
                    # Ensure tensors are on CPU before conversion to numpy
                    dist_fig = ff.create_distplot(
                        [visualization_data["df_true"][-1].flatten().cpu().numpy(), 
                        visualization_data["df_pred"][-1].flatten().cpu().numpy()],
                        group_labels=["df_true", "df_pred"],
                        colors=["#EF553B", "#3366CC"],
                        bin_size=0.1
                    )
                    
                    # Save the distribution plot to local filesystem with ID in filename
                    dist_fig.write_image(f"{visualization_dir}/df_true_vs_pred_{id}.png")
                    
                    # Log to wandb
                    wandb.log(
                        {
                            "df true vs pred": wandb.Plotly(dist_fig),
                            "df true vs pred image": wandb.Image(f"{visualization_dir}/df_true_vs_pred_{id}.png")
                        },
                        commit=True
                    )
                except Exception as e:
                    print(f"ERROR: Failed to generate distribution plot: {e}")
            
            except Exception as e:
                print(f"ERROR: Visualization process failed: {e}")
                print("Continuing with evaluation metrics")

        # size_in_pixel = np.median(np.array(actual_heart_size_in_pixel), axis=0)
        wandb.log({
            # "actual_heart_size h (pixel)": size_in_pixel[0],
            # "actual_heart_size w (pixel)": size_in_pixel[1],
            # "actual_heart_size d (pixel)": size_in_pixel[2],
            "test_score": msh_metric_batch_decoder.aggregate().mean(),
            "average_inference_time": average_inference_time
        })
        
        # Update summary for test score (helps in run comparison)
        wandb.run.summary["test_score"] = msh_metric_batch_decoder.aggregate().mean()
        wandb.run.summary["average_inference_time"] = average_inference_time

    @torch.no_grad()
    def ablation_study(self, save_on):
        # load networks
        self.encoder_ct.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_UNet_CT.pth")))
        self.encoder_mr.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_UNet_MR.pth")))
        self.decoder.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_ResNet.pth")))
        # if self.super_params._4d:
        #     self.NDF.load_state_dict(
        #         torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_NDF.pth")))
        self.GSN.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_GSN.pth")))
        # load the subdivided_faces.faces_levels
        self.subdivided_faces.faces_levels = [torch.load(
            glob.glob(f"{self.ckpt_dir}/*_subdivided_faces_l{level}.pth")[-1]
            ) for level in range(self.super_params.subdiv_levels)]
        self.decoder.eval()
        self.GSN.eval()

        if save_on in "sct":
            modal = "ct"
            encoder = self.encoder_ct
            valid_loader = self.ct_test_loader
            roi_size = self.super_params.crop_window_size  # Always use 3D roi_size for CT
        elif save_on == "cap":
            modal = "mr"
            encoder = self.encoder_mr
            valid_loader = self.mr_test_loader
            roi_size = self.super_params.crop_window_size[:2]
        else:
            raise ValueError("Invalid dataset name")
        encoder.eval()

        # if not self.super_params._4d:
        # Create output folders
        folders = [
            f"ResNet_before-{modal}/myo/f0",
            f"ResNet_after-{modal}/myo/f0",
            f"ResNet_gt-{modal}/myo/f0",
            f"ResNet_df_true-{modal}/myo/f0",
            f"ResNet_df_pred-{modal}/myo/f0",
            "adaptive/myo/f0",
            "loop/myo/f0",
            "unwarp_loop/myo/f0",
            "template_mesh/myo/f0",
            "level_0/myo/f0",
            "level_1/myo/f0",
            "level_2/myo/f0",
        ]

        for folder in folders:
            os.makedirs(os.path.join(self.out_dir, folder), exist_ok=True)
        # else:
        #     os.makedirs(os.path.join(self.out_dir, "myo/f0"), exist_ok=True)

        for i, data in enumerate(valid_loader):
            id = os.path.basename(valid_loader.dataset.data[i][f"{modal}_label"]).replace(".nii.gz", '').replace(".seg.nrrd", '')
            if save_on == "cap":
                id = id.split('-')[0]
            elif save_on == "sct":
                id = id.replace("-ED", "").replace("-ES", "")

            img, seg_true, df_true = (
                data[f"{modal}_image"].to(DEVICE),
                data[f"{modal}_label"].to(DEVICE),
                data[f"{modal}_df"].as_tensor().to(DEVICE),
            )
            num_items_for_unflatten = 1 if modal == 'ct' else 2
            
            # Apply unflatten only for MR
            if modal == 'mr':
                seg_true = seg_true.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
                seg_true_ds = seg_true_ds.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
            # For CT, seg_true and seg_true_ds are assumed to be (B, C, D, H, W)

            seg_pred = sliding_window_inference(
                img, 
                roi_size=roi_size, 
                sw_batch_size=8, 
                predictor=encoder,
                overlap=0.5, 
                mode="gaussian",
                device=torch.device('cpu'),  # Move output stitching to CPU to save GPU memory
                buffer_steps=4,  # Buffer multiple steps before writing to CPU
                buffer_dim=-1,   # Buffer along last spatial dimension
            )
            
            # Apply unflatten only for MR
            if modal == 'mr':
                seg_pred = seg_pred.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
            # For CT, seg_pred is assumed to be (B, NumClasses, D, H, W)
            
            # Use memory-efficient post-transform processing for ablation_study function
            seg_data = []
            for i, (pred, true) in enumerate(zip(seg_pred, seg_true)):
                # Process individually to save memory
                if pred.is_cuda:
                    pred = pred.cpu()
                if true.is_cuda:
                    true = true.cpu()
                result = self.post_transform({"pred": pred, "label": true, "modal": modal})
                seg_data.append(result)
                # Clean up intermediate tensors
                del pred, true, result
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            seg_pred = torch.stack([i["pred"] for i in seg_data], dim=0)
            seg_pred_ds = F.interpolate(seg_pred.as_tensor().to(DEVICE), 
                                            scale_factor=1 / self.super_params.pixdim[-1], 
                                            mode="trilinear")
            
            binary_mask_pred = (torch.argmax(seg_pred_ds, dim=1, keepdim=True) == 0)
            dist_map_pred = (-distance_transform_edt(binary_mask_pred.squeeze(1)) + distance_transform_edt(~binary_mask_pred.squeeze(1))).unsqueeze(1)
            mask = torch.sigmoid(dist_map_pred * self.sigmoid_scale_factor).detach()
            mask = mask * binary_mask_pred
            mask[mask < self.mask_threshold] = 0

            seg_pred_ds_before = seg_pred_ds.clone()
            seg_pred_ds_before = torch.stack([self.pred_transform(i) for i in seg_pred_ds_before])
            seg_pred_ds = seg_pred_ds + mask * self.decoder(seg_pred_ds)
            seg_pred_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ds])
            seg_true = torch.stack([i["label"] for i in seg_data], dim=0)
            seg_true_ds = F.interpolate(seg_true.to(DEVICE),
                                        scale_factor=1 / self.super_params.pixdim[-1], 
                                        mode="nearest-exact")

            # ****** Distance Field Prediction ******
            foreground = seg_pred_ds > 0  # Include RV in foreground
            lv = (seg_pred_ds == 1)
            rv = (seg_pred_ds == 3)  # RV for LV+RV template mesh
            myo = (seg_pred_ds == 2)  # Now contains combined LV-MYO + RV-MYO
            df_pred = torch.stack([
                distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                for i in [foreground, lv, rv, myo]], dim=1)  # 4 channels: foreground, lv, rv, myo

            # if not self.super_params._4d:
            # Save the seg_pred_ds (before and after self.decoder) and seg_true_ds as nib files
            for (phase, idx), _ in zip([('ED', 0), ('ES', 1)], range(seg_true_ds.shape[0])):
                # Before ResNet
                nib.save(nib.nifti1.Nifti1Image(seg_pred_ds_before[idx, 0].cpu().numpy(), np.eye(4)), 
                            f"{self.out_dir}/ResNet_before-{modal}/myo/f0/{id}-{phase}_pred.nii.gz")
                # After ResNet
                nib.save(nib.nifti1.Nifti1Image(seg_pred_ds[idx, 0].cpu().numpy(), np.eye(4)), 
                            f"{self.out_dir}/ResNet_after-{modal}/myo/f0/{id}-{phase}_pred.nii.gz")
                # Ground Truth
                nib.save(nib.nifti1.Nifti1Image(seg_true_ds[idx, 0].cpu().numpy(), np.eye(4)), 
                            f"{self.out_dir}/ResNet_gt-{modal}/myo/f0/{id}-{phase}_true.nii.gz")

            # save the prediction and true distance field as npy files
            np.save(f"{self.out_dir}/ResNet_df_true-{modal}/myo/f0/{id}-df_true.npy", 
                    df_true[0].cpu().numpy())
            np.save(f"{self.out_dir}/ResNet_df_pred-{modal}/myo/f0/{id}-df_pred.npy", 
                    df_pred[0].cpu().numpy())

            if save_on == "sct":
                # warped + adaptive
                template_mesh = self.warp_template_mesh(df_pred)  
                subdiv_mesh_adaptive = self.GSN(template_mesh, self.subdivided_faces.faces_levels, df_pred, self.subdivided_faces.labels_levels)[-1]
                save_obj(
                f"{self.out_dir}/adaptive/myo/f0/{id}.obj", 
                    subdiv_mesh_adaptive.verts_packed(), subdiv_mesh_adaptive.faces_packed()
                )

                # warped + Loop subdivided
                template_mesh = self.warp_template_mesh(df_pred)
                template_mesh = Trimesh(template_mesh.verts_packed().cpu().numpy(), template_mesh.faces_packed().cpu().numpy())
                for _ in range(2): template_mesh = template_mesh.subdivide_loop()
                save_obj(
                f"{self.out_dir}/loop/myo/f0/{id}.obj", 
                    torch.tensor(template_mesh.vertices), torch.tensor(template_mesh.faces)
                )

                # unwarp + Loop subdivided
                template_mesh = self.template_mesh.to(DEVICE)
                template_mesh = Trimesh(template_mesh.verts_packed().cpu().numpy(), template_mesh.faces_packed().cpu().numpy())
                for _ in range(2): template_mesh = template_mesh.subdivide_loop()
                save_obj(
                f"{self.out_dir}/unwarp_loop/myo/f0/{id}.obj", 
                    torch.tensor(template_mesh.vertices), torch.tensor(template_mesh.faces)
                )

                # unwarp (template mesh)
                save_obj(
                    f"{self.out_dir}/template_mesh/myo/f0/{id}.obj", 
                    self.template_mesh.verts_packed(), self.template_mesh.faces_packed()
                )

            # ****** Increamental Subdivision from 0 --> 2 ******
            # template_mesh = self.warp_template_mesh(df_pred)                             # level 0
            template_mesh = self.warp_template_mesh(df_true)                             # level 0
            
            # Convert template mesh to half precision for compatibility with AMP training
            template_mesh = template_mesh.update_padded(template_mesh.verts_padded().to(torch.float16))

            # if not self.super_params._4d and save_on == "sct":
            if save_on == "sct":
                save_obj(
                    f"{self.out_dir}/level_0/myo/f0/{id}.obj", 
                    template_mesh.verts_packed(), template_mesh.faces_packed()
                )

            subdiv_mesh = self.GSN(template_mesh, self.subdivided_faces.faces_levels, df_pred, self.subdivided_faces.labels_levels)   # level 1 & 2: [Meshes, Meshes]

            # if not self.super_params._4d and save_on == "sct":
            if save_on == "sct":
                for level in range(2):
                    save_obj(
                        f"{self.out_dir}/level_{level+1}/myo/f0/{id}.obj", 
                        subdiv_mesh[level].verts_packed(), subdiv_mesh[level].faces_packed()
                    )

            # ****** Compare outputs w/o NODE ******
            # elif self.super_params._4d and save_on == "cap":
            if save_on == "cap":
                subdiv_mesh = subdiv_mesh[-1]
                for i in range(subdiv_mesh._N):
                    # save each mesh as a time instance
                    save_obj(f"{self.out_dir}/myo/f0/{id}-{i:02d}.obj", 
                            subdiv_mesh[i].verts_packed(), subdiv_mesh[i].faces_packed())
            
    @torch.no_grad()
    def validate_segmentation(self, epoch, save_on, commit=True):
        """
        Dedicated validation method for computing dice scores between segmentation predictions and ground truth.
        Phase-aware validation:
        - UNet phase: Validate both encoder_ct AND encoder_mr directly (no ResNet refinement)
        - ResNet phase: Validate only the specific modal encoder WITH ResNet refinement
        """
        print(f"\n--- SEGMENTATION VALIDATION ---")
        
        # Determine current training phase
        is_unet_phase = epoch < self.super_params.pretrain_epochs
        phase_name = "UNet" if is_unet_phase else ("ResNet" if epoch < self.super_params.train_epochs else "GSN")
        print(f"Phase: {phase_name}, Modal: {save_on.upper()}")
        
        log_data_validation = {}
        
        if is_unet_phase:
            # UNet phase: Validate both CT and MR encoders directly
            if self.ct_valid_loader is not None:
                self.encoder_ct.eval()
                ct_dice_scores = {"lv": [], "rv": [], "myo": []}
                
                for step, data_ct in enumerate(self.ct_valid_loader):
                    img_ct, seg_true_ct = (
                        data_ct["ct_image"].to(DEVICE),
                        data_ct["ct_label"].to(DEVICE),
                    )
                    
                    seg_pred_ct = sliding_window_inference(
                        img_ct,
                        roi_size=self.super_params.crop_window_size,
                        sw_batch_size=4,
                        predictor=self.encoder_ct,
                        overlap=0.5,
                        mode="gaussian",
                    )
                    
                    # Convert to one-hot for DiceMetric
                    seg_pred_onehot = self._convert_to_onehot(seg_pred_ct, self.super_params.num_classes, is_prediction=True)
                    seg_true_onehot = self._convert_to_onehot(seg_true_ct, self.super_params.num_classes, is_prediction=False)
                    
                    # Compute dice scores for each class
                    dice_metric = DiceMetric(include_background=False, reduction="mean")
                    dice_scores = dice_metric(seg_pred_onehot, seg_true_onehot)
                    
                    if dice_scores.shape[1] >= 3:  # Ensure we have at least 3 classes
                        ct_dice_scores["lv"].append(dice_scores[0, 0].item())  # Class 1: LV
                        if dice_scores.shape[1] > 2:
                            ct_dice_scores["myo"].append(dice_scores[0, 1].item())  # Class 2: MYO
                        if dice_scores.shape[1] > 3:
                            ct_dice_scores["rv"].append(dice_scores[0, 2].item())  # Class 3: RV
                
                # Log CT validation metrics
                if ct_dice_scores["lv"]:
                    ct_avg_dice = {k: np.mean(v) for k, v in ct_dice_scores.items() if v}
                    log_data_validation.update({
                        f"val_ct_dice_{k}": v for k, v in ct_avg_dice.items()
                    })
                    print(f"CT Dice Scores - LV: {ct_avg_dice.get('lv', 0):.4f}, "
                          f"MYO: {ct_avg_dice.get('myo', 0):.4f}, RV: {ct_avg_dice.get('rv', 0):.4f}")
            
            if self.mr_valid_loader is not None:
                self.encoder_mr.eval()
                mr_dice_scores = {"lv": [], "rv": [], "myo": []}
                
                for step, data_mr in enumerate(self.mr_valid_loader):
                    img_mr, seg_true_mr = (
                        data_mr["mr_image"].to(DEVICE),
                        data_mr["mr_label"].to(DEVICE),
                    )
                    
                    # Filter out slices without labels for MR
                    img_mr, seg_true_mr = self._filter_unlabeled_slices(img_mr, seg_true_mr)
                    
                    seg_pred_mr = sliding_window_inference(
                        img_mr,
                        roi_size=self.super_params.crop_window_size[:2],
                        sw_batch_size=4,
                        predictor=self.encoder_mr,
                        overlap=0.5,
                        mode="gaussian",
                    )
                    
                    # Convert to one-hot for DiceMetric
                    seg_pred_onehot = self._convert_to_onehot(seg_pred_mr, self.super_params.num_classes, is_prediction=True)
                    seg_true_onehot = self._convert_to_onehot(seg_true_mr, self.super_params.num_classes, is_prediction=False)
                    
                    # Compute dice scores for each class
                    dice_metric = DiceMetric(include_background=False, reduction="mean")
                    dice_scores = dice_metric(seg_pred_onehot, seg_true_onehot)
                    
                    if dice_scores.shape[1] >= 3:  # Ensure we have at least 3 classes
                        mr_dice_scores["lv"].append(dice_scores[0, 1].item())  # Class 1: LV
                        if dice_scores.shape[1] > 2:
                            mr_dice_scores["myo"].append(dice_scores[0, 2].item())  # Class 2: MYO
                        if dice_scores.shape[1] > 3:
                            mr_dice_scores["rv"].append(dice_scores[0, 3].item())  # Class 3: RV
                
                # Log MR validation metrics
                if mr_dice_scores["lv"]:
                    mr_avg_dice = {k: np.mean(v) for k, v in mr_dice_scores.items() if v}
                    log_data_validation.update({
                        f"val_mr_dice_{k}": v for k, v in mr_avg_dice.items()
                    })
                    print(f"MR Dice Scores - LV: {mr_avg_dice.get('lv', 0):.4f}, "
                          f"MYO: {mr_avg_dice.get('myo', 0):.4f}, RV: {mr_avg_dice.get('rv', 0):.4f}")
        
        else:
            # ResNet phase: Validate only the specific modal with ResNet refinement
            if save_on == "sct" and self.ct_valid_loader is not None:
                self.encoder_ct.eval()
                self.decoder.eval()
                ct_dice_scores = {"lv": [], "rv": [], "myo": []}
                
                for step, data_ct in enumerate(self.ct_valid_loader):
                    img_ct, seg_true_ct, seg_true_ct_ds = (
                        data_ct["ct_image"].to(DEVICE),
                        data_ct["ct_label"].to(DEVICE),
                        data_ct["ct_label_ds"].to(DEVICE),
                    )
                    
                    # UNet prediction
                    seg_pred_ct = sliding_window_inference(
                        img_ct,
                        roi_size=self.super_params.crop_window_size,
                        sw_batch_size=4,
                        predictor=self.encoder_ct,
                        overlap=0.5,
                        mode="gaussian",
                    )
                    
                    # Apply post-transform to get downsampled version
                    seg_pred_ct_ds = self._memory_efficient_post_transform(seg_pred_ct, seg_true_ct, "ct", to_gpu=True)
                    
                    # Apply ResNet refinement
                    binary_mask_pred = (torch.argmax(seg_pred_ct_ds, dim=1, keepdim=True) == 0)
                    dist_map_pred = (-distance_transform_edt(binary_mask_pred.squeeze(1)) + 
                                    distance_transform_edt(~binary_mask_pred.squeeze(1))).unsqueeze(1)
                    mask = torch.sigmoid(dist_map_pred * self.sigmoid_scale_factor + 1).detach()
                    mask = mask * binary_mask_pred
                    mask[mask < self.mask_threshold] = 0
                    
                    # Apply padding for ResNet compatibility
                    seg_pred_ct_ds_padded, pad_info = self._apply_resnet_padding(seg_pred_ct_ds)
                    resnet_output_padded = self.decoder(seg_pred_ct_ds_padded)
                    resnet_output = self._remove_resnet_padding(resnet_output_padded, pad_info)
                    
                    # Final refined prediction
                    seg_pred_ct_refined = seg_pred_ct_ds + mask * resnet_output
                    
                    # Use downsampled ground truth for comparison
                    seg_true_ct_ds = seg_true_ct_ds.to(DEVICE)
                    
                    # Convert to one-hot for DiceMetric
                    seg_pred_onehot = self._convert_to_onehot(seg_pred_ct_refined, self.super_params.num_classes, is_prediction=True)
                    seg_true_onehot = self._convert_to_onehot(seg_true_ct_ds, self.super_params.num_classes, is_prediction=False)
                    
                    # Compute dice scores for each class
                    dice_metric = DiceMetric(include_background=False, reduction="mean")
                    dice_scores = dice_metric(seg_pred_onehot, seg_true_onehot)
                    
                    if dice_scores.shape[1] >= 3:  # Ensure we have at least 3 classes
                        ct_dice_scores["lv"].append(dice_scores[0, 1].item())  # Class 1: LV
                        if dice_scores.shape[1] > 2:
                            ct_dice_scores["myo"].append(dice_scores[0, 2].item())  # Class 2: MYO
                        if dice_scores.shape[1] > 3:
                            ct_dice_scores["rv"].append(dice_scores[0, 3].item())  # Class 3: RV
                
                # Log CT validation metrics
                if ct_dice_scores["lv"]:
                    ct_avg_dice = {k: np.mean(v) for k, v in ct_dice_scores.items() if v}
                    log_data_validation.update({
                        f"val_ct_dice_{k}": v for k, v in ct_avg_dice.items()
                    })
                    print(f"CT Dice Scores (ResNet refined) - LV: {ct_avg_dice.get('lv', 0):.4f}, "
                          f"MYO: {ct_avg_dice.get('myo', 0):.4f}, RV: {ct_avg_dice.get('rv', 0):.4f}")
            
            elif save_on == "cap" and self.mr_valid_loader is not None:
                # For MR, ResNet phase validation would be similar but we don't have ResNet for MR
                # So we just validate the MR encoder directly
                self.encoder_mr.eval()
                mr_dice_scores = {"lv": [], "rv": [], "myo": []}
                
                for step, data_mr in enumerate(self.mr_valid_loader):
                    img_mr, seg_true_mr = (
                        data_mr["mr_image"].to(DEVICE),
                        data_mr["mr_label"].to(DEVICE),
                    )
                    
                    # Filter out slices without labels for MR
                    img_mr, seg_true_mr = self._filter_unlabeled_slices(img_mr, seg_true_mr)
                    
                    seg_pred_mr = sliding_window_inference(
                        img_mr,
                        roi_size=self.super_params.crop_window_size[:2],
                        sw_batch_size=4,
                        predictor=self.encoder_mr,
                        overlap=0.5,
                        mode="gaussian",
                    )
                    
                    # Convert to one-hot for DiceMetric
                    seg_pred_onehot = self._convert_to_onehot(seg_pred_mr, self.super_params.num_classes, is_prediction=True)
                    seg_true_onehot = self._convert_to_onehot(seg_true_mr, self.super_params.num_classes, is_prediction=False)
                    
                    # Compute dice scores for each class
                    dice_metric = DiceMetric(include_background=False, reduction="mean")
                    dice_scores = dice_metric(seg_pred_onehot, seg_true_onehot)
                    
                    if dice_scores.shape[1] >= 3:  # Ensure we have at least 3 classes
                        mr_dice_scores["lv"].append(dice_scores[0, 1].item())  # Class 1: LV
                        if dice_scores.shape[1] > 2:
                            mr_dice_scores["myo"].append(dice_scores[0, 2].item())  # Class 2: MYO
                        if dice_scores.shape[1] > 3:
                            mr_dice_scores["rv"].append(dice_scores[0, 3].item())  # Class 3: RV
                
                # Log MR validation metrics
                if mr_dice_scores["lv"]:
                    mr_avg_dice = {k: np.mean(v) for k, v in mr_dice_scores.items() if v}
                    log_data_validation.update({
                        f"val_mr_dice_{k}": v for k, v in mr_avg_dice.items()
                    })
                    print(f"MR Dice Scores - LV: {mr_avg_dice.get('lv', 0):.4f}, "
                          f"MYO: {mr_avg_dice.get('myo', 0):.4f}, RV: {mr_avg_dice.get('rv', 0):.4f}")
        
        # Log all validation metrics
        if log_data_validation:
            wandb.log(log_data_validation, step=epoch + 1, commit=commit)
        
        print(f"--- END SEGMENTATION VALIDATION ---\n")
            