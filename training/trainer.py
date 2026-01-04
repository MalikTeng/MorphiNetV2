import os
import torch
import torch.nn.functional as F
import numpy as np
import wandb
from scipy.ndimage import binary_dilation
from collections import OrderedDict
from monai.inferers import sliding_window_inference
from monai.transforms.utils import distance_transform_edt
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
from monai.transforms import (
    Compose, 
    AsDiscrete, 
    KeepLargestConnectedComponent,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MorphiNetTrainer:
    """Handles training for different phases of MorphiNet: UNet, ResNet, and GSN."""
    
    def __init__(self, super_params, models, optimizers, schedulers, scalers, loss_functions, 
                 dataloaders, preprocessor, mesh_ops, inference, orchestrator=None, target=None, dataset=None):
        """
        Initialize the trainer.
        
        Args:
            super_params: Configuration parameters
            models: Dictionary containing model instances
            optimizers: Dictionary containing optimizers
            schedulers: Dictionary containing learning rate schedulers
            scalers: Dictionary containing gradient scalers
            loss_functions: Dictionary containing loss functions
            dataloaders: Dictionary containing data loaders
            preprocessor: Data preprocessor instance
            mesh_ops: Mesh operations instance
            inference: Model inference instance
            orchestrator: Reference to orchestrator for step management
            target: Training target (deprecated, use dataset)
            dataset: Dataset name
        """
        self.super_params = super_params
        # Handle backward compatibility
        self.dataset = dataset if dataset is not None else target
        self.orchestrator = orchestrator
        
        # Models
        self.encoder_mr = models['encoder_mr']
        self.encoder_ct = models['encoder_ct']
        self.decoder = models['decoder']
        self.GSN = models['GSN']
        
        # Optimizers
        self.optimzer_ct_unet = optimizers['ct_unet']
        self.optimzer_mr_unet = optimizers['mr_unet']
        self.optimizer_resnet = optimizers['resnet']
        self.optimizer_gsn = optimizers['gsn']
        
        # Schedulers
        self.lr_scheduler_ct_unet = schedulers['ct_unet']
        self.lr_scheduler_mr_unet = schedulers['mr_unet']
        self.lr_scheduler_resnet = schedulers['resnet']
        self.lr_scheduler_gsn = schedulers['gsn']
        
        # Scalers
        self.scaler_ct_unet = scalers['ct_unet']
        self.scaler_mr_unet = scalers['mr_unet']
        self.scaler_resnet = scalers['resnet']
        self.scaler_gsn = scalers['gsn']
        
        # Loss functions
        self.dice_loss_fn_ct = loss_functions['dice_ct']
        self.dice_loss_fn_mr = loss_functions['dice_mr']
        self.msk_dice_loss_fn = loss_functions['masked_dice']
        
        # Data loaders - store reference to dataloader manager for dynamic access
        self.dataloader_manager = dataloaders
        
        # Helper modules
        self.preprocessor = preprocessor
        self.mesh_ops = mesh_ops
        self.inference = inference
        
        # Loss tracking
        self.unet_loss = OrderedDict({k: np.asarray([]) for k in ["total", "ct", "mr", "seg"]})
        self.resnet_loss = OrderedDict({k: np.asarray([]) for k in ["total", "df"]})
        self.gsn_loss = OrderedDict({k: np.asarray([]) for k in ["total", "chmf", "smooth"]})
        
        # Prediction transform
        self.pred_transform = Compose([
            AsDiscrete(argmax=True),
            KeepLargestConnectedComponent(is_onehot=True, independent=False, connectivity=3),
        ])
    
    @property
    def ct_train_loader(self):
        """Dynamic access to CT training loader."""
        return getattr(self.dataloader_manager, 'ct_train_loader', None)
    
    @property  
    def mr_train_loader(self):
        """Dynamic access to MR training loader."""
        return getattr(self.dataloader_manager, 'mr_train_loader', None)
    
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
    
    def train_iter(self, epoch, phase, commit_log=True):
        """
        Main training iteration for different phases.
        
        Args:
            epoch: Current epoch number
            phase: Training phase ('unet', 'resnet', 'gsn')
            commit_log: Whether to commit logs to wandb
        """
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1} - {phase.upper()} TRAINING")
        print(f"{'='*60}")
        
        if phase == "unet":
            self._train_unet_phase(epoch, commit_log)
        elif phase == "resnet":
            self._train_resnet_phase(epoch, commit_log)
        elif phase == "gsn":
            self._train_gsn_phase(epoch, commit_log)
        else:
            raise ValueError(f"Unknown training phase: {phase}")
    
    def _train_unet_phase(self, epoch, commit_log=True):
        """Train UNet models for both CT and MR."""
        self.encoder_mr.train()
        self.encoder_ct.train()

        train_loss_epoch = dict(total=0.0, ct=0.0, mr=0.0)
        log_data_unet = {}
        
        # Train CT segmentation encoder
        log_ct_step = np.random.randint(0, len(self.ct_train_loader)) if self.ct_train_loader is not None and len(self.ct_train_loader) > 0 else -1
        ct_step_count = 0
        if self.ct_train_loader is not None:
            for step, data_ct in enumerate(self.ct_train_loader):
                ct_step_count = step + 1
                img_ct, seg_true_ct = (
                    data_ct["ct_image"].as_tensor().to(DEVICE),
                    data_ct["ct_label"].as_tensor().to(DEVICE),
                )

                self.optimzer_ct_unet.zero_grad()
                with torch.autocast(device_type=DEVICE, enabled=False):
                    # Direct forward pass during training (no sliding window)
                    seg_pred_ct = self.encoder_ct(img_ct)
                    loss = self.dice_loss_fn_ct(seg_pred_ct.to(DEVICE), seg_true_ct)

                self.scaler_ct_unet.scale(loss).backward()
                self.scaler_ct_unet.step(self.optimzer_ct_unet)
                self.scaler_ct_unet.update()
                
                loss_value = loss.item()
                train_loss_epoch["ct"] += loss_value
                train_loss_epoch["total"] += loss_value
                
                # Log wandb data for single random step per epoch
                if step == log_ct_step:
                    # Prepare visualization data for CT (shape: [N,C,H,W,D])
                    input_img_ct_slice = img_ct[0, 0, :, :, img_ct.shape[4] // 2] 
                    gt_slice_ct = seg_true_ct[0, 0, :, :, seg_true_ct.shape[4] // 2] 
                    pred_slice_ct = torch.argmax(seg_pred_ct[0, :, :, :, seg_pred_ct.shape[4] // 2], dim=0)
                    
                    # Convert to numpy arrays for wandb logging using _prepare_slice_for_wandb
                    input_img_ct_viz = self._prepare_slice_for_wandb(input_img_ct_slice, is_segmentation=False)
                    gt_slice_ct_viz = self._prepare_slice_for_wandb(gt_slice_ct, is_segmentation=True, num_classes=self.super_params.num_classes)
                    pred_slice_ct_viz = self._prepare_slice_for_wandb(pred_slice_ct, is_segmentation=True, num_classes=self.super_params.num_classes)
                    
                    log_data_unet.update({
                        "unet/ct_input_slice": wandb.Image(input_img_ct_viz, caption=f"CT Input - Epoch {epoch+1}"),
                        "unet/ct_gt_slice": wandb.Image(gt_slice_ct_viz, caption=f"CT Ground Truth - Epoch {epoch+1}"),
                        "unet/ct_pred_slice": wandb.Image(pred_slice_ct_viz, caption=f"CT Prediction - Epoch {epoch+1}"),
                    })

        train_loss_epoch["ct"] = train_loss_epoch["ct"] / ct_step_count if self.ct_train_loader is not None and ct_step_count > 0 else 0.0
        
        if self.ct_train_loader is not None and len(self.ct_train_loader) > 0:
            print(f"CT UNet Training - Loss: {train_loss_epoch['ct']:.4f}, LR: {self.optimzer_ct_unet.param_groups[0]['lr']:.6f}")
            
        self.lr_scheduler_ct_unet.step(train_loss_epoch["ct"])

        # Train MR segmentation encoder
        log_mr_step = np.random.randint(0, len(self.mr_train_loader)) if self.mr_train_loader is not None and len(self.mr_train_loader) > 0 else -1
        mr_step_count = 0
        if self.mr_train_loader is not None:
            for step, data_mr in enumerate(self.mr_train_loader):
                mr_step_count = step + 1
                img_mr, seg_true_mr = (
                    data_mr["mr_image"].as_tensor().to(DEVICE),
                    data_mr["mr_label"].as_tensor().to(DEVICE),
                )

                # Filter out slices without labels
                img_mr, seg_true_mr = self.preprocessor._filter_unlabeled_slices(img_mr, seg_true_mr)

                self.optimzer_mr_unet.zero_grad()
                with torch.autocast(device_type=DEVICE, enabled=False):
                    # Direct forward pass during training (no sliding window)
                    seg_pred_mr = self.encoder_mr(img_mr)
                    loss = torch.stack([self.dice_loss_fn_mr(seg_pred_mr[:, i].to(DEVICE), seg_true_mr) for i in range(seg_pred_mr.shape[1])]).mean()

                self.scaler_mr_unet.scale(loss).backward()
                self.scaler_mr_unet.step(self.optimzer_mr_unet)
                self.scaler_mr_unet.update()
                
                loss_value = loss.item()
                train_loss_epoch["mr"] += loss_value
                train_loss_epoch["total"] += loss_value
                
                # Log wandb data for single random step per epoch  
                if step == log_mr_step:
                    # Get middle slice for visualization
                    input_img_mr_slice = img_mr[img_mr.shape[0] // 2, 0, :, :]
                    gt_slice_mr = seg_true_mr[seg_true_mr.shape[0] // 2, 0, :, :]
                    pred_slice_mr = torch.argmax(seg_pred_mr[seg_pred_mr.shape[0] // 2, 0, :, :, :], dim=0)    # deep_supervision enabled, multi-layer output exported.
                    
                    # Convert to numpy arrays for wandb logging using _prepare_slice_for_wandb
                    input_img_mr_viz = self._prepare_slice_for_wandb(input_img_mr_slice, is_segmentation=False)
                    gt_slice_mr_viz = self._prepare_slice_for_wandb(gt_slice_mr, is_segmentation=True, num_classes=self.super_params.num_classes)
                    pred_slice_mr_viz = self._prepare_slice_for_wandb(pred_slice_mr, is_segmentation=True, num_classes=self.super_params.num_classes)
                    
                    log_data_unet.update({
                        "unet/mr_input_slice": wandb.Image(input_img_mr_viz, caption=f"MR Input - Epoch {epoch+1}"),
                        "unet/mr_gt_slice": wandb.Image(gt_slice_mr_viz, caption=f"MR Ground Truth - Epoch {epoch+1}"),
                        "unet/mr_pred_slice": wandb.Image(pred_slice_mr_viz, caption=f"MR Prediction - Epoch {epoch+1}"),
                    })

        train_loss_epoch["mr"] = train_loss_epoch["mr"] / mr_step_count if self.mr_train_loader is not None and mr_step_count > 0 else 0.0
        
        if self.mr_train_loader is not None and len(self.mr_train_loader) > 0:
            print(f"MR UNet Training - Loss: {train_loss_epoch['mr']:.4f}, LR: {self.optimzer_mr_unet.param_groups[0]['lr']:.6f}")
        
        self.lr_scheduler_mr_unet.step(train_loss_epoch["mr"])

        train_loss_epoch["total"] = train_loss_epoch["ct"] + train_loss_epoch["mr"]
        train_loss_epoch["seg"] = train_loss_epoch["total"]

        for k in self.unet_loss.keys():
            self.unet_loss[k] = np.append(self.unet_loss[k], train_loss_epoch[k])

        # Add losses to wandb logging
        log_data_unet["unet/train_loss_ct"] = train_loss_epoch["ct"]
        log_data_unet["unet/train_loss_mr"] = train_loss_epoch["mr"]
        log_data_unet["unet/train_loss_total"] = train_loss_epoch["total"]
        
        print(f"UNet Total Loss: {train_loss_epoch['total']:.4f} (CT: {train_loss_epoch['ct']:.4f}, MR: {train_loss_epoch['mr']:.4f})")
        print(f"{'='*60}")

        # Always log to the same step to ensure consistency
        if log_data_unet:
            # Use continuous epoch-based logging for sweep runs to avoid step conflicts
            if hasattr(self.super_params, 'is_sweep_run') and self.super_params.is_sweep_run:
                continuous_epoch = self.orchestrator.get_next_continuous_epoch() if self.orchestrator else epoch + 1
                wandb.log({**log_data_unet, "epoch": continuous_epoch}, commit=commit_log)
            else:
                step = self.orchestrator.get_next_step() if self.orchestrator else epoch + 1
                wandb.log(log_data_unet, step=step, commit=commit_log)
    
    def _train_resnet_phase(self, epoch, commit_log=True):
        """
        Train ResNet decoder for segmentation refinement using CT data only.
        
        Architecture Note:
        - ResNet decoder is trained ONLY on CT data
        - During inference, the trained decoder can be applied to both CT and MR encoder outputs
        - MR data is used for validation/testing but NOT for training the ResNet decoder
        """
        self.encoder_ct.eval()
        self.decoder.train()

        train_loss_epoch = dict(total=0.0, df=0.0)
        ct_step_count = 0
        
        # Process CT data
        if self.ct_train_loader is not None:
            for step, data_ct in enumerate(self.ct_train_loader):
                ct_step_count = step + 1
                img_ct, seg_true_ct = (
                    data_ct["ct_image"].to(DEVICE),
                    data_ct["ct_label"].to(DEVICE),
                )
                
                self.optimizer_resnet.zero_grad()
                with torch.autocast(device_type=DEVICE, enabled=False):
                    # Use sliding window inference with GPU processing but CPU storage
                    with torch.no_grad():
                        seg_pred_ct = sliding_window_inference(
                            img_ct,
                            roi_size=self.super_params.crop_window_size,
                            sw_batch_size=8,
                            predictor=self.encoder_ct,
                            overlap=0.5,
                            mode="gaussian",
                            sw_device=DEVICE,  # Process windows on GPU
                            device=torch.device('cpu'),  # Store results on CPU
                            buffer_steps=4,
                            buffer_dim=-1,
                        )
                    
                    seg_pred_ct_ds_decoder_size = self.preprocessor._memory_efficient_post_transform(
                        seg_pred_ct, seg_true_ct, "ct", to_gpu=True, decoder_size=True)
                    
                    seg_true_ct_ds_decoder_size = self.preprocessor._generate_downsampled_gt(seg_true_ct, "ct", decoder_size=True)
                    
                    seg_pred_ct_ds = self.preprocessor._memory_efficient_post_transform(
                        seg_pred_ct, seg_true_ct, "ct", to_gpu=True, decoder_size=False)
                    
                    # Calculate mask for refinement
                    binary_mask_pred = (torch.argmax(seg_pred_ct_ds_decoder_size, dim=1, keepdim=True) > 0)
                    mask = torch.zeros_like(binary_mask_pred)
                    seg_np = binary_mask_pred[0, 0].cpu().numpy().astype(bool)
                    dilated_np = binary_dilation(seg_np, iterations=20)
                    mask[0, 0] = torch.from_numpy(dilated_np.astype(np.float32)).to(binary_mask_pred.device)
                    mask[binary_mask_pred == 1] = 0
                    
                    resnet_output = self.decoder(seg_pred_ct_ds)
                    
                    # Apply refinement
                    seg_pred_ct_ds_final = seg_pred_ct_ds_decoder_size + mask * resnet_output
                    
                    loss = self.msk_dice_loss_fn(seg_pred_ct_ds_final, seg_true_ct_ds_decoder_size)

                self.scaler_resnet.scale(loss).backward()
                self.scaler_resnet.step(self.optimizer_resnet)
                self.scaler_resnet.update()
                
                loss_value = loss.item()
                train_loss_epoch["total"] += loss_value
                train_loss_epoch["df"] += loss_value

        # Calculate average losses (CT only)
        for k, v in train_loss_epoch.items():
            train_loss_epoch[k] = v / ct_step_count if ct_step_count > 0 else 0.0
            self.resnet_loss[k] = np.append(self.resnet_loss.get(k, np.array([])), train_loss_epoch[k])

        print(f"ResNet Training (CT only) - Loss: {train_loss_epoch['total']:.4f}, LR: {self.optimizer_resnet.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")

        # Use epoch-based logging for sweep runs to avoid step conflicts
        resnet_log_data = {
            "resnet/train_loss_total": train_loss_epoch["total"],
            "resnet/train_loss_df": train_loss_epoch["df"]
        }
        
        if hasattr(self.super_params, 'is_sweep_run') and self.super_params.is_sweep_run:
            continuous_epoch = self.orchestrator.get_next_continuous_epoch() if self.orchestrator else epoch + 1
            wandb.log({**resnet_log_data, "epoch": continuous_epoch}, commit=commit_log)
        else:
            step = self.orchestrator.get_next_step() if self.orchestrator else epoch + 1
            wandb.log(resnet_log_data, step=step, commit=commit_log)

        self.lr_scheduler_resnet.step(train_loss_epoch["total"])
    
    def _train_gsn_phase(self, epoch, commit_log=True):
        """Train GSN for mesh refinement."""
        self.encoder_ct.eval()
        self.decoder.eval()
        self.GSN.train()

        finetune_loss_epoch = dict(total=0.0, chmf=0.0, smooth=0.0)
        gsn_step_count = 0
        if self.ct_train_loader is not None:
            for step, data_ct in enumerate(self.ct_train_loader):
                gsn_step_count = step + 1
                img_ct, seg_true_ct = (
                    data_ct["ct_image"].to(DEVICE),
                    data_ct["ct_label"].to(DEVICE)
                )
                
                seg_true_ct_ds = self.preprocessor._generate_downsampled_gt(seg_true_ct, "ct", decoder_size=32)
                mesh_true_ct = self.mesh_ops.surface_extractor(seg_true_ct_ds.to(DEVICE), labels=2)

                self.optimizer_gsn.zero_grad()
                with torch.autocast(device_type=DEVICE, enabled=False):
                    # Use sliding window inference with GPU processing but CPU storage
                    with torch.no_grad():
                        seg_pred_ct = sliding_window_inference(
                            img_ct,
                            roi_size=self.super_params.crop_window_size,
                            sw_batch_size=8,
                            predictor=self.encoder_ct,
                            overlap=0.5,
                            mode="gaussian",
                            sw_device=DEVICE,  # Process windows on GPU
                            device=torch.device('cpu'),  # Store results on CPU
                            buffer_steps=4,
                            buffer_dim=-1,
                        )
                    
                    # Process predictions through full pipeline with custom sequential transformation
                    seg_pred_ct_ds_decoder_size = self.preprocessor._memory_efficient_post_transform(
                        seg_pred_ct, seg_true_ct, "ct", to_gpu=True, decoder_size=True)
                    
                    seg_pred_ct_ds = self.preprocessor._memory_efficient_post_transform(
                        seg_pred_ct, seg_true_ct, "ct", to_gpu=True, decoder_size=False)
                    
                    # Calculate mask and apply ResNet
                    binary_mask_pred = (torch.argmax(seg_pred_ct_ds_decoder_size, dim=1, keepdim=True) > 0)
                    mask = torch.zeros_like(binary_mask_pred)
                    seg_np = binary_mask_pred[0, 0].cpu().numpy().astype(bool)
                    dilated_np = binary_dilation(seg_np, iterations=20)
                    mask[0, 0] = torch.from_numpy(dilated_np.astype(np.float32)).to(binary_mask_pred.device)
                    mask[binary_mask_pred == 1] = 0

                    resnet_output = self.decoder(seg_pred_ct_ds)
                    seg_pred_ct_ds_final = seg_pred_ct_ds_decoder_size + mask * resnet_output
                    seg_pred_ct_ds_final = torch.stack([self.pred_transform(i) for i in seg_pred_ct_ds_final])
                    seg_pred_ct_ds_final = F.interpolate(seg_pred_ct_ds_final, size=(32, 32, 32), mode="trilinear", align_corners=False)
                    
                    # Generate distance fields
                    foreground = seg_pred_ct_ds_final > 0
                    lv = (seg_pred_ct_ds_final == 1)
                    rv = (seg_pred_ct_ds_final == 3)
                    myo = (seg_pred_ct_ds_final == 2)
                    df_pred_ct = torch.stack([
                        distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                        for i in [foreground, lv, rv, myo]], dim=1)
                    
                    # Warp template and apply GSN
                    template_mesh = self.mesh_ops.warp_template_mesh(df_pred_ct.detach())
                    
                    level_outs = self.GSN(template_mesh, self.mesh_ops.subdivided_faces.faces_levels)

                    # Calculate losses
                    loss_chmf, loss_smooth = 0.0, 0.0
                    for l, subdiv_mesh in enumerate(level_outs):
                        verts_label = self.mesh_ops.subdivided_faces.labels_levels[l]
                        surface_mask = torch.any(torch.stack([verts_label == i for i in [0, 1, 2, 3]]), dim=0)
                        surface_verts = subdiv_mesh.verts_padded()[:, surface_mask]
                        
                        loss_chmf += chamfer_distance(
                            surface_verts, 
                            mesh_true_ct[0].verts_padded(),
                            point_reduction="mean", batch_reduction="mean"
                        )[0] 
                        loss_smooth += mesh_laplacian_smoothing(subdiv_mesh.update_padded(subdiv_mesh.verts_padded().to(torch.float32)), method="cot")
                    
                    loss = self.super_params.lambda_0 * loss_chmf + self.super_params.lambda_1 * loss_smooth
                
                self.scaler_gsn.scale(loss).backward()
                self.scaler_gsn.step(self.optimizer_gsn)
                self.scaler_gsn.update()
                
                loss_value = loss.item()
                loss_chmf_value = loss_chmf.item()
                loss_smooth_value = loss_smooth.item()
                finetune_loss_epoch["total"] += loss_value
                finetune_loss_epoch["chmf"] += loss_chmf_value
                finetune_loss_epoch["smooth"] += loss_smooth_value

                # Memory cleanup
                del seg_pred_ct, seg_pred_ct_ds, seg_pred_ct_ds_final, binary_mask_pred, mask
                del foreground, lv, myo, df_pred_ct, template_mesh, level_outs
                del loss_chmf, loss_smooth, loss
                torch.cuda.empty_cache()

        # Calculate average losses and update tracking
        for k, v in finetune_loss_epoch.items():
            finetune_loss_epoch[k] = v / gsn_step_count if self.ct_train_loader is not None and gsn_step_count > 0 else 0.0
            self.gsn_loss[k] = np.append(self.gsn_loss[k], finetune_loss_epoch[k])

        # Always print GSN training progress
        print(f"GSN Training - Total Loss: {finetune_loss_epoch['total']:.4f} "
              f"(Chamfer: {finetune_loss_epoch['chmf']:.4f}, Smooth: {finetune_loss_epoch['smooth']:.4f})")
        print(f"GSN Training - LR: {self.optimizer_gsn.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")

        # Log to WandB
        gsn_log_data = {
            "gsn/train_loss_total": finetune_loss_epoch["total"],
            "gsn/train_loss_chamfer": finetune_loss_epoch["chmf"],
            "gsn/train_loss_smooth": finetune_loss_epoch["smooth"]
        }
        
        if hasattr(self.super_params, 'is_sweep_run') and self.super_params.is_sweep_run:
            continuous_epoch = self.orchestrator.get_next_continuous_epoch() if self.orchestrator else epoch + 1
            wandb.log({**gsn_log_data, "epoch": continuous_epoch}, commit=commit_log)
        else:
            step = self.orchestrator.get_next_step() if self.orchestrator else epoch + 1
            wandb.log(gsn_log_data, step=step, commit=commit_log)

        # Update learning rate scheduler
        if self.ct_train_loader is not None and gsn_step_count > 0:
            self.lr_scheduler_gsn.step(finetune_loss_epoch["total"])