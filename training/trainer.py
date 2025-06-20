import os
import torch
import numpy as np
import wandb
from collections import OrderedDict
from monai.inferers import sliding_window_inference
from monai.transforms.utils import distance_transform_edt
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MorphiNetTrainer:
    """Handles training for different phases of MorphiNet: UNet, ResNet, and GSN."""
    
    def __init__(self, super_params, models, optimizers, schedulers, scalers, loss_functions, 
                 dataloaders, preprocessor, mesh_ops, inference, target=None):
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
            target: Training target
        """
        self.super_params = super_params
        self.target = target
        
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
        
        # Data loaders
        self.ct_train_loader = dataloaders.get('ct_train_loader')
        self.mr_train_loader = dataloaders.get('mr_train_loader')
        
        # Helper modules
        self.preprocessor = preprocessor
        self.mesh_ops = mesh_ops
        self.inference = inference
        
        # Loss tracking
        self.unet_loss = OrderedDict({k: np.asarray([]) for k in ["total", "seg"]})
        self.resnet_loss = OrderedDict({k: np.asarray([]) for k in ["total", "df"]})
        self.gsn_loss = OrderedDict({k: np.asarray([]) for k in ["total", "chmf", "smooth"]})
        
        # Sigmoid parameters
        self.sigmoid_scale_factor = super_params.sigmoid_scale_factor
        self.mask_threshold = super_params.mask_threshold
        
        # Prediction transform
        from monai.transforms import AsDiscrete
        self.pred_transform = AsDiscrete(argmax=True, to_onehot=self.super_params.num_classes)
    
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
                        roi_size=self.super_params.crop_window_size,
                        sw_batch_size=8, 
                        predictor=self.encoder_ct,
                        overlap=0.5, 
                        mode="gaussian",
                    ) 
                    loss = self.dice_loss_fn_ct(seg_pred_ct.to(DEVICE), seg_true_ct)

                self.scaler_ct_unet.scale(loss).backward()
                self.scaler_ct_unet.step(self.optimzer_ct_unet)
                self.scaler_ct_unet.update()
                
                train_loss_epoch["ct"] += loss.item()

                # Logging for CT
                if step == log_ct_step:
                    case_id_ct = os.path.basename(self.ct_train_loader.dataset.data[step]["ct_label"]).replace(".nii.gz", '').replace(".seg.nrrd", '')

                    if img_ct.dim() == 5 and img_ct.shape[2] > 0:
                        depth_slice_idx_ct = img_ct.shape[2] // 2
                        
                        input_img_ct_slice = img_ct[0, 0, depth_slice_idx_ct, :, :]
                        input_img_ct_viz = self.preprocessor._prepare_slice_for_wandb(input_img_ct_slice, is_segmentation=False)

                        gt_slice_ct = seg_true_ct[0, 0, depth_slice_idx_ct, :, :]
                        gt_slice_ct_viz = self.preprocessor._prepare_slice_for_wandb(gt_slice_ct, is_segmentation=True, num_classes=self.super_params.num_classes)

                        pred_slice_ct = torch.argmax(seg_pred_ct[0, :, depth_slice_idx_ct, :, :].to(DEVICE), dim=0)
                        pred_slice_ct_viz = self.preprocessor._prepare_slice_for_wandb(pred_slice_ct, is_segmentation=True, num_classes=self.super_params.num_classes)
                        
                        log_data_unet["unet/ct_input_image"] = wandb.Image(input_img_ct_viz, caption=f"Case ID: {case_id_ct}")
                        log_data_unet["unet/ct_ground_truth"] = wandb.Image(gt_slice_ct_viz, caption=f"Case ID: {case_id_ct}")
                        log_data_unet["unet/ct_prediction"] = wandb.Image(pred_slice_ct_viz, caption=f"Case ID: {case_id_ct}")

        train_loss_epoch["ct"] = train_loss_epoch["ct"] / (step + 1) if self.ct_train_loader is not None and len(self.ct_train_loader) > 0 else 0.0
        
        if self.ct_train_loader is not None and len(self.ct_train_loader) > 0:
            print(f"CT UNet Training - Loss: {train_loss_epoch['ct']:.4f}, LR: {self.optimzer_ct_unet.param_groups[0]['lr']:.6f}")
            
        self.lr_scheduler_ct_unet.step(train_loss_epoch["ct"])

        # Train MR segmentation encoder
        log_mr_step = np.random.randint(0, len(self.mr_train_loader)) if self.mr_train_loader is not None and len(self.mr_train_loader) > 0 else -1
        if self.mr_train_loader is not None:
            for step, data_mr in enumerate(self.mr_train_loader):
                img_mr, seg_true_mr = (
                    data_mr["mr_image"].as_tensor().to(DEVICE),
                    data_mr["mr_label"].as_tensor().to(DEVICE),
                )

                # Filter out slices without labels
                img_mr, seg_true_mr = self.preprocessor._filter_unlabeled_slices(img_mr, seg_true_mr)

                self.optimzer_mr_unet.zero_grad()
                with torch.autocast(device_type=DEVICE):
                    seg_pred_mr = sliding_window_inference(
                        img_mr,
                        roi_size=self.super_params.crop_window_size[:2],
                        sw_batch_size=8,
                        predictor=self.encoder_mr,
                        overlap=0.5,
                        mode="gaussian",
                    )
                    loss = self.dice_loss_fn_mr(seg_pred_mr.to(DEVICE), seg_true_mr)

                self.scaler_mr_unet.scale(loss).backward()
                self.scaler_mr_unet.step(self.optimzer_mr_unet)
                self.scaler_mr_unet.update()
                
                train_loss_epoch["mr"] += loss.item()

                # Logging for MR
                if step == log_mr_step:
                    case_id_mr = os.path.basename(self.mr_train_loader.dataset.data[step]["mr_label"]).replace(".nii.gz", '').replace(".seg.nrrd", '')
                    case_id_mr = case_id_mr.split('-')[0]
                    
                    slice_idx_mr = seg_true_mr.shape[0] // 4

                    input_img_mr_slice = img_mr[slice_idx_mr, 0]
                    input_img_mr_viz = self.preprocessor._prepare_slice_for_wandb(input_img_mr_slice, is_segmentation=False)

                    gt_slice_mr = seg_true_mr[slice_idx_mr, 0]
                    gt_slice_mr_viz = self.preprocessor._prepare_slice_for_wandb(gt_slice_mr, is_segmentation=True, num_classes=self.super_params.num_classes)

                    pred_slice_mr = torch.argmax(seg_pred_mr[slice_idx_mr].to(DEVICE), dim=0)
                    pred_slice_mr_viz = self.preprocessor._prepare_slice_for_wandb(pred_slice_mr, is_segmentation=True, num_classes=self.super_params.num_classes)

                    log_data_unet["unet/mr_input_image"] = wandb.Image(input_img_mr_viz, caption=f"Case ID: {case_id_mr}")
                    log_data_unet["unet/mr_ground_truth"] = wandb.Image(gt_slice_mr_viz, caption=f"Case ID: {case_id_mr}")
                    log_data_unet["unet/mr_prediction"] = wandb.Image(pred_slice_mr_viz, caption=f"Case ID: {case_id_mr}")

        train_loss_epoch["mr"] = train_loss_epoch["mr"] / (step + 1) if self.mr_train_loader is not None and len(self.mr_train_loader) > 0 else 0.0
        
        if self.mr_train_loader is not None and len(self.mr_train_loader) > 0:
            print(f"MR UNet Training - Loss: {train_loss_epoch['mr']:.4f}, LR: {self.optimzer_mr_unet.param_groups[0]['lr']:.6f}")
        
        self.lr_scheduler_mr_unet.step(train_loss_epoch["mr"])

        train_loss_epoch["total"] = train_loss_epoch["ct"] + train_loss_epoch["mr"]
        train_loss_epoch["seg"] = train_loss_epoch["total"]

        for k, v in self.unet_loss.items():
            self.unet_loss[k] = np.append(self.unet_loss[k], train_loss_epoch[k])

        # Add losses to wandb logging
        log_data_unet["unet/train_loss_ct"] = train_loss_epoch["ct"]
        log_data_unet["unet/train_loss_mr"] = train_loss_epoch["mr"]
        log_data_unet["unet/train_loss_total"] = train_loss_epoch["total"]
        
        print(f"UNet Total Loss: {train_loss_epoch['total']:.4f} (CT: {train_loss_epoch['ct']:.4f}, MR: {train_loss_epoch['mr']:.4f})")
        print(f"{'='*60}")

        if log_data_unet:
            wandb.log(log_data_unet, step=epoch + 1, commit=commit_log)
    
    def _train_resnet_phase(self, epoch, commit_log=True):
        """Train ResNet for distance field prediction."""
        self.encoder_ct.eval()
        self.decoder.train()

        train_loss_epoch = dict(total=0.0, df=0.0)
        if self.ct_train_loader is not None:
            for step, data_ct in enumerate(self.ct_train_loader):
                img_ct, seg_true_ct = (
                    data_ct["ct_image"].to(DEVICE),
                    data_ct["ct_label"].to(DEVICE),
                )
                
                self.optimizer_resnet.zero_grad()
                with torch.autocast(device_type=DEVICE):
                    seg_pred_ct = sliding_window_inference(
                        img_ct,
                        roi_size=self.super_params.crop_window_size,
                        sw_batch_size=8,
                        predictor=self.encoder_ct,
                        overlap=0.5,
                        mode="gaussian",
                    )
                    
                    # Process predictions and generate ground truth at decoder size
                    seg_pred_ct_ds_decoder_size = self.preprocessor._memory_efficient_post_transform(
                        seg_pred_ct, seg_true_ct, "ct", to_gpu=True, decoder_size=True)
                    
                    seg_true_ct_ds_decoder_size = torch.stack([
                        self.preprocessor._generate_downsampled_gt(seg_true_item, "ct", decoder_size=True)
                        for seg_true_item in seg_true_ct
                    ])
                    
                    seg_pred_ct_ds = self.preprocessor._memory_efficient_post_transform(
                        seg_pred_ct, seg_true_ct, "ct", to_gpu=True, decoder_size=False)
                    
                    # Calculate mask for refinement
                    binary_mask_pred = (torch.argmax(seg_pred_ct_ds_decoder_size, dim=1, keepdim=True) == 0)
                    dist_map_pred = (-distance_transform_edt(binary_mask_pred.squeeze(1)) + distance_transform_edt(~binary_mask_pred.squeeze(1))).unsqueeze(1)
                    mask = torch.sigmoid(dist_map_pred * self.sigmoid_scale_factor + 1).detach()
                    mask = mask * binary_mask_pred
                    mask[mask < self.mask_threshold] = 0
                    
                    # Apply ResNet with padding
                    seg_pred_ct_ds_padded, pad_info = self.inference._apply_resnet_padding(seg_pred_ct_ds)
                    resnet_output_padded = self.decoder(seg_pred_ct_ds_padded)
                    resnet_output = self.inference._remove_resnet_padding(resnet_output_padded, pad_info)
                    
                    # Apply refinement
                    seg_pred_ct_ds_final = seg_pred_ct_ds_decoder_size + mask * resnet_output
                    loss = self.msk_dice_loss_fn(seg_pred_ct_ds_final, seg_true_ct_ds_decoder_size)

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

        wandb.log({
            "resnet/train_loss_total": train_loss_epoch["total"]
        }, step=epoch + 1, commit=commit_log)

        self.lr_scheduler_resnet.step(train_loss_epoch["total"])
    
    def _train_gsn_phase(self, epoch, commit_log=True):
        """Train GSN for mesh refinement."""
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
                
                # Generate ground truth mesh
                seg_true_ct_ds = torch.stack([
                    self.preprocessor._generate_downsampled_gt(seg_true_item, "ct", decoder_size=False)
                    for seg_true_item in seg_true_ct
                ])
                mesh_true_ct = self.mesh_ops.surface_extractor(seg_true_ct_ds.to(DEVICE), labels=2)

                self.optimizer_gsn.zero_grad()
                with torch.autocast(device_type=DEVICE):
                    seg_pred_ct = sliding_window_inference(
                        img_ct,
                        roi_size=self.super_params.crop_window_size,
                        sw_batch_size=4,
                        predictor=self.encoder_ct,
                        overlap=0.5,
                        mode="gaussian",
                        device=torch.device('cpu'),
                        buffer_steps=4,
                        buffer_dim=-1,
                    )
                    
                    # Process predictions through full pipeline
                    seg_pred_ct_ds_decoder_size = self.preprocessor._memory_efficient_post_transform(
                        seg_pred_ct, seg_true_ct, "ct", to_gpu=True, decoder_size=True)
                    
                    seg_pred_ct_ds = self.preprocessor._memory_efficient_post_transform(
                        seg_pred_ct, seg_true_ct, "ct", to_gpu=True, decoder_size=False)
                    
                    # Calculate mask and apply ResNet
                    binary_mask_pred = (torch.argmax(seg_pred_ct_ds_decoder_size, dim=1, keepdim=True) == 0)
                    dist_map_pred = (-distance_transform_edt(binary_mask_pred.squeeze(1)) + distance_transform_edt(~binary_mask_pred.squeeze(1))).unsqueeze(1)
                    mask = torch.sigmoid(dist_map_pred * self.sigmoid_scale_factor + 1).detach()
                    mask = mask * binary_mask_pred
                    mask[mask < self.mask_threshold] = 0

                    seg_pred_ct_ds_padded, pad_info = self.inference._apply_resnet_padding(seg_pred_ct_ds)
                    resnet_output_padded = self.decoder(seg_pred_ct_ds_padded)
                    resnet_output = self.inference._remove_resnet_padding(resnet_output_padded, pad_info)
                    
                    seg_pred_ct_ds = seg_pred_ct_ds_decoder_size + mask * resnet_output
                    seg_pred_ct_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ct_ds])
                    
                    # Generate distance fields
                    foreground = seg_pred_ct_ds > 0
                    lv = (seg_pred_ct_ds == 1)
                    rv = (seg_pred_ct_ds == 3)
                    myo = (seg_pred_ct_ds == 2)
                    df_pred_ct = torch.stack([
                        distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                        for i in [foreground, lv, rv, myo]], dim=1)
                    
                    # Warp template and apply GSN
                    template_mesh = self.mesh_ops.warp_template_mesh(df_pred_ct.detach())
                    template_mesh = template_mesh.update_padded(template_mesh.verts_padded().to(torch.float16))
                    
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
                        loss_smooth += mesh_laplacian_smoothing(subdiv_mesh, method="cot")
                    
                    loss = self.super_params.lambda_0 * loss_chmf + self.super_params.lambda_1 * loss_smooth

                self.scaler_gsn.scale(loss).backward()
                self.scaler_gsn.step(self.optimizer_gsn)
                self.scaler_gsn.update()
                
                finetune_loss_epoch["total"] += loss.item()
                finetune_loss_epoch["chmf"] += loss_chmf.item()
                finetune_loss_epoch["smooth"] += loss_smooth.item()
                
                # Memory cleanup
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

            wandb.log({
                "gsn/train_loss_total": finetune_loss_epoch["total"],
                "gsn/train_loss_chamfer": finetune_loss_epoch["chmf"],
                "gsn/train_loss_smooth": finetune_loss_epoch["smooth"]
            }, step=epoch + 1, commit=commit_log)

            self.lr_scheduler_gsn.step(finetune_loss_epoch["total"])