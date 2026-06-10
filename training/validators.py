import os
import torch
import torch.nn.functional as F
import numpy as np
import wandb
from scipy.ndimage import binary_dilation
from utils.tools import draw_plotly
from utils.path_config import get_path_default
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MSEMetric
from monai.transforms.utils import distance_transform_edt
from monai.transforms import (
    AsDiscrete, 
    Compose, 
    KeepLargestConnectedComponent,
)
from pytorch3d.io import save_obj


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def debug_pixdim_info(tensor, name, context=""):
    """
    Debug utility to extract and display pixdim information from MetaTensor objects.
    
    Args:
        tensor: MetaTensor object or regular tensor
        name: Name identifier for the tensor
        context: Additional context information
    """
    try:
        if hasattr(tensor, 'pixdim'):
            pixdim = tensor.pixdim.cpu().numpy() if hasattr(tensor.pixdim, 'cpu') else tensor.pixdim
            print(f"DEBUG [{context}] {name}: pixdim={pixdim}")
        elif hasattr(tensor, 'affine'):
            # Calculate pixdim from affine matrix
            affine = tensor.affine.cpu().numpy() if hasattr(tensor.affine, 'cpu') else tensor.affine
            pixdim = [np.linalg.norm(affine[:3, i]) for i in range(3)]
            print(f"DEBUG [{context}] {name}: pixdim (from affine)={pixdim}")
        else:
            print(f"DEBUG [{context}] {name}: No pixdim/affine info available")
        
        # Also display shape information
        if hasattr(tensor, 'shape'):
            shape = tensor.shape
            print(f"DEBUG [{context}] {name}: shape={shape}")
        elif hasattr(tensor, 'get_array'):
            array = tensor.get_array()
            shape = array.shape
            print(f"DEBUG [{context}] {name}: shape={shape}")
            
    except Exception as e:
        print(f"DEBUG [{context}] {name}: Error extracting pixdim info: {e}")


class MorphiNetValidator:
    """Handles validation for MorphiNet full pipeline."""
    
    def __init__(self, super_params, models, dataloaders, preprocessor, mesh_ops, inference, ckpt_dir, orchestrator=None):
        """
        Initialize the validator.
        
        Args:
            super_params: Configuration parameters
            models: Dictionary containing model instances
            dataloaders: Dictionary containing data loaders
            preprocessor: Data preprocessor instance
            mesh_ops: Mesh operations instance
            inference: Model inference instance
            ckpt_dir: Checkpoint directory path
            orchestrator: Reference to orchestrator for step management
        """
        self.super_params = super_params
        self.ckpt_dir = ckpt_dir
        self.orchestrator = orchestrator
        
        # Models
        self.encoder_mr = models['encoder_mr']
        self.encoder_ct = models['encoder_ct']
        self.decoder = models['decoder']
        self.GSN = models['GSN']
        
        # Data loaders - store reference to manager for dynamic access
        self.dataloader_manager = dataloaders
        
        # Helper modules
        self.preprocessor = preprocessor
        self.mesh_ops = mesh_ops
        self.inference = inference
        
        # Evaluation score tracking
        self.best_eval_score = 0.0
        self.eval_df_score = {"myo": np.asarray([])}
        self.eval_msh_score = {"myo": np.asarray([])}
        
        # Prediction transform
        self.pred_transform = Compose([
            AsDiscrete(argmax=True),
            KeepLargestConnectedComponent(is_onehot=True, independent=False, connectivity=3),
        ])
        
        # Rasterizer
        if hasattr(mesh_ops, 'rasterizer'):
            self.rasterizer = mesh_ops.rasterizer
    
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
    
    def _generate_visualizations(self, cached_data):
        """Generate and save visualizations."""
        visualization_dir = f"{self.ckpt_dir}/visualizations"
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Prepare log data for wandb
        log_data_viz = {}
        
        # Create visualizations with wandb logging
        try:
            # Visualization 1: Segmentation Ground Truth vs Mesh Prediction
            draw_plotly(
                seg_true=cached_data["seg_true_ds"], 
                mesh_pred=cached_data["subdiv_mesh"],
                save_html=True,
                save_dir=visualization_dir,
                filename="seg_true_ds_vs_mesh_pred.html",
                export_static=True,
                export_png_filename="seg_true_ds_vs_mesh_pred.png"
            )
            
            # Log to wandb if PNG was created successfully
            png_path = os.path.join(visualization_dir, "seg_true_ds_vs_mesh_pred.png")
            if os.path.exists(png_path):
                log_data_viz["gsn/seg_true_vs_mesh_pred"] = wandb.Image(png_path)
            
            # Visualization 2: Segmentation Ground Truth vs Segmentation Prediction
            if "seg_pred_ds" in cached_data:
                draw_plotly(
                    seg_true=cached_data["seg_true_ds"], 
                    seg_pred=cached_data["seg_pred_ds"],
                    save_html=True,
                    save_dir=visualization_dir,
                    filename="seg_true_ds_vs_seg_pred_ds.html",
                    export_static=True,
                    export_png_filename="seg_true_ds_vs_seg_pred_ds.png"
                )
                
                png_path = os.path.join(visualization_dir, "seg_true_ds_vs_seg_pred_ds.png")
                if os.path.exists(png_path):
                    log_data_viz["gsn/seg_true_vs_seg_pred"] = wandb.Image(png_path)
            
            # Visualization 3: Template Mesh vs Distance Field Prediction
            draw_plotly(
                seg_true=cached_data["seg_true_ds"], 
                mesh_pred=cached_data["template_mesh"],
                save_html=True,
                save_dir=visualization_dir,
                filename="seg_true_ds_vs_template_mesh.html",
                export_static=True,
                export_png_filename="seg_true_ds_vs_template_mesh.png"
            )
            
            png_path = os.path.join(visualization_dir, "seg_true_ds_vs_template_mesh.png")
            if os.path.exists(png_path):
                log_data_viz["gsn/template_vs_seg_true"] = wandb.Image(png_path)
            
            # # Visualization 4: Distance Field Prediction vs Ground Truth
            # if "df_pred" in cached_data:
            #     draw_plotly(
            #         seg_true=cached_data["seg_true_ds"], 
            #         df_pred=cached_data["df_pred"],
            #         save_html=True,
            #         save_dir=visualization_dir,
            #         filename="seg_true_ds_vs_df_pred.html",
            #         export_static=True,
            #         export_png_filename="seg_true_ds_vs_df_pred.png"
            #     )
                
            #     png_path = os.path.join(visualization_dir, "seg_true_ds_vs_df_pred.png")
            #     if os.path.exists(png_path):
            #         log_data_viz["gsn/seg_true_vs_df_pred"] = wandb.Image(png_path)
            
            # # Visualization 5: Distance Field Distribution Plot (if available)
            # if "df_true" in cached_data and "df_pred" in cached_data:
            #     try:
            #         import plotly.figure_factory as ff
                    
            #         # Ensure tensors are on CPU before conversion to numpy
            #         dist_fig = ff.create_distplot(
            #             [cached_data["df_true"][-1].flatten().cpu().numpy(), 
            #              cached_data["df_pred"][-1].flatten().cpu().numpy()],
            #             group_labels=["df_true", "df_pred"],
            #             colors=["#EF553B", "#3366CC"],
            #             bin_size=0.1
            #         )
                    
            #         # Save the distribution plot
            #         dist_fig.write_image(f"{visualization_dir}/df_true_vs_pred.png")
            #         log_data_viz["gsn/df_true_vs_pred"] = wandb.Image(f"{visualization_dir}/df_true_vs_pred.png")
                    
            #     except Exception as e:
            #         print(f"Warning: Could not generate distance field distribution plot: {e}")
            
            print("Visualizations saved successfully!")
            
            # Log all visualizations to wandb if any were created
            if log_data_viz:
                # Use commit=False here since this is called from _save_best_model which already commits
                # No need for step increment here as it's called within the same validation context
                wandb.log(log_data_viz, commit=False)
                print(f"Uploaded {len(log_data_viz)} visualizations to wandb")
            
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
    
    def _save_model_checkpoints(self, epoch):
        """Save model checkpoints for current epoch."""
        ckpt_weight_path = os.path.join(self.ckpt_dir, "trained_weights")
        os.makedirs(ckpt_weight_path, exist_ok=True)
        
        torch.save(self.GSN.state_dict(), os.path.join(ckpt_weight_path, f"{epoch + 1}_GSN.pth"))
        
        # Save subdivision faces
        for level, faces in enumerate(self.mesh_ops.subdivided_faces.faces_levels):
            torch.save(faces, os.path.join(ckpt_weight_path, f"{epoch+1}_subdivided_faces_l{level}.pth"))
    
    def _save_best_model(self, epoch, eval_score_epoch, cached_data):
        """Save best model and generate visualizations."""
        ckpt_weight_path = os.path.join(self.ckpt_dir, "trained_weights")
        
        torch.save(self.GSN.state_dict(), os.path.join(ckpt_weight_path, f"best_GSN.pth"))
        
        # Save best subdivision faces
        for level, faces in enumerate(self.mesh_ops.subdivided_faces.faces_levels):
            torch.save(faces, os.path.join(ckpt_weight_path, f"best_subdivided_faces_l{level}.pth"))
        
        # Update best score
        self.best_eval_score = eval_score_epoch
        wandb.run.summary["best_eval_score"] = eval_score_epoch
        
        print(f"*** NEW BEST VALIDATION SCORE: {eval_score_epoch:.4f} ***")
        print(f"Saving best model and generating visualizations...")
        
        # Generate visualizations
        self._generate_visualizations(cached_data)
    
    def _save_unet_checkpoints(self, epoch, modal):
        """Save UNet model checkpoints for current epoch."""
        ckpt_weight_path = os.path.join(self.ckpt_dir, "trained_weights")
        os.makedirs(ckpt_weight_path, exist_ok=True)
        
        # Save both CT and MR UNet models
        torch.save(self.encoder_ct.state_dict(), os.path.join(ckpt_weight_path, f"{epoch + 1}_UNet_CT.pth"))
        torch.save(self.encoder_mr.state_dict(), os.path.join(ckpt_weight_path, f"{epoch + 1}_UNet_MR.pth"))
    
    def _save_best_unet(self, epoch, modal, dice_score):
        """Save best UNet model when validation score improves."""
        ckpt_weight_path = os.path.join(self.ckpt_dir, "trained_weights")
        os.makedirs(ckpt_weight_path, exist_ok=True)
        
        # Save best UNet models
        torch.save(self.encoder_ct.state_dict(), os.path.join(ckpt_weight_path, f"best_UNet_CT.pth"))
        torch.save(self.encoder_mr.state_dict(), os.path.join(ckpt_weight_path, f"best_UNet_MR.pth"))
        
        print(f"New best UNet model saved! {modal.upper()} Dice: {dice_score:.4f}")
        wandb.run.summary[f"best_unet_{modal}_dice"] = dice_score
    
    def _save_resnet_checkpoints(self, epoch):
        """Save ResNet model checkpoints for current epoch."""
        ckpt_weight_path = os.path.join(self.ckpt_dir, "trained_weights")
        os.makedirs(ckpt_weight_path, exist_ok=True)
        
        # Save ResNet models
        torch.save(self.decoder.state_dict(), os.path.join(ckpt_weight_path, f"{epoch + 1}_ResNet.pth"))
    
    def _save_best_resnet(self, epoch, dice_score):
        """Save best ResNet model when validation score improves."""
        ckpt_weight_path = os.path.join(self.ckpt_dir, "trained_weights")
        os.makedirs(ckpt_weight_path, exist_ok=True)
        
        # Save best ResNet models
        torch.save(self.decoder.state_dict(), os.path.join(ckpt_weight_path, f"best_ResNet.pth"))
        
        print(f"New best ResNet model saved! CT Dice: {dice_score:.4f}")
        wandb.run.summary[f"best_resnet_ct_dice"] = dice_score
    
    def validate_unet(self, epoch, modal):
        """
        Validate UNet performance only (Phase 1).
        
        Args:
            epoch: Current epoch number
            modal: Dataset to validate on ('sct' for CT, 'cap' for MR)
        
        Returns:
            Validation dice score
        """
        print(f"\n--- UNET VALIDATION ---")
        print(f"Phase: UNet Only, Modal: {modal.upper()}")
        
        # Choose validation components
        if modal == "ct":
            encoder = self.encoder_ct
            valid_loader = self.dataloader_manager.ct_valid_loader
            roi_size = self.super_params.crop_window_size
        elif modal == "mr":
            encoder = self.encoder_mr
            valid_loader = self.dataloader_manager.mr_valid_loader
            roi_size = self.super_params.crop_window_size[:2]
        else:
            raise ValueError(f"Invalid validation modality: {modal}. Use 'ct' or 'mr'")
        
        encoder.eval()
        
        # Save UNet checkpoints during validation
        self._save_unet_checkpoints(epoch, modal)
        
        # Initialize metrics
        dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        
        with torch.no_grad():
            for step, data in enumerate(valid_loader):
                img, seg_true = (
                    data[f"{modal}_image"].to(DEVICE),
                    data[f"{modal}_label"].to(DEVICE),
                )
                
                # Filter out slices without labels for MR
                if modal == 'mr':
                    img, seg_true = self.preprocessor._filter_unlabeled_slices(img, seg_true)
                    
                    # Check if we have any data left after filtering
                    if img.shape[0] == 0:
                        print(f"Warning: No labeled slices found in MR batch {step}, skipping")
                        continue
                    
                    # Adjust roi_size based on actual image dimensions
                    if len(img.shape) == 4:  # (B, C, H, W) - 2D slices
                        roi_size = roi_size  # Keep 2D roi_size
                    elif len(img.shape) == 5:  # (B, C, H, W, D) - 3D volume
                        roi_size = self.super_params.crop_window_size  # Use 3D roi_size
                
                # Run segmentation inference
                seg_pred = sliding_window_inference(
                    img, 
                    roi_size=roi_size, 
                    sw_batch_size=8, 
                    predictor=encoder,
                    overlap=0.5, 
                    mode="gaussian",
                )
                
                # Convert to one-hot for metric computation
                seg_pred_onehot = self.inference._convert_to_onehot(seg_pred, self.super_params.num_classes, is_prediction=True)
                seg_true_onehot = self.inference._convert_to_onehot(seg_true, self.super_params.num_classes, is_prediction=False)

                dice_metric(seg_pred_onehot, seg_true_onehot)
        
        # Calculate and log results
        dice_scores = dice_metric.aggregate()
        
        # Extract individual dice scores for cardiac structures
        if len(dice_scores) >= 3:
            lv_dice = dice_scores[0].item() if not torch.isnan(dice_scores[0]) else 0.0
            myo_dice = dice_scores[1].item() if not torch.isnan(dice_scores[1]) else 0.0
            rv_dice = dice_scores[2].item() if not torch.isnan(dice_scores[2]) else 0.0
            avg_dice = dice_scores.mean().item() if not torch.isnan(dice_scores.mean()) else 0.0
        else:
            lv_dice = myo_dice = rv_dice = avg_dice = 0.0
        
        print(f"{modal.upper()} Dice Scores - LV: {lv_dice:.4f}, MYO: {myo_dice:.4f}, RV: {rv_dice:.4f}")
        print(f"Average Dice Score: {avg_dice:.4f}")
        
        # Log segmentation validation metrics with proper step management
        unet_val_data = {
            f"unet/val_{modal}_dice_lv": lv_dice,
            f"unet/val_{modal}_dice_myo": myo_dice,
            f"unet/val_{modal}_dice_rv": rv_dice,
            f"unet/val_{modal}_dice_avg": avg_dice
        }
        
        # Use epoch-based logging for sweep runs to avoid step conflicts
        commit_log = True if modal == "mr" else False  # validate_unet is called twice for CT and MR
        if hasattr(self.super_params, 'is_sweep_run') and self.super_params.is_sweep_run:
            # For validation, use current continuous epoch (don't increment)
            continuous_epoch = self.orchestrator.get_current_continuous_epoch() if self.orchestrator else epoch + 1
            wandb.log({**unet_val_data, "epoch": continuous_epoch}, commit=commit_log)
        else:
            step = self.orchestrator.get_next_step() if self.orchestrator else epoch + 1
            wandb.log(unet_val_data, step=step, commit=commit_log)
        
        # Check for best model and save if improved (UNet-specific)
        # Track best score for each modality independently during training
        current_best_key = f"best_unet_{modal}_dice"
        if not hasattr(self, current_best_key):
            setattr(self, current_best_key, 0.0)
        
        if avg_dice > getattr(self, current_best_key):
            setattr(self, current_best_key, avg_dice)
            self._save_best_unet(epoch, modal, avg_dice)
        
        return avg_dice
    
    def validate_resnet(self, epoch):
        """
        Validate ResNet performance (UNet + ResNet pipeline without GSN).
        Focuses on segmentation refinement quality through dice scores.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average dice score for segmentation quality
        """
        print(f"\n--- RESNET VALIDATION ---")
        print(f"Phase: UNet + ResNet Only, Modal: CT")
                
        self.decoder.eval()
        self.encoder_ct.eval()
        
        # Save ResNet checkpoints during validation
        self._save_resnet_checkpoints(epoch)
        
        # Initialize dice metric only (no distance field metrics needed for ResNet validation)
        dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        
        with torch.no_grad():
            for step, data in enumerate(self.dataloader_manager.ct_valid_loader):
                # Load only image and label data (no distance fields needed)
                img, seg_true = (
                    data[f"ct_image"].to(DEVICE),
                    data[f"ct_label"].to(DEVICE),
                )            
                
                # Run inference through UNet
                seg_pred = sliding_window_inference(
                    img, 
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
                
                # Process predictions through ResNet pipeline (unflatten handled internally)
                seg_pred_ds_decoder_size = self.preprocessor._memory_efficient_post_transform(
                    seg_pred, seg_true, "ct", to_gpu=True, decoder_size=True)
                                
                seg_pred_ds = self.preprocessor._memory_efficient_post_transform(
                    seg_pred, seg_true, "ct", to_gpu=True, decoder_size=False)
                                
                # Calculate mask for refinement
                binary_mask_pred = (torch.argmax(seg_pred_ds_decoder_size, dim=1, keepdim=True) > 0)
                mask = torch.zeros_like(binary_mask_pred)
                seg_np = binary_mask_pred[0, 0].cpu().numpy().astype(bool)
                dilated_np = binary_dilation(seg_np, iterations=20)
                mask[0, 0] = torch.from_numpy(dilated_np.astype(np.float32)).to(binary_mask_pred.device)
                mask[binary_mask_pred == 1] = 0
                
                # Combine predictions (ResNet refined segmentation)
                resnet_output = self.decoder(seg_pred_ds)
                seg_pred_ds_refined = seg_pred_ds_decoder_size + mask * resnet_output
                
                # Generate downsampled ground truth at decoder size
                seg_true_ds_decoder_size = self.preprocessor._generate_downsampled_gt(seg_true, "ct", decoder_size=True)
                                
                # Convert to one-hot for dice metric computation (ResNet refined vs downsampled ground truth)
                seg_pred_onehot = self.inference._convert_to_onehot(seg_pred_ds_refined, self.super_params.num_classes, is_prediction=True)
                seg_true_onehot = self.inference._convert_to_onehot(seg_true_ds_decoder_size, self.super_params.num_classes, is_prediction=False)
                
                # Compute dice metric (ResNet validation - segmentation quality only)
                dice_metric(seg_pred_onehot, seg_true_onehot)
        
        # Calculate dice scores
        dice_scores = dice_metric.aggregate()
        
        # Extract individual dice scores for cardiac structures
        if len(dice_scores) >= 3:
            lv_dice = dice_scores[0].item() if not torch.isnan(dice_scores[0]) else 0.0
            myo_dice = dice_scores[1].item() if not torch.isnan(dice_scores[1]) else 0.0
            rv_dice = dice_scores[2].item() if not torch.isnan(dice_scores[2]) else 0.0
            avg_dice = dice_scores.mean().item() if not torch.isnan(dice_scores.mean()) else 0.0
        else:
            lv_dice = myo_dice = rv_dice = avg_dice = 0.0
        
        print(f"CT Dice Scores (ResNet refined) - LV: {lv_dice:.4f}, MYO: {myo_dice:.4f}, RV: {rv_dice:.4f}")
        print(f"Average Dice Score: {avg_dice:.4f}")
        
        # Log ResNet validation metrics (segmentation quality only - no distance fields)
        resnet_val_data = {
            f"resnet/val_ct_dice_lv": lv_dice,
            f"resnet/val_ct_dice_myo": myo_dice,
            f"resnet/val_ct_dice_rv": rv_dice,
            f"resnet/val_ct_dice_avg": avg_dice
        }
        
        # Use epoch-based logging for sweep runs to avoid step conflicts
        if hasattr(self.super_params, 'is_sweep_run') and self.super_params.is_sweep_run:
            continuous_epoch = self.orchestrator.get_current_continuous_epoch() if self.orchestrator else epoch + 1
            wandb.log({**resnet_val_data, "epoch": continuous_epoch}, commit=True)
        else:
            step = self.orchestrator.get_next_step() if self.orchestrator else epoch + 1
            wandb.log(resnet_val_data, step=step, commit=True)
        
        # Check for best model and save if improved (ResNet-specific) 
        # Track best score for CT modality during training (ResNet is CT-trained)
        current_best_key = f"best_resnet_ct_dice"
        if not hasattr(self, current_best_key):
            setattr(self, current_best_key, 0.0)
        
        if avg_dice > getattr(self, current_best_key):
            setattr(self, current_best_key, avg_dice)
            self._save_best_resnet(epoch, avg_dice)
        
        return avg_dice
    
    def validate_gsn(self, epoch):
        """
        Validate GSN performance with full pipeline (Phase 3).
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Validation metrics (eval_score_epoch, df_score_epoch)
        """
        print(f"\n--- GSN VALIDATION ---")
        print(f"Phase: Full Pipeline (UNet + ResNet + GSN), Modal: CT")
        
        self.decoder.eval()
        self.encoder_ct.eval()
        self.GSN.eval()
        
        # Save model checkpoints
        self._save_model_checkpoints(epoch)
                
        # Initialize metrics
        df_metric_batch_decoder = MSEMetric(reduction="mean_batch")
        msh_metric_batch_decoder = DiceMetric(include_background=False, reduction="mean_batch")
        
        cached_data = dict()
        choice_case = np.random.choice(len(self.dataloader_manager.ct_valid_loader), 1)[0]
        
        with torch.no_grad():
            for step, data in enumerate(self.dataloader_manager.ct_valid_loader):
                img, seg_true, df_true = (
                    data[f"ct_image"].to(DEVICE),
                    data[f"ct_label"].to(DEVICE),
                    data[f"ct_df"].as_tensor().to(DEVICE),
                )

                # Generate downsampled ground truth at decoder size
                seg_true_ds = self.preprocessor._generate_downsampled_gt(seg_true, "ct", decoder_size=True)
                
                # Run inference through full pipeline
                seg_pred = sliding_window_inference(
                    img, 
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
                
                # Process predictions through ResNet pipeline
                seg_pred_ds_decoder_size = self.preprocessor._memory_efficient_post_transform(
                    seg_pred, seg_true, "ct", to_gpu=True, decoder_size=True)
                
                seg_pred_ds = self.preprocessor._memory_efficient_post_transform(
                    seg_pred, seg_true, "ct", to_gpu=True, decoder_size=False)
                
                # Calculate mask for refinement
                binary_mask_pred = (torch.argmax(seg_pred_ds_decoder_size, dim=1, keepdim=True) > 0)
                mask = torch.zeros_like(binary_mask_pred)
                seg_np = binary_mask_pred[0, 0].cpu().numpy().astype(bool)
                dilated_np = binary_dilation(seg_np, iterations=20)
                mask[0, 0] = torch.from_numpy(dilated_np.astype(np.float32)).to(binary_mask_pred.device)
                mask[binary_mask_pred == 1] = 0
                
                # Combine predictions
                resnet_output = self.decoder(seg_pred_ds)
                seg_pred_ds = seg_pred_ds_decoder_size + mask * resnet_output
                seg_pred_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ds])
                
                # Generate distance fields
                foreground = seg_pred_ds > 0
                lv = (seg_pred_ds == 1)
                rv = (seg_pred_ds == 3)
                myo = (seg_pred_ds == 2)
                df_pred = torch.stack([
                    distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                    for i in [foreground, lv, rv, myo]
                    ], dim=1)
                
                df_metric_batch_decoder(df_pred, df_true)
                
                # Generate mesh predictions
                template_mesh = self.mesh_ops.warp_template_mesh(
                    F.interpolate(df_pred, size=(32, 32, 32), mode="trilinear", align_corners=False)
                )

                subdiv_mesh = self.GSN(template_mesh, self.mesh_ops.subdivided_faces.faces_levels, df_pred, self.mesh_ops.subdivided_faces.labels_levels)[-1]
                
                # Rasterize mesh for comparison
                voxeld_mesh = self.rasterizer(
                    subdiv_mesh[0].verts_padded(), subdiv_mesh[0].faces_padded())
                
                seg_true_ds = (seg_true_ds == 2).to(torch.float32)
                msh_metric_batch_decoder(voxeld_mesh, seg_true_ds)

                # For ease of demonstration, resize data before caching
                seg_pred_ds = F.interpolate(seg_pred_ds, size=(16, 16, 16), mode="nearest-exact")
                seg_true_ds = F.interpolate(seg_true_ds, size=(16, 16, 16), mode="nearest-exact")

                # Cache data for visualization
                if step == choice_case:
                    cached_data = {
                        # "df_true": df_true[0].cpu(),
                        # "df_pred": df_pred[0].cpu(),
                        "seg_pred_ds": seg_pred_ds[0].cpu(),
                        "seg_true_ds": seg_true_ds[0].cpu(),
                        "subdiv_mesh": subdiv_mesh[0].cpu(),
                        "template_mesh": template_mesh[0].cpu(),
                    }

        # Calculate metrics
        eval_score_epoch = msh_metric_batch_decoder.aggregate().mean()
        df_score_epoch = df_metric_batch_decoder.aggregate().mean()
        
        # Update tracking arrays
        self.eval_df_score["myo"] = np.append(self.eval_df_score["myo"], df_metric_batch_decoder.aggregate().cpu())
        self.eval_msh_score["myo"] = np.append(self.eval_msh_score["myo"], msh_metric_batch_decoder.aggregate().cpu())
        
        print(f"Mesh Dice Score: {eval_score_epoch:.4f}")
        print(f"Distance Field MSE: {df_score_epoch:.4f}")
        print(f"Current Best Score: {self.best_eval_score:.4f}")
        
        # Log validation metrics
        log_data_valid = {
            "gsn/val_mesh_dice": eval_score_epoch,
            "gsn/val_df_mse": df_score_epoch
        }
        
        # Check for best model and save if improved
        if eval_score_epoch > self.best_eval_score:
            self._save_best_model(epoch, eval_score_epoch, cached_data)
        
        # Use epoch-based logging for sweep runs to avoid step conflicts
        if hasattr(self.super_params, 'is_sweep_run') and self.super_params.is_sweep_run:
            continuous_epoch = self.orchestrator.get_current_continuous_epoch() if self.orchestrator else epoch + 1
            wandb.log({**log_data_valid, "epoch": continuous_epoch}, commit=True)
        else:
            step = self.orchestrator.get_next_step() if self.orchestrator else epoch + 1
            wandb.log(log_data_valid, step=step, commit=True)
        
        return eval_score_epoch, df_score_epoch
    
    def test_unet(self, test_loader, modal):
        """
        Test UNet segmentation performance using provided test loader.
        
        Args:
            test_loader: Test data loader
            modal: Modality ('ct' or 'mr')
        
        Returns:
            Average dice score
        """
        # Deprecated: testing APIs moved to pipeline/testing.py
        raise NotImplementedError("Use pipeline.testing.MorphiNetTester for testing")

        # Choose components based on modality
        if modal == "ct":
            encoder = self.encoder_ct
            roi_size = self.super_params.crop_window_size
        elif modal == "mr":
            encoder = self.encoder_mr
            roi_size = self.super_params.crop_window_size[:2]
        else:
            raise ValueError(f"Invalid modality: {modal}. Use 'ct' or 'mr'")
        
        encoder.eval()
        
        # Initialize metrics
        dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        
        log_data_test = {}

        with torch.no_grad():
            for step, data in enumerate(test_loader):
                img, seg_true = (
                    data[f"{modal}_image"].to(DEVICE),
                    data[f"{modal}_label"].to(DEVICE),
                )

                # Filter out slices without labels for MR
                if modal == 'mr':
                    img, seg_true = self.preprocessor._filter_unlabeled_slices(img, seg_true)
                    
                    # Check if we have any data left after filtering
                    if img.shape[0] == 0:
                        print(f"Warning: No labeled slices found in test batch {step}, skipping")
                        continue
                    
                    # Adjust roi_size based on actual image dimensions
                    if len(img.shape) == 4:  # (B, C, H, W) - 2D slices
                        roi_size = roi_size  # Keep 2D roi_size
                    elif len(img.shape) == 5:  # (B, C, H, W, D) - 3D volume
                        roi_size = self.super_params.crop_window_size  # Use 3D roi_size
                
                # Run segmentation inference
                seg_pred = sliding_window_inference(
                    img, 
                    roi_size=roi_size, 
                    sw_batch_size=8, 
                    predictor=encoder,
                    overlap=0.5, 
                    mode="gaussian",
                )
                
                # Convert to one-hot for metric computation
                seg_pred_onehot = self.inference._convert_to_onehot(seg_pred, self.super_params.num_classes, is_prediction=True)
                seg_true_onehot = self.inference._convert_to_onehot(seg_true, self.super_params.num_classes, is_prediction=False)
                
                dice_metric(seg_pred_onehot, seg_true_onehot)

                if modal == 'ct' and len(img.shape) == 5:
                    slice_idx = img.shape[4] // 2
                    input_slice = img[0, 0, :, :, slice_idx]
                    gt_slice = seg_true[0, 0, :, :, slice_idx]
                    pred_slice = torch.argmax(seg_pred[0, :, :, :, slice_idx], dim=0)
                elif modal == 'mr' and len(img.shape) == 4 and img.shape[0] > 0:
                    slice_idx = img.shape[0] // (4 if self.super_params.test_dataset.lower() == 'cap' else 2)
                    input_slice = img[slice_idx, 0, :, :]
                    gt_slice = seg_true[slice_idx, 0, :, :]
                    pred_slice = torch.argmax(seg_pred[slice_idx, :, :, :], dim=0)
                else:
                    input_slice, gt_slice, pred_slice = None, None, None

                if input_slice is not None:
                    input_viz = self._prepare_slice_for_wandb(input_slice, is_segmentation=False)
                    gt_viz = self._prepare_slice_for_wandb(gt_slice, is_segmentation=True, num_classes=self.super_params.num_classes)
                    pred_viz = self._prepare_slice_for_wandb(pred_slice, is_segmentation=True, num_classes=self.super_params.num_classes)
                    
                    log_data_test.update({
                        f"unet/test_{modal}_input_slice": wandb.Image(input_viz, caption=f"Test Input - {modal.upper()}"),
                        f"unet/test_{modal}_gt_slice": wandb.Image(gt_viz, caption=f"Test Ground Truth - {modal.upper()}"),
                        f"unet/test_{modal}_pred_slice": wandb.Image(pred_viz, caption=f"Test Prediction - {modal.upper()}"),
                    })
                    
                    # Use epoch-based logging for sweep runs to avoid step conflicts
                    if hasattr(self.super_params, 'is_sweep_run') and self.super_params.is_sweep_run:
                        # For testing, use a special test epoch value
                        wandb.log({**log_data_test, "epoch": 0}, commit=True)
                    else:
                        step = self.orchestrator.get_next_step() if self.orchestrator else 0
                        wandb.log(log_data_test, step=step, commit=True)
        
        # Calculate and return results
        dice_scores = dice_metric.aggregate()
        
        # Extract individual dice scores for cardiac structures
        if len(dice_scores) >= 3:
            lv_dice = dice_scores[0].item() if not torch.isnan(dice_scores[0]) else 0.0
            myo_dice = dice_scores[1].item() if not torch.isnan(dice_scores[1]) else 0.0
            rv_dice = dice_scores[2].item() if not torch.isnan(dice_scores[2]) else 0.0
            avg_dice = dice_scores.mean().item() if not torch.isnan(dice_scores.mean()) else 0.0
        else:
            lv_dice = myo_dice = rv_dice = avg_dice = 0.0
        
        print(f"{modal.upper()} Test Dice Scores - LV: {lv_dice:.4f}, MYO: {myo_dice:.4f}, RV: {rv_dice:.4f}")
        print(f"Average Dice Score: {avg_dice:.4f}")
        
        log_data_test.update({
            f"unet/test_{modal}_dice_lv": lv_dice,
            f"unet/test_{modal}_dice_myo": myo_dice,
            f"unet/test_{modal}_dice_rv": rv_dice,
            f"unet/test_{modal}_dice_avg": avg_dice
        })
        wandb.log(log_data_test, commit=True)

        return avg_dice
    
    def test_resnet(self, test_loader, modal):
        """
        Test ResNet performance using provided test loader.
        
        Args:
            test_loader: Test data loader
            modal: Modality ('ct' or 'mr')
        
        Returns:
            Average dice score for segmentation quality
        """
        # Deprecated: testing APIs moved to pipeline/testing.py
        raise NotImplementedError("Use pipeline.testing.MorphiNetTester for testing")
        
        # Choose components based on modality
        if modal == "ct":
            encoder = self.encoder_ct
            roi_size = self.super_params.crop_window_size
        elif modal == "mr":
            encoder = self.encoder_mr
            roi_size = self.super_params.crop_window_size[:2]
        else:
            raise ValueError(f"Invalid modality: {modal}. Use 'ct' or 'mr'")
        
        encoder.eval()
        self.decoder.eval()
        
        # Initialize dice metric
        dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        
        with torch.no_grad():
            for step, data in enumerate(test_loader):
                img, seg_true = (
                    data[f"{modal}_image"].to(DEVICE),
                    data[f"{modal}_label"].to(DEVICE),
                )
                
                # Filter out slices without labels for MR 
                if modal == 'mr':
                    img, seg_true = self.preprocessor._filter_unlabeled_slices(img, seg_true)
                    
                    # Check if we have any data left after filtering
                    if img.shape[0] == 0:
                        print(f"Warning: No labeled slices found in MR batch {step}, skipping")
                        continue
                    
                    # Adjust roi_size based on actual image dimensions 
                    if len(img.shape) == 4:  # (B, C, H, W) - 2D slices
                        roi_size = roi_size  # Keep 2D roi_size
                    elif len(img.shape) == 5:  # (B, C, D, H, W) - 3D volume
                        roi_size = self.super_params.crop_window_size  # Use 3D roi_size
                
                # Run inference through UNet
                seg_pred = sliding_window_inference(
                    img, 
                    roi_size=roi_size, 
                    sw_batch_size=8, 
                    predictor=encoder,
                    overlap=0.5, 
                    mode="gaussian",
                )
                
                # Process predictions through ResNet pipeline (unflatten handled internally)
                seg_pred_ds_decoder_size = self.preprocessor._memory_efficient_post_transform(
                    seg_pred, seg_true, modal, to_gpu=True, decoder_size=True)
                
                seg_pred_ds = self.preprocessor._memory_efficient_post_transform(
                    seg_pred, seg_true, modal, to_gpu=True, decoder_size=False)
                
                # Calculate mask for refinement
                binary_mask_pred = (torch.argmax(seg_pred_ds_decoder_size, dim=1, keepdim=True) == 0)
                dist_map_pred = (-distance_transform_edt(binary_mask_pred.squeeze(1)) + distance_transform_edt(~binary_mask_pred.squeeze(1))).unsqueeze(1)
                mask = torch.sigmoid(dist_map_pred * self.sigmoid_scale_factor + 1).detach()
                mask = mask * binary_mask_pred
                mask[mask < self.mask_threshold] = 0
                
                # Apply decoder (ResNet) with padding
                seg_pred_ds_padded, pad_info = self.inference._apply_resnet_padding(seg_pred_ds)
                
                resnet_output_padded = self.decoder(seg_pred_ds_padded)
                
                resnet_output = self.inference._remove_resnet_padding(resnet_output_padded, pad_info)
                
                # Combine predictions (ResNet refined segmentation)
                seg_pred_ds_refined = seg_pred_ds_decoder_size + mask * resnet_output
                
                # Generate downsampled ground truth at decoder size (pass full tensor, not items)
                seg_true_ds_decoder_size = self.preprocessor._generate_downsampled_gt(seg_true, modal, decoder_size=True)
                
                # Convert to one-hot for dice metric computation (ResNet refined vs downsampled ground truth)
                seg_pred_onehot = self.inference._convert_to_onehot(seg_pred_ds_refined, self.super_params.num_classes, is_prediction=True)
                seg_true_onehot = self.inference._convert_to_onehot(seg_true_ds_decoder_size, self.super_params.num_classes, is_prediction=False)
                
                # Compute dice metric (ResNet testing - segmentation quality only)
                dice_metric(seg_pred_onehot, seg_true_onehot)
        
        # Calculate dice scores
        dice_scores = dice_metric.aggregate()
        
        # Extract individual dice scores for cardiac structures
        if len(dice_scores) >= 3:
            lv_dice = dice_scores[0].item() if not torch.isnan(dice_scores[0]) else 0.0
            myo_dice = dice_scores[1].item() if not torch.isnan(dice_scores[1]) else 0.0
            rv_dice = dice_scores[2].item() if not torch.isnan(dice_scores[2]) else 0.0
            avg_dice = dice_scores.mean().item() if not torch.isnan(dice_scores.mean()) else 0.0
        else:
            lv_dice = myo_dice = rv_dice = avg_dice = 0.0
        
        print(f"{modal.upper()} Test Dice Scores (ResNet refined) - LV: {lv_dice:.4f}, MYO: {myo_dice:.4f}, RV: {rv_dice:.4f}")
        print(f"Average Dice Score: {avg_dice:.4f}")
        
        # Log ResNet test metrics (consistent with validate_resnet)  
        resnet_test_data = {
            f"resnet/test_{modal}_dice_lv": lv_dice,
            f"resnet/test_{modal}_dice_myo": myo_dice,
            f"resnet/test_{modal}_dice_rv": rv_dice,
            f"resnet/test_{modal}_dice_avg": avg_dice
        }
        
        # Use epoch-based logging for sweep runs to avoid step conflicts
        if hasattr(self.super_params, 'is_sweep_run') and self.super_params.is_sweep_run:
            # For testing, use a special test epoch value
            wandb.log({**resnet_test_data, "epoch": 0}, commit=True)
        else:
            step = self.orchestrator.get_next_step() if self.orchestrator else 0
            wandb.log(resnet_test_data, step=step, commit=True)
        
        return avg_dice
    
    def test_gsn(self, test_loader, modal):
        """
        Test full pipeline performance (UNet + ResNet + GSN) using provided test loader.
        
        Args:
            test_loader: Test data loader
            modal: Modality ('ct' or 'mr')
        
        Returns:
            Dictionary containing mesh dice score and distance field MSE
        """
        # Deprecated: testing APIs moved to pipeline/testing.py
        raise NotImplementedError("Use pipeline.testing.MorphiNetTester for testing")
        
        # Choose components based on modality
        if modal == "ct":
            encoder = self.encoder_ct
            roi_size = self.super_params.crop_window_size
        elif modal == "mr":
            encoder = self.encoder_mr
            roi_size = self.super_params.crop_window_size[:2]
        else:
            raise ValueError(f"Invalid modality: {modal}. Use 'ct' or 'mr'")
        
        encoder.eval()
        self.decoder.eval()
        self.GSN.eval()
        
        # Initialize metrics
        # UNet and ResNet Dice (both CT/MR), plus DF MSE and Mesh Dice
        unet_dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        resnet_dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        df_metric_batch = MSEMetric(reduction="mean_batch")
        msh_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
        
        # Export directory setup
        dataset_name = getattr(self.orchestrator, 'dataset', 'unknown') or 'unknown'
        output_root = getattr(self.super_params, 'output_root', get_path_default('MORPHINET_OUTPUT_ROOT'))
        export_dir = os.path.join(output_root, dataset_name, 'MorphiNet', 'myo', 'f0')
        os.makedirs(export_dir, exist_ok=True)
        # Ablation directory (intermediate artifacts) - only for CAP/SCOTHEART
        export_ablation = dataset_name in {"cap", "scotheart"}
        ablation_dir = os.path.join(output_root, 'ablation', 'MorphiNet', 'myo', 'f0')
        if export_ablation:
            os.makedirs(ablation_dir, exist_ok=True)

        with torch.no_grad():
            for step, data in enumerate(test_loader):
                img, seg_true, df_true = (
                    data[f"{modal}_image"].to(DEVICE),
                    data[f"{modal}_label"].to(DEVICE),
                    data[f"{modal}_df"].as_tensor().to(DEVICE),
                )
                
                # Filter out slices without labels for MR 
                if modal == 'mr':
                    img, seg_true = self.preprocessor._filter_unlabeled_slices(img, seg_true)
                    
                    # Check if we have any data left after filtering
                    if img.shape[0] == 0:
                        print(f"Warning: No labeled slices found in MR batch {step}, skipping")
                        continue
                    
                    # Adjust roi_size based on actual image dimensions 
                    if len(img.shape) == 4:  # (B, C, H, W) - 2D slices
                        roi_size = roi_size  # Keep 2D roi_size
                    elif len(img.shape) == 5:  # (B, C, D, H, W) - 3D volume
                        roi_size = self.super_params.crop_window_size  # Use 3D roi_size
                
                # Generate downsampled ground truth at decoder size (unflatten handled internally)
                seg_true_ds = self.preprocessor._generate_downsampled_gt(seg_true, modal, decoder_size=True)
                
                # Run inference through full pipeline
                seg_pred = sliding_window_inference(
                    img, 
                    roi_size=roi_size, 
                    sw_batch_size=8, 
                    predictor=encoder,
                    overlap=0.5, 
                    mode="gaussian",
                )
                
                # --- UNet Dice (pre-ResNet) ---
                seg_pred_onehot_unet = self.inference._convert_to_onehot(seg_pred, self.super_params.num_classes, is_prediction=True)
                seg_true_onehot_unet = self.inference._convert_to_onehot(seg_true, self.super_params.num_classes, is_prediction=False)
                unet_dice_metric(seg_pred_onehot_unet, seg_true_onehot_unet)

                # --- UNet mesh export (MYO) ---
                try:
                    # Create hard label map from UNet prediction
                    if seg_pred.ndim == 5:
                        unet_labels = torch.argmax(seg_pred, dim=1, keepdim=True)  # [B,1,H,W,D]
                    else:
                        unet_labels = torch.argmax(seg_pred, dim=1, keepdim=True)
                    # Extract MYO surface
                    unet_mesh_list = self.mesh_ops.surface_extractor(unet_labels, labels=2)
                    if export_ablation and len(unet_mesh_list) > 0:
                        unet_meshes = unet_mesh_list[0]  # Meshes with batch B
                        verts_b = unet_meshes.verts_padded()
                        faces_b = unet_meshes.faces_padded()
                        case_ids = data.get(f"{modal}_case_id", [])
                        if isinstance(case_ids, str):
                            case_ids = [case_ids]
                        for b_idx in range(verts_b.shape[0]):
                            case_id = case_ids[b_idx] if b_idx < len(case_ids) else f"case_{step:04d}_{b_idx}"
                            save_obj(os.path.join(ablation_dir, f"{case_id}_unet_myo.obj"),
                                     verts_b[b_idx].to(torch.float32), faces_b[b_idx].to(torch.int32))
                except Exception as e:
                    print(f"Warning: UNet mesh export failed: {e}")

                # Process predictions through ResNet pipeline (unflatten handled internally)
                seg_pred_ds_decoder_size = self.preprocessor._memory_efficient_post_transform(
                    seg_pred, seg_true, modal, to_gpu=True, decoder_size=True)
                
                seg_pred_ds = self.preprocessor._memory_efficient_post_transform(
                    seg_pred, seg_true, modal, to_gpu=True, decoder_size=False)
                
                # Calculate mask for refinement
                binary_mask_pred = (torch.argmax(seg_pred_ds_decoder_size, dim=1, keepdim=True) == 0)
                dist_map_pred = (-distance_transform_edt(binary_mask_pred.squeeze(1)) + distance_transform_edt(~binary_mask_pred.squeeze(1))).unsqueeze(1)
                mask = torch.sigmoid(dist_map_pred * self.sigmoid_scale_factor + 1).detach()
                mask = mask * binary_mask_pred
                mask[mask < self.mask_threshold] = 0
                
                # Apply decoder (ResNet) with padding
                seg_pred_ds_padded, pad_info = self.inference._apply_resnet_padding(seg_pred_ds)
                resnet_output_padded = self.decoder(seg_pred_ds_padded)
                resnet_output = self.inference._remove_resnet_padding(resnet_output_padded, pad_info)
                
                # Combine predictions
                seg_pred_ds = seg_pred_ds_decoder_size + mask * resnet_output
                seg_pred_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ds])

                # --- ResNet Dice (post-ResNet, decoder size) ---
                seg_true_ds_decoder_size = self.preprocessor._generate_downsampled_gt(seg_true, modal, decoder_size=True)
                seg_pred_onehot_resnet = self.inference._convert_to_onehot(seg_pred_ds, self.super_params.num_classes, is_prediction=True)
                seg_true_onehot_resnet = self.inference._convert_to_onehot(seg_true_ds_decoder_size, self.super_params.num_classes, is_prediction=False)
                resnet_dice_metric(seg_pred_onehot_resnet, seg_true_onehot_resnet)

                # --- ResNet mesh export (MYO) ---
                try:
                    resnet_labels = torch.argmax(seg_pred_ds, dim=1, keepdim=True)
                    resnet_mesh_list = self.mesh_ops.surface_extractor(resnet_labels, labels=2)
                    if export_ablation and len(resnet_mesh_list) > 0:
                        resnet_meshes = resnet_mesh_list[0]
                        verts_b = resnet_meshes.verts_padded()
                        faces_b = resnet_meshes.faces_padded()
                        case_ids = data.get(f"{modal}_case_id", [])
                        if isinstance(case_ids, str):
                            case_ids = [case_ids]
                        for b_idx in range(verts_b.shape[0]):
                            case_id = case_ids[b_idx] if b_idx < len(case_ids) else f"case_{step:04d}_{b_idx}"
                            save_obj(os.path.join(ablation_dir, f"{case_id}_resnet_myo.obj"),
                                     verts_b[b_idx].to(torch.float32), faces_b[b_idx].to(torch.int32))
                except Exception as e:
                    print(f"Warning: ResNet mesh export failed: {e}")
                
                # Generate distance fields
                foreground = seg_pred_ds > 0
                lv = (seg_pred_ds == 1)
                rv = (seg_pred_ds == 3)
                myo = (seg_pred_ds == 2)
                df_pred = torch.stack([
                    distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                    for i in [foreground, lv, rv, myo]], dim=1)
                
                df_metric_batch(df_pred, df_true)
                
                # Generate mesh predictions (warped template and GSN predicted are in NDC and MYO)
                template_mesh = self.mesh_ops.warp_template_mesh(df_pred)
                # Export warped template mesh per case (ablation)
                try:
                    if export_ablation:
                        verts_b = template_mesh.verts_padded()
                    faces_b = template_mesh.faces_padded()
                    case_ids = data.get(f"{modal}_case_id", [])
                    if isinstance(case_ids, str):
                        case_ids = [case_ids]
                    for b_idx in range(verts_b.shape[0]):
                        case_id = case_ids[b_idx] if b_idx < len(case_ids) else f"case_{step:04d}_{b_idx}"
                        save_obj(os.path.join(ablation_dir, f"{case_id}_template_warped_myo.obj"),
                                 verts_b[b_idx].to(torch.float32), faces_b[b_idx].to(torch.int32))
                except Exception as e:
                    print(f"Warning: Warped template mesh export failed: {e}")

                # GSN multi-level outputs
                all_levels = self.GSN(template_mesh, self.mesh_ops.subdivided_faces.faces_levels, df_pred, self.mesh_ops.subdivided_faces.labels_levels)
                subdiv_mesh = all_levels[-1]
                # Export each level per case (ablation)
                try:
                    if export_ablation:
                        case_ids = data.get(f"{modal}_case_id", [])
                    if isinstance(case_ids, str):
                        case_ids = [case_ids]
                    for lvl_idx, lvl_mesh in enumerate(all_levels):
                        verts_b = lvl_mesh.verts_padded()
                        faces_b = lvl_mesh.faces_padded()
                        for b_idx in range(verts_b.shape[0]):
                            case_id = case_ids[b_idx] if b_idx < len(case_ids) else f"case_{step:04d}_{b_idx}"
                            save_obj(os.path.join(ablation_dir, f"{case_id}_gsn_l{lvl_idx+1}_myo.obj"),
                                     verts_b[b_idx].to(torch.float32), faces_b[b_idx].to(torch.int32))
                except Exception as e:
                    print(f"Warning: GSN multi-level export failed: {e}")
                
                # Rasterize mesh for comparison
                voxeld_mesh = torch.cat([
                    self.rasterizer(
                        pred_mesh.verts_padded(), pred_mesh.faces_padded())
                    for pred_mesh in subdiv_mesh
                    ], dim=0)
                
                # Create dilated mask for ground truth
                dilated_mask = torch.zeros_like(seg_true_ds)
                for batch_idx in range(seg_true_ds.shape[0]):
                    seg_np = seg_true_ds[batch_idx, 0].cpu().numpy().astype(bool)
                    dilated_np = binary_dilation(seg_np, iterations=2)
                    dilated_mask[batch_idx, 0] = torch.from_numpy(dilated_np.astype(np.float32)).to(seg_true_ds.device)
                
                # Apply mask and compute mesh metric
                voxeld_mesh_masked = voxeld_mesh * dilated_mask
                seg_true_ds = (seg_true_ds == 2).to(torch.float32)
                msh_metric_batch(voxeld_mesh_masked, seg_true_ds)

                # --- Export predicted mesh per case ---
                case_ids = data.get(f"{modal}_case_id", [])
                if isinstance(case_ids, str):
                    case_ids = [case_ids]
                # Handle batched Meshes
                # subdiv_mesh is a list of Meshes per GCN layer; we used last layer: Meshes batch size = B
                verts_b = subdiv_mesh.verts_padded()  # (B, N, 3)
                faces_b = subdiv_mesh.faces_padded()  # (B, M, 3)
                B = verts_b.shape[0]
                for b_idx in range(B):
                    case_id = case_ids[b_idx] if b_idx < len(case_ids) else f"case_{step:04d}_{b_idx}"
                    V = verts_b[b_idx].to(torch.float32)
                    F = faces_b[b_idx].to(torch.int32)
                    out_path = os.path.join(export_dir, f"{case_id}_myo.obj")
                    # Mesh is already in NDC and MYO; save as OBJ
                    try:
                        save_obj(out_path, V, F)
                    except Exception as e:
                        print(f"Warning: Failed to save OBJ for {case_id}: {e}")
        
        # Calculate metrics
        mesh_dice_score = msh_metric_batch.aggregate().mean()
        df_mse_score = df_metric_batch.aggregate().mean()
        # Aggregate UNet/ResNet Dice
        unet_scores = unet_dice_metric.aggregate()
        resnet_scores = resnet_dice_metric.aggregate()
        def _extract_avg(dice_tensor):
            return dice_tensor.mean().item() if dice_tensor is not None and not torch.isnan(dice_tensor.mean()) else 0.0
        unet_avg = _extract_avg(unet_scores)
        resnet_avg = _extract_avg(resnet_scores)
        
        print(f"{modal.upper()} Test Results:")
        print(f"  UNet Dice (avg): {unet_avg:.4f}")
        print(f"  ResNet Dice (avg): {resnet_avg:.4f}")
        print(f"  Mesh Dice Score: {mesh_dice_score:.4f}")
        print(f"  Distance Field MSE: {df_mse_score:.4f}")
        
        # Log scalar summaries
        wandb.log({
            f"unet/test_{modal}_dice_avg": unet_avg,
            f"resnet/test_{modal}_dice_avg": resnet_avg,
            f"gsn/test_{modal}_mesh_dice": mesh_dice_score.item(),
            f"gsn/test_{modal}_df_mse": df_mse_score.item(),
        }, commit=True)

        return {
            'mesh_dice': mesh_dice_score.item(),
            'df_mse': df_mse_score.item()
        }