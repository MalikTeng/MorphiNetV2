import os
import torch
import numpy as np
import wandb
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MSEMetric
from monai.transforms.utils import distance_transform_edt
from monai.transforms import AsDiscrete
from scipy.ndimage import binary_dilation
from utils.tools import draw_plotly


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MorphiNetValidator:
    """Handles validation for MorphiNet full pipeline."""
    
    def __init__(self, super_params, models, dataloaders, preprocessor, mesh_ops, inference, ckpt_dir):
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
        """
        self.super_params = super_params
        self.ckpt_dir = ckpt_dir
        
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
        self.pred_transform = AsDiscrete(argmax=True, to_onehot=self.super_params.num_classes)
        
        # Sigmoid parameters
        self.sigmoid_scale_factor = super_params.sigmoid_scale_factor
        self.mask_threshold = super_params.mask_threshold
        
        # Rasterizer
        if hasattr(mesh_ops, 'rasterizer'):
            self.rasterizer = mesh_ops.rasterizer
    
    def validate(self, epoch, save_on):
        """
        Perform full network validation.
        
        Args:
            epoch: Current epoch number
            save_on: Validation dataset ('sct' for CT, 'cap' for MR)
        
        Returns:
            Validation metrics (eval_score_epoch, df_score_epoch)
        """
        print(f"\n--- FULL NETWORK VALIDATION ---")
        print(f"Phase: Full Pipeline (UNet + ResNet + GSN), Modal: {save_on.upper()}")
        
        self.decoder.eval()
        self.GSN.eval()
        
        # Save model checkpoints
        self._save_model_checkpoints(epoch)
        
        # Choose the validation loader and encoder
        if save_on == "ct":
            modal = "ct"
            encoder = self.encoder_ct
            valid_loader = self.dataloader_manager.ct_valid_loader
            roi_size = self.super_params.crop_window_size
        elif save_on == "mr":
            modal = "mr"
            encoder = self.encoder_mr
            valid_loader = self.dataloader_manager.mr_valid_loader
            roi_size = self.super_params.crop_window_size[:2]
        else:
            raise ValueError(f"Invalid validation modality: {save_on}. Use 'ct' or 'mr'")
        
        encoder.eval()
        
        # Initialize metrics
        df_metric_batch_decoder = MSEMetric(reduction="mean_batch")
        msh_metric_batch_decoder = DiceMetric(include_background=False, reduction="mean_batch")
        
        cached_data = dict()
        choice_case = np.random.choice(len(valid_loader), 1)[0]
        
        with torch.no_grad():
            for step, data in enumerate(valid_loader):
                img, seg_true, df_true = (
                    data[f"{modal}_image"].to(DEVICE),
                    data[f"{modal}_label"].to(DEVICE),
                    data[f"{modal}_df"].as_tensor().to(DEVICE),
                )
                num_items_for_unflatten = 1 if modal == 'ct' else 2
                
                # Apply unflatten only for MR
                if modal == 'mr':
                    seg_true = seg_true.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
                
                # Generate downsampled ground truth at decoder size
                seg_true_ds = torch.stack([
                    self.preprocessor._generate_downsampled_gt(seg_true_item, modal, decoder_size=True)
                    for seg_true_item in seg_true
                ])
                
                # Run inference through full pipeline
                seg_pred = sliding_window_inference(
                    img, 
                    roi_size=roi_size, 
                    sw_batch_size=8, 
                    predictor=encoder,
                    overlap=0.5, 
                    mode="gaussian",
                    device=torch.device('cpu'),
                    buffer_steps=4,
                    buffer_dim=-1,
                )
                
                # Apply unflatten only for MR
                if modal == 'mr':
                    seg_pred = seg_pred.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
                
                # Process predictions through ResNet pipeline
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
                
                # Apply decoder and combine predictions
                decoder_output = self.decoder(seg_pred_ds)
                seg_pred_ds = seg_pred_ds_decoder_size + mask * decoder_output
                seg_pred_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ds])
                
                # Generate distance fields
                foreground = seg_pred_ds > 0
                lv = (seg_pred_ds == 1)
                rv = (seg_pred_ds == 3)
                myo = (seg_pred_ds == 2)
                df_pred = torch.stack([
                    distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                    for i in [foreground, lv, rv, myo]], dim=1)
                
                df_metric_batch_decoder(df_pred, df_true)
                
                # Generate mesh predictions
                template_mesh = self.mesh_ops.warp_template_mesh(df_pred)
                template_mesh = template_mesh.update_padded(template_mesh.verts_padded().to(torch.float32))
                
                subdiv_mesh = self.GSN(template_mesh, self.mesh_ops.subdivided_faces.faces_levels, df_pred, self.mesh_ops.subdivided_faces.labels_levels)[-1]
                
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
                msh_metric_batch_decoder(voxeld_mesh_masked, seg_true_ds)
                
                # Cache data for visualization
                if step == choice_case:
                    cached_data = {
                        "df_true": df_true[0].cpu(),
                        "df_pred": df_pred[0].cpu(),
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
            "validation/mesh_dice": eval_score_epoch,
            "validation/df_mse": df_score_epoch
        }
        
        # Check for best model and save if improved
        if eval_score_epoch > self.best_eval_score:
            self._save_best_model(epoch, eval_score_epoch, cached_data)
        
        wandb.log(log_data_valid, step=epoch + 1, commit=True)
        
        return eval_score_epoch, df_score_epoch
    
    def _save_model_checkpoints(self, epoch):
        """Save model checkpoints for current epoch."""
        ckpt_weight_path = os.path.join(self.ckpt_dir, "trained_weights")
        os.makedirs(ckpt_weight_path, exist_ok=True)
        
        torch.save(self.encoder_ct.state_dict(), os.path.join(ckpt_weight_path, f"{epoch + 1}_UNet_CT.pth"))
        torch.save(self.encoder_mr.state_dict(), os.path.join(ckpt_weight_path, f"{epoch + 1}_UNet_MR.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(ckpt_weight_path, f"{epoch + 1}_ResNet.pth"))
        torch.save(self.GSN.state_dict(), os.path.join(ckpt_weight_path, f"{epoch + 1}_GSN.pth"))
        
        # Save subdivision faces
        for level, faces in enumerate(self.mesh_ops.subdivided_faces.faces_levels):
            torch.save(faces, os.path.join(ckpt_weight_path, f"{epoch+1}_subdivided_faces_l{level}.pth"))
    
    def _save_best_model(self, epoch, eval_score_epoch, cached_data):
        """Save best model and generate visualizations."""
        ckpt_weight_path = os.path.join(self.ckpt_dir, "trained_weights")
        
        # Save best model weights
        torch.save(self.encoder_ct.state_dict(), os.path.join(ckpt_weight_path, f"best_UNet_CT.pth"))
        torch.save(self.encoder_mr.state_dict(), os.path.join(ckpt_weight_path, f"best_UNet_MR.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(ckpt_weight_path, f"best_ResNet.pth"))
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
    
    def _generate_visualizations(self, cached_data):
        """Generate and save visualizations."""
        visualization_dir = f"{self.ckpt_dir}/visualizations"
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Create visualizations
        try:
            draw_plotly(
                seg_true=cached_data["seg_true_ds"], 
                mesh_pred=cached_data["subdiv_mesh"],
                save_html=True,
                save_dir=visualization_dir,
                filename="seg_true_ds_vs_mesh_pred.html",
            )
            
            draw_plotly(
                seg_true=cached_data["seg_true_ds"], 
                mesh_pred=cached_data["template_mesh"],
                save_html=True,
                save_dir=visualization_dir,
                filename="seg_true_ds_vs_template_mesh.html",
            )
            
            print("Visualizations saved successfully!")
            
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
    
    def validate_segmentation(self, epoch, save_on):
        """
        Validate segmentation performance only (UNet phase).
        
        Args:
            epoch: Current epoch number
            save_on: Dataset to validate on ('sct' for CT, 'cap' for MR)
        
        Returns:
            Validation dice score
        """
        print(f"\n--- SEGMENTATION VALIDATION ---")
        print(f"Phase: UNet Only, Modal: {save_on.upper()}")
        
        # Choose validation components
        if save_on == "ct":
            modal = "ct"
            encoder = self.encoder_ct
            valid_loader = self.dataloader_manager.ct_valid_loader
            roi_size = self.super_params.crop_window_size
        elif save_on == "mr":
            modal = "mr"
            encoder = self.encoder_mr
            valid_loader = self.dataloader_manager.mr_valid_loader
            roi_size = self.super_params.crop_window_size[:2]
        else:
            raise ValueError(f"Invalid validation modality: {save_on}. Use 'ct' or 'mr'")
        
        encoder.eval()
        
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
                    
                    # Debug: Check data dimensions after filtering
                    print(f"Debug: After filtering - img shape: {img.shape}, roi_size: {roi_size}")
                    
                    # Adjust roi_size based on actual image dimensions
                    if len(img.shape) == 4:  # (B, C, H, W) - 2D slices
                        roi_size = roi_size  # Keep 2D roi_size
                    elif len(img.shape) == 5:  # (B, C, D, H, W) - 3D volume
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
                
                # No unflatten needed for segmentation validation (data already filtered)
                
                # Convert to one-hot for metric computation
                seg_pred_onehot = self.inference._convert_to_onehot(seg_pred, self.super_params.num_classes, is_prediction=True)
                seg_true_onehot = self.inference._convert_to_onehot(seg_true, self.super_params.num_classes, is_prediction=False)
                
                dice_metric(seg_pred_onehot, seg_true_onehot)
        
        # Calculate and log results
        dice_score = dice_metric.aggregate().mean()
        
        print(f"Segmentation Dice Score: {dice_score:.4f}")
        
        # Log segmentation validation metrics
        wandb.log({
            f"segmentation_validation/{modal}_dice": dice_score
        }, step=epoch + 1, commit=True)
        
        return dice_score
    
    def validate_resnet(self, epoch, save_on):
        """
        Validate ResNet performance (UNet + ResNet pipeline without GSN).
        
        Args:
            epoch: Current epoch number
            save_on: Dataset to validate on ('ct' or 'mr')
        
        Returns:
            Distance field MSE score
        """
        print(f"\n--- RESNET VALIDATION ---")
        print(f"Phase: UNet + ResNet Only, Modal: {save_on.upper()}")
        
        self.decoder.eval()
        
        # Choose validation components
        if save_on == "ct":
            modal = "ct"
            encoder = self.encoder_ct
            valid_loader = self.dataloader_manager.ct_valid_loader
            roi_size = self.super_params.crop_window_size
        elif save_on == "mr":
            modal = "mr"
            encoder = self.encoder_mr
            valid_loader = self.dataloader_manager.mr_valid_loader
            roi_size = self.super_params.crop_window_size[:2]
        else:
            raise ValueError(f"Invalid validation modality: {save_on}. Use 'ct' or 'mr'")
        
        encoder.eval()
        
        # Initialize metrics
        df_metric_batch_decoder = MSEMetric(reduction="mean_batch")
        
        with torch.no_grad():
            for step, data in enumerate(valid_loader):
                img, seg_true, df_true = (
                    data[f"{modal}_image"].to(DEVICE),
                    data[f"{modal}_label"].to(DEVICE),
                    data[f"{modal}_df"].as_tensor().to(DEVICE),
                )
                num_items_for_unflatten = 1 if modal == 'ct' else 2
                
                # Apply unflatten only for MR
                if modal == 'mr':
                    seg_true = seg_true.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
                
                # Run inference through UNet
                seg_pred = sliding_window_inference(
                    img, 
                    roi_size=roi_size, 
                    sw_batch_size=8, 
                    predictor=encoder,
                    overlap=0.5, 
                    mode="gaussian",
                    device=torch.device('cpu'),
                    buffer_steps=4,
                    buffer_dim=-1,
                )
                
                # Apply unflatten only for MR
                if modal == 'mr':
                    seg_pred = seg_pred.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
                
                # Process predictions through ResNet pipeline
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
                
                # Apply decoder (ResNet) and combine predictions
                decoder_output = self.decoder(seg_pred_ds)
                seg_pred_ds = seg_pred_ds_decoder_size + mask * decoder_output
                seg_pred_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ds])
                
                # Generate distance fields from ResNet output
                foreground = seg_pred_ds > 0
                lv = (seg_pred_ds == 1)
                rv = (seg_pred_ds == 3)
                myo = (seg_pred_ds == 2)
                df_pred = torch.stack([
                    distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                    for i in [foreground, lv, rv, myo]], dim=1)
                
                # Compute distance field metric (ResNet validation)
                df_metric_batch_decoder(df_pred, df_true)
        
        # Calculate metrics
        df_score_epoch = df_metric_batch_decoder.aggregate().mean()
        
        print(f"Distance Field MSE: {df_score_epoch:.4f}")
        
        # Log ResNet validation metrics
        wandb.log({
            f"resnet_validation/{modal}_df_mse": df_score_epoch
        }, step=epoch + 1, commit=True)
        
        return df_score_epoch