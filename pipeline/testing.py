"""
MorphiNet Testing Pipeline Module (Rewritten)

This module contains all testing logic for MorphiNet models using the same
dataloader and transform infrastructure as training for consistency.
"""

import os
import time
import torch
import numpy as np
import wandb
from monai.inferers import sliding_window_inference
from monai.transforms.utils import distance_transform_edt
from monai.transforms import AsDiscrete
# Note: Histogram matching imports removed - functionality moved to HistogramMatchd transform

# Dataset registry mapping dataset identifiers to metadata
DATASET_REGISTRY = {
    "acdc": {
        "modality": "mr",
        "json": "./dataset/dataset_task21_f0.json",
        "data_dir": "/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset021_ACDC",
    },
    "mmwhs": {
        "modality": "ct",
        "json": "./dataset/dataset_task22_f0.json",
        "data_dir": "/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset022_MMWHS_CT",
    },
    "cap": {
        "modality": "mr",
        "json": "./dataset/dataset_task11_f0.json",
        "data_dir": "/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX",
    },
    "scotheart": {
        "modality": "ct",
        "json": "./dataset/dataset_task20_f0.json",
        "data_dir": "/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset020_SCOTHEART",
    },
}


def _prepare_slice_for_wandb(slice_tensor, is_segmentation, num_classes=None):
    """
    Prepares a 2D tensor slice for logging to Weights & Biases as an image.
    Handles normalization for input images and scaling for segmentation masks.
    (Following the same pattern as training/trainer.py)
    """
    if hasattr(slice_tensor, 'cpu'):
        slice_np = slice_tensor.cpu().numpy().astype(np.float32)
    else:
        slice_np = np.array(slice_tensor).astype(np.float32)
    
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


def _log_wandb_images(image_tensor, true_labels, pred_labels, caption_prefix, modal, dataset, num_classes=4, step=None, tester=None):
    """
    Logs visualization of image, ground truth, and prediction to WandB.
    For both MR and CT data, also logs intensity histogram and intensity step.
    Following the same pattern as training/trainer.py
    
    Args:
        tester: MorphiNetTester instance for histogram matching operations
    """
    try:
        # Select appropriate slice for visualization
        if modal == 'ct' and len(image_tensor.shape) == 5:  # [N,C,H,W,D] format
            # For CT data, use middle slice
            slice_idx = image_tensor.shape[4] // 2
            input_img_slice = image_tensor[0, 0, :, :, slice_idx]
            gt_slice = true_labels[0, 0, :, :, slice_idx] if len(true_labels.shape) == 5 else true_labels[0, :, :, slice_idx]
            pred_slice = torch.argmax(pred_labels[0, :, :, :, slice_idx], dim=0) if len(pred_labels.shape) == 5 else torch.argmax(pred_labels[0, :, :, :], dim=0)
        elif modal == 'mr' and len(image_tensor.shape) == 4:  # [N*D,C,H,W] format
            # For MR data, use middle slice from flattened data
            denominator = 4 if dataset == 'cap' else 2
            slice_idx = image_tensor.shape[0] // denominator
            input_img_slice = image_tensor[slice_idx, 0, :, :] 
            gt_slice = true_labels[slice_idx, :, :] if len(true_labels.shape) == 3 else true_labels[slice_idx, 0, :, :]
            pred_slice = torch.argmax(pred_labels[slice_idx, :, :, :], dim=0)
        else:
            # Fallback for other formats
            input_img_slice = image_tensor.squeeze()
            gt_slice = true_labels.squeeze() 
            pred_slice = torch.argmax(pred_labels, dim=1).squeeze()
            
            # If still multi-dimensional, take first slice
            if input_img_slice.ndim > 2:
                input_img_slice = input_img_slice[0]
            if gt_slice.ndim > 2:
                gt_slice = gt_slice[0]
            if pred_slice.ndim > 2:
                pred_slice = pred_slice[0]
        
        # Note: Histogram matching logic has been moved to HistogramMatchd transform
        # and runs automatically during data preprocessing. No manual handling needed here.
        # Convert to numpy arrays for wandb logging using _prepare_slice_for_wandb
        input_img_viz = _prepare_slice_for_wandb(input_img_slice, is_segmentation=False)
        gt_slice_viz = _prepare_slice_for_wandb(gt_slice, is_segmentation=True, num_classes=num_classes)
        pred_slice_viz = _prepare_slice_for_wandb(pred_slice, is_segmentation=True, num_classes=num_classes)
        
        # Log individual images following training pattern
        log_data = {
            f"{caption_prefix}_input": wandb.Image(input_img_viz, caption=f"{modal.upper()} Input"),
            f"{caption_prefix}_gt": wandb.Image(gt_slice_viz, caption=f"{modal.upper()} Ground Truth"),
            f"{caption_prefix}_pred": wandb.Image(pred_slice_viz, caption=f"{modal.upper()} Prediction"),
        }
        
        # Intensity analytics (now shared by MR and CT)
        if modal in ("mr", "ct"):
            def _hist_and_step(slice_array: np.ndarray):
                """Return wandb.Histogram and the Δ between two lowest unique intensities."""
                flat = slice_array.flatten()
                histogram = wandb.Histogram(flat, num_bins=50)
                uniq = np.unique(flat)
                step = float(uniq[1] - uniq[0]) if len(uniq) > 1 else 0.0
                return histogram, step

            # Convert to numpy once; works for both tensor and ndarray input
            slice_np = input_img_slice.cpu().numpy() if torch.is_tensor(input_img_slice) else np.asarray(input_img_slice)
            hist, intensity_step = _hist_and_step(slice_np)

            log_data[f"{caption_prefix}_input_histogram"] = hist
            log_data[f"{caption_prefix}_intensity_step"] = intensity_step

        wandb.log(log_data, step=step)
        
    except Exception as e:
        print(f"      Warning: Could not log visualization: {e}")


class MorphiNetTester:
    """
    MorphiNet testing class that mirrors the training workflow.
    
    Uses the same dataloaders, transforms, and DiceMetric as training
    but with models in eval mode and torch.no_grad() context.
    """
    
    def __init__(self, pipeline, super_params):
        """
        Initialize the tester.
        
        Args:
            pipeline: MorphiNetPipeline instance with loaded models
            super_params: Configuration parameters
        """
        self.pipeline = pipeline
        self.params = super_params
        self.metrics = pipeline.metrics  # Use same metrics as training
        self.registry = DATASET_REGISTRY
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Note: Histogram matching initialization removed - handled automatically by HistogramMatchd transform
    
    def test(self):
        """
        Run comprehensive testing on specified datasets and phases.
        
        Returns:
            Dictionary containing test results
        """
        results = {
            'unet_results': {},
            'resnet_results': {},
            'datasets_tested': [],
            'phases_tested': []
        }
        
        # Resolve datasets and phases to test
        datasets_to_test = self._resolve_datasets()
        phases_to_test = self._resolve_phases()
        
        results['datasets_tested'] = datasets_to_test
        results['phases_tested'] = phases_to_test
        
        print(f"Testing {len(datasets_to_test)} dataset(s) and {len(phases_to_test)} phase(s)")
        
        # Test each dataset/phase combination
        for dataset in datasets_to_test:
            for phase in phases_to_test:
                print(f"\n--- Testing {phase.upper()} on {dataset.upper()} ---")
                
                # Configure dataset parameters and build dataloader
                dataloader, modal = self._build_test_dataloader(dataset)
                
                if dataloader is None:
                    print(f"Warning: Could not build dataloader for {dataset}")
                    continue
                
                # Set models to eval mode
                self._set_models_eval(modal, phase)
                
                # Run testing loop with torch.no_grad()
                with torch.no_grad():
                    test_metrics = self._run_test_loop(dataloader, modal, dataset, phase)
                
                # Store results
                if phase == "unet":
                    results['unet_results'][dataset] = test_metrics
                else:
                    results['resnet_results'][dataset] = test_metrics
        
        # Note: Histogram matching finalization removed - handled automatically by HistogramMatchd transform
        
        # Log results to WandB
        self._log_wandb_results(results)
        
        return results
    
    def _resolve_datasets(self):
        """Resolve which datasets to test based on parameters."""
        if self.params.test_dataset == "both":
            return ["acdc", "mmwhs"]  # Legacy meaning
        else:
            dataset = self.params.test_dataset
            if dataset not in self.registry:
                print(f"Warning: Unsupported dataset {dataset}")
                return []
            return [dataset]
    
    def _resolve_phases(self):
        """Resolve which phases to test based on parameters."""
        if self.params.test_phase == "both":
            return ["unet", "resnet"]
        else:
            return [self.params.test_phase]
    
    def _build_test_dataloader(self, dataset):
        """
        Build test dataloader for a specific dataset using the orchestrator's infrastructure.
        
        Args:
            dataset: Dataset name
            
        Returns:
            Tuple of (dataloader, modality)
        """
        # Get dataset metadata
        dataset_info = self.registry[dataset]
        modal = dataset_info["modality"]
        
        # Configure dataset parameters
        if modal == "mr":
            self.params.mr_data_dir = dataset_info["data_dir"]
            self.params.mr_json_dir = dataset_info["json"]
        else:
            self.params.ct_data_dir = dataset_info["data_dir"]
            self.params.ct_json_dir = dataset_info["json"]
        
        # Use orchestrator's dataloader manager to build test loader
        try:
            # Configure validation modality to ensure correct test data is loaded
            original_validation_modality = self.params.validation_modality
            self.params.validation_modality = modal
            
            # Prepare test dataloaders using the existing infrastructure
            self.pipeline.orchestrator.prepare_dataloaders(
                data_types=["test"], 
                training_phase="unet",  # Use unet phase for simplicity
                validation_phase="validation",
                include_test=True
            )
            
            # Get the appropriate test loader
            if modal == "mr":
                dataloader = self.pipeline.orchestrator.dataloader_manager.mr_test_loader
            else:
                dataloader = self.pipeline.orchestrator.dataloader_manager.ct_test_loader
            
            # Restore original validation modality
            self.params.validation_modality = original_validation_modality
            
            if dataloader is None:
                print(f"Warning: No test data found for {dataset} ({modal})")
                return None, modal
            
            print(f"Built test dataloader for {dataset} with {len(dataloader)} batches")
            return dataloader, modal
            
        except Exception as e:
            print(f"Error building dataloader for {dataset}: {e}")
            return None, modal
    
    def _set_models_eval(self, modal, phase):
        """Set appropriate models to evaluation mode."""
        models = self.pipeline.orchestrator.models
        
        if phase == "unet":
            if modal == "mr":
                models['encoder_mr'].eval()
            else:
                models['encoder_ct'].eval()
        elif phase == "resnet":
            # ResNet phase needs both encoder and decoder
            if modal == "mr":
                models['encoder_mr'].eval()
            else:
                models['encoder_ct'].eval()
            models['decoder'].eval()
        else:  # Full pipeline
            for model in models.values():
                model.eval()
    
    def _run_test_loop(self, dataloader, modal, dataset, phase):
        """
        Run the testing loop for a specific dataset/phase combination.
        
        Args:
            dataloader: Test dataloader
            modal: Data modality ('mr' or 'ct')
            dataset: Dataset name
            phase: Test phase ('unet' or 'resnet')
            
        Returns:
            Dictionary containing metrics
        """
        print(f"Running {phase.upper()} inference on {dataset.upper()} ({modal.upper()}) dataset...")
        
        # Initialize metric tracking
        dice_scores = []
        mse_scores = []
        sample_count = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Get images and labels with correct keys
                if modal == "mr":
                    images = batch["mr_image"].to(self.device)
                    labels = batch["mr_label"].to(self.device)
                else:
                    images = batch["ct_image"].to(self.device)
                    labels = batch["ct_label"].to(self.device)
                
                # Removed repetitive batch processing print to avoid clutter
                
                if phase == "unet":
                    # UNet testing (returns dice and prediction logits)
                    dice_score, pred_logits = self._test_unet_batch(images, labels, modal)
                    if dice_score is not None:
                        dice_scores.append(dice_score)
                    # Log visualization for every batch
                    _log_wandb_images(
                        image_tensor=images.cpu(),
                        true_labels=labels.cpu(),
                        pred_labels=pred_logits.cpu(),
                        caption_prefix=f"{phase}/{dataset}",
                        modal=modal,
                        dataset=dataset,
                        num_classes=self.params.num_classes,
                        step=batch_idx,
                        tester=None,  # No longer needed - histogram matching handled by HistogramMatchd transform
                    )
                
                elif phase == "resnet":
                    # ResNet testing
                    mse_score = self._test_resnet_batch(images, labels, modal)
                    if mse_score is not None:
                        mse_scores.append(mse_score)
                
                sample_count += images.shape[0]
                
                # Limit samples if specified
                if self.params.max_samples > 0 and sample_count >= self.params.max_samples:
                    break
                    
            except Exception as e:
                print(f"    Error processing batch {batch_idx + 1}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        results = {
            'phase': phase,
            'dataset': dataset,
            'modality': modal,
            'num_samples_tested': sample_count,
            'processing_time': processing_time,
            'test_status': 'completed'
        }
        
        if phase == 'unet':
            mean_dice = np.mean(dice_scores) if dice_scores else 0.0
            iou_score = mean_dice / (2 - mean_dice) if mean_dice < 2.0 and mean_dice > 0 else 0.0
            
            results.update({
                'metrics': {
                    'dice_score': mean_dice,
                    'iou_score': iou_score,
                    'mse_score': None
                }
            })
            print(f"  Average Dice: {mean_dice:.4f}, Average IoU: {iou_score:.4f}")
            
        else:  # resnet
            mean_mse = np.mean(mse_scores) if mse_scores else 0.0
            results.update({
                'metrics': {
                    'dice_score': None,
                    'iou_score': None,
                    'mse_score': mean_mse
                }
            })
            print(f"  Average MSE: {mean_mse:.6f}")
        
        print(f"Phase {phase.upper()} testing completed successfully")
        return results
    
    def _test_unet_batch(self, images, labels, modal):
        """
        Test UNet on a single batch.
        
        Args:
            images: Input images
            labels: Ground truth labels
            modal: Data modality
            
        Returns:
            Mean dice score for the batch
        """
        # Select appropriate encoder and ROI size
        if modal == 'ct':
            encoder = self.pipeline.orchestrator.models['encoder_ct']
            roi_size = self.params.crop_window_size
        else:  # mr
            encoder = self.pipeline.orchestrator.models['encoder_mr']
            roi_size = self.params.crop_window_size[:2]
        
        # Run sliding window inference
        pred_logits = sliding_window_inference(
            inputs=images,
            roi_size=roi_size,
            sw_batch_size=4,
            predictor=encoder,
            overlap=0.5,
            mode="gaussian",
        )
        
        # Convert predictions to discrete labels
        pred_labels = torch.argmax(pred_logits, dim=1, keepdim=True)
        
        # Compute Dice using the same DiceMetric as training
        self.metrics.dice_metric.reset()
        self.metrics.dice_metric(pred_labels, labels)
        dice_per_class = self.metrics.dice_metric.aggregate()
        
        # Calculate mean dice across classes
        mean_dice = dice_per_class.mean().item()
        
        # Return both dice score and logits for further logging upstream
        return mean_dice, pred_logits
    
    def _test_resnet_batch(self, images, labels, modal):
        """
        Test ResNet on a single batch (UNet + ResNet pipeline).
        
        Args:
            images: Input images
            labels: Ground truth labels
            modal: Data modality
            
        Returns:
            MSE score for distance field prediction
        """
        # Step 1: UNet forward pass
        if modal == 'ct':
            encoder = self.pipeline.orchestrator.models['encoder_ct']
            roi_size = self.params.crop_window_size
        else:  # mr
            encoder = self.pipeline.orchestrator.models['encoder_mr']
            roi_size = self.params.crop_window_size[:2]
        
        seg_pred_logits = sliding_window_inference(
            inputs=images,
            roi_size=roi_size,
            sw_batch_size=4,
            predictor=encoder,
            overlap=0.5,
            mode="gaussian",
        )
        
        # Step 2: Apply preprocessing (same as training validators)
        # Handle MR unflatten
        if modal == 'mr':
            num_items_for_unflatten = images.shape[0] // (images.shape[0] // 2 if images.shape[0] > 1 else 1)
            if seg_pred_logits.shape[0] > 1:
                seg_pred_logits = seg_pred_logits.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
                labels = labels.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
        
        # Step 3: Post-transform for ResNet input
        preprocessor = self.pipeline.orchestrator.preprocessor
        seg_pred_ds_decoder_size = preprocessor._memory_efficient_post_transform(
            seg_pred_logits, labels, modal, to_gpu=True, decoder_size=True
        )
        
        seg_pred_ds = preprocessor._memory_efficient_post_transform(
            seg_pred_logits, labels, modal, to_gpu=True, decoder_size=False
        )
        
        # Step 4: Calculate refinement mask
        binary_mask_pred = (torch.argmax(seg_pred_ds_decoder_size, dim=1, keepdim=True) == 0)
        dist_map_pred = (-distance_transform_edt(binary_mask_pred.squeeze(1)) + 
                        distance_transform_edt(~binary_mask_pred.squeeze(1))).unsqueeze(1)
        mask = torch.sigmoid(dist_map_pred * self.params.sigmoid_scale_factor + 1).detach()
        mask = mask * binary_mask_pred
        mask[mask < self.params.mask_threshold] = 0
        
        # Step 5: ResNet forward pass
        decoder = self.pipeline.orchestrator.models['decoder']
        decoder_output = decoder(seg_pred_ds)
        
        # Step 6: Combine predictions
        seg_pred_ds_refined = seg_pred_ds_decoder_size + mask * decoder_output
        
        # Step 7: Generate ground truth for comparison
        seg_true_ds_decoder_size = torch.stack([
            preprocessor._generate_downsampled_gt(labels[0], modal, decoder_size=True)
        ])
        
        # Step 8: Convert to distance fields and compute MSE
        df_pred = self._generate_distance_fields_from_segmentation(seg_pred_ds_refined)
        df_true = self._generate_distance_fields_from_segmentation(seg_true_ds_decoder_size)
        
        # Ensure tensor shapes match for MSE calculation
        if df_pred.shape != df_true.shape:
            # Adjust batch dimensions if needed
            if df_pred.shape[0] != df_true.shape[0]:
                # Use the minimum batch size to avoid shape mismatch
                min_batch = min(df_pred.shape[0], df_true.shape[0])
                df_pred = df_pred[:min_batch]
                df_true = df_true[:min_batch]
        
        # Calculate MSE
        self.metrics.mse_metric.reset()
        self.metrics.mse_metric(df_pred.cpu(), df_true.cpu())
        mse_score = self.metrics.mse_metric.aggregate()
        
        return mse_score.item()
    
    def _generate_distance_fields_from_segmentation(self, seg_pred_ds):
        """Generate distance fields from segmentation predictions (same as training validators)."""
        # Convert predictions to discrete labels
        pred_transform = AsDiscrete(argmax=True, to_onehot=4)  # 4 classes
        seg_pred_ds = torch.stack([pred_transform(i) for i in seg_pred_ds])
        
        # Generate distance fields for each class
        foreground = seg_pred_ds > 0
        lv = (seg_pred_ds == 1)
        rv = (seg_pred_ds == 3)
        myo = (seg_pred_ds == 2)
        
        df_pred = torch.stack([
            distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
            for i in [foreground, lv, rv, myo]], dim=1)
        
        return df_pred
    
    # Note: _finalize_histogram_matching method removed - functionality handled automatically by HistogramMatchd transform
    
    def _log_wandb_results(self, results):
        """Log test results to WandB."""
        print("\n--- Logging Test Results to WandB ---")
        
        summary_metrics = {}
        performance_metrics = {}
        
        for dataset in results['datasets_tested']:
            if dataset in results['unet_results']:
                unet_results = results['unet_results'][dataset]
                summary_metrics[f'unet_{dataset}_dice'] = unet_results['metrics']['dice_score']
                summary_metrics[f'unet_{dataset}_iou'] = unet_results['metrics']['iou_score']
                summary_metrics[f'unet_{dataset}_samples'] = unet_results['num_samples_tested']
                performance_metrics[f'unet_{dataset}_processing_time'] = unet_results.get('processing_time', 0.0)
            
            if dataset in results['resnet_results']:
                resnet_results = results['resnet_results'][dataset]
                summary_metrics[f'resnet_{dataset}_mse'] = resnet_results['metrics']['mse_score']
                summary_metrics[f'resnet_{dataset}_samples'] = resnet_results['num_samples_tested']
                performance_metrics[f'resnet_{dataset}_processing_time'] = resnet_results.get('processing_time', 0.0)
        
        # Log metrics to WandB
        wandb.log(summary_metrics)
        wandb.log(performance_metrics)
        
        # Log configuration
        wandb.log({
            'test_configuration': {
                'phases_tested': results['phases_tested'],
                'datasets_tested': results['datasets_tested'],
                'max_samples': self.params.max_samples,
                'total_phases': len(results['phases_tested']),
                'total_datasets': len(results['datasets_tested'])
            }
        })
        
        # Print summary
        print("Test results summary:")
        for metric, value in summary_metrics.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
        
        print("Test results logged to WandB successfully")


def run_full_test(pipeline, super_params):
    """
    Public interface: Run complete testing pipeline on MorphiNet models.
    
    This function creates a MorphiNetTester instance and runs comprehensive testing.
    
    Args:
        pipeline: MorphiNetPipeline instance (must be in inference mode)
        super_params: Configuration parameters containing test settings
    
    Returns:
        Dictionary containing comprehensive test results
    """
    print("="*80)
    print("MORPHINET INFERENCE PIPELINE (REWRITTEN)")
    print("="*80)
    print(f"Testing phase: {super_params.test_phase}")
    print(f"Testing dataset: {super_params.test_dataset}")
    
    try:
        # Create tester instance
        tester = MorphiNetTester(pipeline, super_params)
        
        # Execute testing
        test_results = tester.test()
        
        print("\nTesting completed successfully!")
        return test_results
        
    except Exception as e:
        print(f"Testing failed with error: {e}")
        raise e 