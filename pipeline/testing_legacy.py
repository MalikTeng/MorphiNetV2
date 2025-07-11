"""
MorphiNet Testing Pipeline Module

This module contains all testing logic for MorphiNet models, providing
real model inference capabilities for UNet and ResNet phases.
"""

import os
import sys
import time
import json
from pathlib import Path
import torch
import numpy as np
import wandb
from monai.inferers import sliding_window_inference
from scipy.ndimage import distance_transform_edt

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


def _log_wandb_images(image_tensor, true_labels, pred_labels, caption_prefix, modal, num_classes=4):
    """
    Logs visualization of image, ground truth, and prediction to WandB.
    Following the same pattern as training/trainer.py
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
            slice_idx = image_tensor.shape[0] // 4
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
        
        wandb.log(log_data)
        
    except Exception as e:
        print(f"      Warning: Could not log visualization: {e}")


def _run_inference_tests(pipeline, super_params):
    """
    Run inference tests on specified datasets and phases.
    
    Args:
        pipeline: MorphiNetPipeline instance
        super_params: Configuration parameters
    
    Returns:
        Dictionary containing test results
    """
    results = {
        'unet_results': {},
        'resnet_results': {},
        'datasets_tested': [],
        'phases_tested': []
    }
    
    # Determine datasets to test
    if super_params.test_dataset == "both":
        datasets_to_test = ["acdc", "mmwhs"]  # keep legacy meaning
    else:
        datasets_to_test = [super_params.test_dataset]

    # Filter out unsupported dataset names gracefully
    datasets_to_test = [d for d in datasets_to_test if d in DATASET_REGISTRY]
    if not datasets_to_test:
        print(f"Warning: No supported dataset specified (received {super_params.test_dataset}).")
        return results
    
    # Determine phases to test
    phases_to_test = []
    if super_params.test_phase == "both":
        phases_to_test = ["unet", "resnet"]
    else:
        phases_to_test = [super_params.test_phase]
    
    results['datasets_tested'] = datasets_to_test
    results['phases_tested'] = phases_to_test
    
    print(f"Testing {len(datasets_to_test)} dataset(s) and {len(phases_to_test)} phase(s)")
    
    # Run tests for each combination
    for dataset in datasets_to_test:
        for phase in phases_to_test:
            print(f"\n--- Testing {phase.upper()} on {dataset.upper()} ---")
            
            # Configure dataset parameters using registry
            dataset_info = DATASET_REGISTRY[dataset]
            modality = dataset_info["modality"]

            if modality == "mr":
                super_params.mr_data_dir = dataset_info["data_dir"]
                super_params.mr_json_dir = dataset_info["json"]
            else:
                super_params.ct_data_dir = dataset_info["data_dir"]
                super_params.ct_json_dir = dataset_info["json"]
            
            # Run inference test
            phase_results = _test_single_phase(pipeline, phase, dataset, modality, super_params)
            
            # Store results
            if phase == "unet":
                results['unet_results'][dataset] = phase_results
            else:
                results['resnet_results'][dataset] = phase_results
    
    return results


def _test_single_phase(pipeline, phase, dataset, modality, super_params):
    """
    Test a single phase (UNet or ResNet) on a specific dataset.
    
    Args:
        pipeline: MorphiNetPipeline instance
        phase: Phase to test ('unet' or 'resnet')
        dataset: Dataset name ('acdc' or 'mmwhs')
        modality: Data modality ('ct' or 'mr')
        super_params: Configuration parameters
    
    Returns:
        Dictionary containing phase test results
    """
    print(f"Running {phase.upper()} inference on {dataset.upper()} ({modality.upper()}) dataset...")
    
    try:
        # Import metrics
        from evaluation.metrics import MorphiNetMetrics
        
        # Initialize metrics
        metrics_calculator = MorphiNetMetrics(
            num_classes=super_params.num_classes,
            include_background=False
        )
        
        # Load test data
        test_results = _load_and_test_data(pipeline, phase, dataset, modality, super_params, metrics_calculator)
        
        # Calculate metrics based on phase
        if phase == 'unet':
            # UNet evaluation: Dice and IoU scores
            dice_score = test_results.get('dice_score', 0.0)
            iou_score = test_results.get('iou_score', 0.0)
            
            metrics = {
                'dice_score': dice_score,
                'iou_score': iou_score,
                'mse_score': None
            }
        else:  # resnet
            # ResNet evaluation: MSE for distance field prediction
            mse_score = test_results.get('mse_score', 0.0)
            
            metrics = {
                'dice_score': None,
                'iou_score': None,
                'mse_score': mse_score
            }
        
        results = {
            'phase': phase,
            'dataset': dataset,
            'modality': modality,
            'num_samples_tested': test_results.get('num_samples', super_params.max_samples if super_params.max_samples > 0 else 5),
            'metrics': metrics,
            'processing_time': test_results.get('processing_time', 0.0),
            'visualizations': [],
            'test_status': 'completed'
        }
        
        print(f"Phase {phase.upper()} testing completed successfully")
        print(f"  Metrics: {metrics}")
        return results
        
    except Exception as e:
        print(f"Error in {phase.upper()} testing: {e}")
        # Return fallback results with error status
        return {
            'phase': phase,
            'dataset': dataset,
            'modality': modality,
            'num_samples_tested': 0,
            'metrics': {
                'dice_score': None,
                'iou_score': None,
                'mse_score': None
            },
            'visualizations': [],
            'test_status': 'failed',
            'error': str(e)
        }


def _load_and_test_data(pipeline, phase, dataset, modality, super_params, metrics_calculator):
    """
    Load test data and perform inference with metric calculation.
    
    Args:
        pipeline: MorphiNetPipeline instance
        phase: Phase to test ('unet' or 'resnet')
        dataset: Dataset name ('acdc' or 'mmwhs')
        modality: Data modality ('ct' or 'mr')
        super_params: Configuration parameters
        metrics_calculator: MorphiNetMetrics instance
    
    Returns:
        Dictionary containing calculated metrics and processing info
    """
    start_time = time.time()
    
    # Load dataset configuration using registry
    dataset_info = DATASET_REGISTRY.get(dataset)
    if dataset_info is None:
        print(f"Warning: Unsupported dataset {dataset}")
        return {}

    config_path = Path(dataset_info["json"])
    data_base_path = Path(dataset_info["data_dir"])
    
    if not config_path.exists():
        print(f"Warning: Dataset config not found: {config_path}")
        # Return empty results, don't use mock data
        return {}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get test samples (limit to max_samples if specified)
    test_samples = config.get("test", [])[:super_params.max_samples] if super_params.max_samples > 0 else config.get("test", [])
    
    if not test_samples:
        print(f"No test samples found for dataset {dataset}")
        # Return empty results, don't use mock data
        return {}
    
    print(f"Testing on {len(test_samples)} samples")
    
    # Initialize metric tracking
    dice_scores = []
    iou_scores = []
    mse_scores = []
    
    # Process each test sample
    for i, sample in enumerate(test_samples):
        try:
            print(f"  Processing sample {i+1}/{len(test_samples)}: {sample.get('image', 'unknown')}")
            
            # Create sample data paths
            sample_paths = _create_sample_paths(sample, data_base_path, modality)
            
            if not all(Path(p).exists() for p in sample_paths.values()):
                print(f"    Warning: Sample files not found, skipping")
                continue
            
            # Run inference based on phase
            if phase == 'unet':
                # UNet testing: segmentation prediction
                sample_dice, sample_iou = _test_unet_sample(pipeline, sample_paths, metrics_calculator, i, dataset)
                if sample_dice is not None:
                    dice_scores.append(sample_dice)
                if sample_iou is not None:
                    iou_scores.append(sample_iou)
                    
            else:  # resnet
                # ResNet testing: distance field prediction
                sample_mse = _test_resnet_sample(pipeline, sample_paths, metrics_calculator)
                if sample_mse is not None:
                    mse_scores.append(sample_mse)
        
        except Exception as e:
            print(f"    Error processing sample {i+1}: {e}")
            continue
    
    processing_time = time.time() - start_time
    
    # Calculate average metrics
    results = {
        'num_samples': len(test_samples),
        'processing_time': processing_time
    }
    
    if phase == 'unet':
        results['dice_score'] = np.mean(dice_scores) if dice_scores else 0.0
        results['iou_score'] = np.mean(iou_scores) if iou_scores else 0.0
        print(f"  Average Dice: {results['dice_score']:.4f}, Average IoU: {results['iou_score']:.4f}")
    else:
        results['mse_score'] = np.mean(mse_scores) if mse_scores else 0.0
        print(f"  Average MSE: {results['mse_score']:.6f}")
    
    return results


def _create_sample_paths(sample, data_base_path, modality):
    """Create file paths for a sample."""
    if modality == "mr":
        return {
            "mr_image": data_base_path / sample["image"],
            "mr_label": data_base_path / sample["label"]
        }
    else:  # ct
        return {
            "ct_image": data_base_path / sample["image"], 
            "ct_label": data_base_path / sample["label"]
        }


def _test_unet_sample(pipeline, sample_paths, metrics_calculator, sample_idx, dataset):
    """Test UNet inference on a single sample."""
    try:
        # Load and process data through the pipeline
        from data.components import UniversalCanonicalResampled
        
        # Configure transforms based on sample modality
        if "mr_image" in sample_paths:
            keys = ["mr_image", "mr_label"]
            modal = "mr"
        else:
            keys = ["ct_image", "ct_label"]
            modal = "ct"
        
        # Create transform
        transform = UniversalCanonicalResampled(
            keys=keys,
            dataset=dataset,
            modal=modal,
            target_spacing=(2.0, 2.0, 2.0),
            allow_missing_keys=False
        )
        
        # Process sample
        processed_data = transform(sample_paths)
        
        # Get processed tensors
        image_tensor = processed_data[keys[0]]
        label_tensor = processed_data[keys[1]]
        
        # Ensure tensor is in correct format for inference
        if hasattr(image_tensor, 'as_tensor'):
            image_tensor = image_tensor.as_tensor()
        if hasattr(label_tensor, 'as_tensor'):
            label_tensor = label_tensor.as_tensor()
        
        # Apply proper data collation using the same pattern as training pipeline
        from data.dataset_utils import collate_4D_batch
        
        # Prepare data in the format expected by collate_4D_batch
        sample_data = [{keys[0]: image_tensor, keys[1]: label_tensor}]
        collated_data = collate_4D_batch(sample_data)
        
        image_tensor = collated_data[keys[0]]
        label_tensor = collated_data[keys[1]]
        
        # Perform actual model inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if modal == 'ct':
            encoder = pipeline.orchestrator.models['encoder_ct']
            roi_size = pipeline.super_params.crop_window_size
            image_tensor_dev = image_tensor.to(device)
        else:  # mr
            encoder = pipeline.orchestrator.models['encoder_mr']
            roi_size = pipeline.super_params.crop_window_size[:2]
            image_tensor_dev = image_tensor.to(device)
        
        encoder.eval()
        
        with torch.no_grad():
            pred_tensor_dev = sliding_window_inference(
                inputs=image_tensor_dev,
                roi_size=roi_size,
                sw_batch_size=4,
                predictor=encoder,
                overlap=0.5,
                mode="gaussian",
            )
        
        # Convert prediction to labels and move to CPU
        pred_labels = torch.argmax(pred_tensor_dev, dim=1).cpu()
        
        # Get ground truth labels
        true_labels = label_tensor.get_array() if hasattr(label_tensor, 'get_array') else label_tensor
        if torch.is_tensor(true_labels):
            true_labels = true_labels.cpu().numpy()
        
        # Remove extra channel dimension if present (from collate_4D_batch)
        if len(true_labels.shape) == 4 and true_labels.shape[1] == 1:
            true_labels = true_labels.squeeze(1)  # Remove channel dimension for MR data
        elif len(true_labels.shape) == 5 and true_labels.shape[1] == 1:
            true_labels = true_labels.squeeze(1)  # Remove channel dimension for CT data
        
        # Calculate metrics - average across all slices
        if len(pred_labels.shape) > 2:  # Multi-slice data
            dice_scores = []
            for slice_idx in range(pred_labels.shape[0]):
                try:
                    pred_slice = pred_labels[slice_idx:slice_idx+1]  # Keep batch dim
                    true_slice = torch.from_numpy(true_labels[slice_idx:slice_idx+1]) if isinstance(true_labels, np.ndarray) else true_labels[slice_idx:slice_idx+1]
                    
                    slice_dice = metrics_calculator.compute_dice_score(pred_slice, true_slice)
                    
                    # Aggregate dice score for this slice (mean across spatial dimensions)
                    if torch.is_tensor(slice_dice):
                        slice_dice_mean = slice_dice.mean().item()
                    else:
                        slice_dice_mean = slice_dice
                    
                    if not np.isnan(slice_dice_mean) and slice_dice_mean > 0:
                        dice_scores.append(slice_dice_mean)
                except Exception as e:
                    print(f"      Warning: Error computing dice for slice {slice_idx}: {e}")
                    continue
            
            dice_score = np.mean(dice_scores) if dice_scores else 0.0
        else:
            # Single slice
            dice_score = metrics_calculator.compute_dice_score(
                pred_labels.unsqueeze(0),  # Add batch for metric
                torch.from_numpy(true_labels).unsqueeze(0) if isinstance(true_labels, np.ndarray) else true_labels.unsqueeze(0)
            )
            dice_score = dice_score.item() if torch.is_tensor(dice_score) else dice_score
        
        # Calculate IoU from Dice
        iou_score = dice_score / (2 - dice_score) if dice_score < 2.0 and dice_score > 0 else 0.0
        
        # Log visualization for the first sample of each dataset
        if sample_idx == 0:
            try:
                caption_prefix = f"unet/{dataset}_sample_{sample_idx}"
                # Use the new visualization function following training pattern
                _log_wandb_images(
                    image_tensor=image_tensor,
                    true_labels=true_labels,
                    pred_labels=pred_tensor_dev,
                    caption_prefix=caption_prefix,
                    modal=modal,
                    num_classes=4
                )
            except Exception as e:
                print(f"      Warning: Could not log visualization: {e}")

        return dice_score, iou_score
    
    except Exception as e:
        print(f"      UNet sample test error: {e}")
        return None, None


def _generate_true_df(label_tensor):
    """Generates the ground truth distance field from a label tensor."""
    if hasattr(label_tensor, 'get_array'):
        label_array = label_tensor.get_array()
    else:
        label_array = label_tensor.numpy()
    
    # Assumes label 0 is background
    binary_mask = (label_array.squeeze() == 0)
    
    df = -distance_transform_edt(binary_mask) + distance_transform_edt(~binary_mask)
    return torch.from_numpy(df).float().unsqueeze(0) # Add channel dim


def _generate_distance_fields_from_segmentation(seg_pred_ds):
    """Generate distance fields from segmentation predictions (following training validator pattern)."""
    from monai.transforms import AsDiscrete
    from monai.transforms.utils import distance_transform_edt
    
    # Convert predictions to discrete labels
    pred_transform = AsDiscrete(argmax=True, to_onehot=4)  # 4 classes
    seg_pred_ds = torch.stack([pred_transform(i) for i in seg_pred_ds])
    
    # Generate distance fields for each class (same as training validators)
    foreground = seg_pred_ds > 0
    lv = (seg_pred_ds == 1)
    rv = (seg_pred_ds == 3)
    myo = (seg_pred_ds == 2)
    
    df_pred = torch.stack([
        distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
        for i in [foreground, lv, rv, myo]], dim=1)
    
    return df_pred


def _test_resnet_sample(pipeline, sample_paths, metrics_calculator):
    """Test ResNet inference on a single sample."""
    try:
        # Load and process data similar to UNet
        from data.components import UniversalCanonicalResampled
        
        # Configure transforms
        if "mr_image" in sample_paths:
            keys = ["mr_image", "mr_label"]
            dataset = "acdc"
            modal = "mr"
        else:
            keys = ["ct_image", "ct_label"]
            dataset = "mmwhs"
            modal = "ct"
        
        transform = UniversalCanonicalResampled(
            keys=keys,
            dataset=dataset,
            modal=modal,
            target_spacing=(2.0, 2.0, 2.0),
            allow_missing_keys=False
        )
        
        processed_data = transform(sample_paths)
        
        image_tensor = processed_data[keys[0]]
        label_tensor = processed_data[keys[1]]

        # Ensure tensor is in correct format for inference
        if hasattr(image_tensor, 'as_tensor'):
            image_tensor = image_tensor.as_tensor()
        if hasattr(label_tensor, 'as_tensor'):
            label_tensor = label_tensor.as_tensor()
        
        # Apply proper data collation using the same pattern as training pipeline
        from data.dataset_utils import collate_4D_batch
        
        # Prepare data in the format expected by collate_4D_batch
        sample_data = [{keys[0]: image_tensor, keys[1]: label_tensor}]
        collated_data = collate_4D_batch(sample_data)
        
        image_tensor = collated_data[keys[0]]
        label_tensor = collated_data[keys[1]]

        # --- UNet Pass ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if modal == 'ct':
            encoder = pipeline.orchestrator.models['encoder_ct']
            roi_size = pipeline.super_params.crop_window_size
        else:  # mr
            encoder = pipeline.orchestrator.models['encoder_mr']
            roi_size = pipeline.super_params.crop_window_size[:2]

        encoder.eval()
        image_tensor_dev = image_tensor.to(device)
        label_tensor_dev = label_tensor.to(device)
        
        with torch.no_grad():
            seg_pred_logits = sliding_window_inference(
                inputs=image_tensor_dev,
                roi_size=roi_size,
                sw_batch_size=4,
                predictor=encoder,
                overlap=0.5,
                mode="gaussian",
            )
        
        # Apply unflatten for MR data (following training validator pattern)
        if modal == 'mr':
            num_items_for_unflatten = 1  # For single sample testing, use 1 instead of 2
            seg_pred_logits = seg_pred_logits.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)
            label_tensor_dev = label_tensor_dev.unflatten(0, (num_items_for_unflatten, -1)).swapaxes(1, 2)

        # --- ResNet Pass ---
        decoder = pipeline.orchestrator.models['decoder'].to(device)
        decoder.eval()
        
        # Prepare segmentation for decoder (following training validator pattern)
        preprocessor = pipeline.orchestrator.preprocessor
        
        # Process predictions through ResNet pipeline (same as training validators)
        seg_pred_ds_decoder_size = preprocessor._memory_efficient_post_transform(
            seg_pred_logits, label_tensor_dev, modal, to_gpu=True, decoder_size=True
        )
        
        seg_pred_ds = preprocessor._memory_efficient_post_transform(
            seg_pred_logits, label_tensor_dev, modal, to_gpu=True, decoder_size=False
        )
        
        # Calculate mask for refinement (same as training validators)
        binary_mask_pred = (torch.argmax(seg_pred_ds_decoder_size, dim=1, keepdim=True) == 0)
        from monai.transforms.utils import distance_transform_edt
        dist_map_pred = (-distance_transform_edt(binary_mask_pred.squeeze(1)) + distance_transform_edt(~binary_mask_pred.squeeze(1))).unsqueeze(1)
        mask = torch.sigmoid(dist_map_pred * pipeline.super_params.sigmoid_scale_factor + 1).detach()
        mask = mask * binary_mask_pred
        mask[mask < pipeline.super_params.mask_threshold] = 0
        
        # Apply decoder with padding (same as training validators)
        try:
            seg_pred_ds_padded, pad_info = pipeline.orchestrator.inference._apply_resnet_padding(seg_pred_ds)
            with torch.no_grad():
                resnet_output_padded = decoder(seg_pred_ds_padded)
            decoder_output = pipeline.orchestrator.inference._remove_resnet_padding(resnet_output_padded, pad_info)
        except Exception as e:
            print(f"      Warning: ResNet padding failed, trying direct inference: {e}")
            with torch.no_grad():
                decoder_output = decoder(seg_pred_ds)
        
        # Combine predictions (ResNet refined segmentation)
        seg_pred_ds_refined = seg_pred_ds_decoder_size + mask * decoder_output
        
        # Generate downsampled ground truth for fair comparison
        seg_true_ds_decoder_size = torch.stack([
            preprocessor._generate_downsampled_gt(label_tensor_dev[0], modal, decoder_size=True)
        ])
        
        # Calculate MSE between refined segmentation and ground truth
        # Convert to distance fields for MSE calculation
        df_pred = _generate_distance_fields_from_segmentation(seg_pred_ds_refined)
        df_true = _generate_distance_fields_from_segmentation(seg_true_ds_decoder_size)
        
        # Calculate MSE
        mse_score = metrics_calculator.compute_mse_score(df_pred.cpu(), df_true.cpu())
        
        return mse_score.item() if torch.is_tensor(mse_score) else mse_score
    
    except Exception as e:
        print(f"      ResNet sample test error: {e}")
        return None


def _log_test_results(test_results, super_params):
    """
    Log test results to WandB for systematic reporting.
    
    Args:
        test_results: Dictionary containing test results
        super_params: Configuration parameters
    """
    print("\n--- Logging Test Results to WandB ---")
    
    # Log summary metrics
    summary_metrics = {}
    performance_metrics = {}
    
    for dataset in test_results['datasets_tested']:
        if dataset in test_results['unet_results']:
            unet_results = test_results['unet_results'][dataset]
            summary_metrics[f'unet_{dataset}_dice'] = unet_results['metrics']['dice_score']
            summary_metrics[f'unet_{dataset}_iou'] = unet_results['metrics']['iou_score']
            summary_metrics[f'unet_{dataset}_samples'] = unet_results['num_samples_tested']
            performance_metrics[f'unet_{dataset}_processing_time'] = unet_results.get('processing_time', 0.0)
            performance_metrics[f'unet_{dataset}_status'] = unet_results.get('test_status', 'unknown')
        
        if dataset in test_results['resnet_results']:
            resnet_results = test_results['resnet_results'][dataset]
            summary_metrics[f'resnet_{dataset}_mse'] = resnet_results['metrics']['mse_score']
            summary_metrics[f'resnet_{dataset}_samples'] = resnet_results['num_samples_tested']
            performance_metrics[f'resnet_{dataset}_processing_time'] = resnet_results.get('processing_time', 0.0)
            performance_metrics[f'resnet_{dataset}_status'] = resnet_results.get('test_status', 'unknown')
    
    # Log metrics to WandB
    wandb.log(summary_metrics)
    wandb.log(performance_metrics)
    
    # Log configuration
    wandb.log({
        'test_configuration': {
            'phases_tested': test_results['phases_tested'],
            'datasets_tested': test_results['datasets_tested'],
            'max_samples': super_params.max_samples,
            'total_phases': len(test_results['phases_tested']),
            'total_datasets': len(test_results['datasets_tested'])
        }
    })
    
    # Print summary to console
    print("Test results summary:")
    for metric, value in summary_metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
    
    print("Test results logged to WandB successfully")


def run_full_test(pipeline, super_params):
    """
    Public interface: Run complete testing pipeline on MorphiNet models.
    
    This function orchestrates the full testing workflow including:
    - Loading trained model weights
    - Running inference on specified datasets and phases
    - Computing metrics and generating visualizations
    - Logging results to WandB
    
    Args:
        pipeline: MorphiNetPipeline instance (must be in inference mode)
        super_params: Configuration parameters containing test settings
    
    Returns:
        Dictionary containing comprehensive test results
    """
    print("="*80)
    print("MORPHINET INFERENCE PIPELINE")
    print("="*80)
    print(f"Testing phase: {super_params.test_phase}")
    print(f"Testing dataset: {super_params.test_dataset}")
    
    try:
        # Execute inference testing
        test_results = _run_inference_tests(pipeline, super_params)
        
        # Log results to WandB
        _log_test_results(test_results, super_params)
        
        print("\nTesting completed successfully!")
        return test_results
        
    except Exception as e:
        print(f"Testing failed with error: {e}")
        raise e 