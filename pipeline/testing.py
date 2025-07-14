"""
Streamlined MorphiNet Testing Module

Uses existing validation infrastructure (validate_unet, validate_resnet)
for testing with minimal additional code.
"""

import wandb
from training.validators import MorphiNetValidator


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


def run_full_test(pipeline, super_params):
    """
    Streamlined testing using existing validation infrastructure.
    
    Args:
        pipeline: MorphiNetPipeline instance
        super_params: Configuration parameters
    """
    print("="*80)
    print("STREAMLINED MORPHINET TESTING")
    print("="*80)
    print(f"Testing phase: {super_params.test_phase}")
    print(f"Testing dataset: {super_params.test_dataset}")
    
    # Resolve dataset and phase to test
    dataset_to_test = _resolve_dataset(super_params)
    phase_to_test = _resolve_phase(super_params)
    
    if not dataset_to_test or not phase_to_test:
        print("Invalid dataset or phase specified. Aborting test.")
        return

    # Create validator instance using existing orchestrator infrastructure  
    ckpt_dir = getattr(pipeline.orchestrator, 'ckpt_dir', './checkpoints')
    validator = MorphiNetValidator(
        super_params=super_params,
        models=pipeline.orchestrator.models,
        dataloaders=pipeline.orchestrator.dataloader_manager,
        preprocessor=pipeline.orchestrator.preprocessor,
        mesh_ops=pipeline.orchestrator.mesh_ops,
        inference=pipeline.orchestrator.inference,
        ckpt_dir=ckpt_dir,
        orchestrator=pipeline.orchestrator
    )
    
    # Test the specified dataset and phase
    print(f"\n--- Testing {phase_to_test.upper()} on {dataset_to_test.upper()} ---")
    
    # Configure dataset for testing
    dataset_info = DATASET_REGISTRY[dataset_to_test]
    modal = dataset_info["modality"]
    
    # Temporarily configure dataset parameters
    original_params = _configure_dataset_params(super_params, dataset_info, modal)
    
    try:
        # Prepare test dataloaders
        pipeline.orchestrator.prepare_dataloaders(
            data_types=["test"],
            phase=phase_to_test,
            test_modal=modal
        )
        
        # Check if test loader was created
        test_loader = None
        if modal == "mr":
            test_loader = pipeline.orchestrator.dataloader_manager.mr_test_loader
        else:
            test_loader = pipeline.orchestrator.dataloader_manager.ct_test_loader
        
        if test_loader is None:
            print(f"Warning: No test data found for {dataset_to_test} ({modal})")
            return
        
        # Map test phase to appropriate test method
        save_on = "mr" if modal == "mr" else "ct"
        
        if phase_to_test == "unet":
            _run_unet_test(validator, test_loader, save_on, super_params)
        elif phase_to_test == "resnet":
            _run_resnet_test(validator, test_loader, save_on, super_params)
        elif phase_to_test == "gsn":
            _run_gsn_test(validator, test_loader, save_on, super_params, dataset_to_test)
            
    except Exception as e:
        print(f"Error testing {dataset_to_test}: {e}")
        
    finally:
        # Restore original parameters
        _restore_dataset_params(super_params, original_params)
    
    print("\nTesting completed successfully!")


def _resolve_dataset(super_params):
    """Resolve which dataset to test based on parameters."""
    dataset = super_params.test_dataset
    if dataset not in DATASET_REGISTRY:
        print(f"Warning: Unsupported dataset '{dataset}'")
        return None
    return dataset


def _resolve_phase(super_params):
    """Resolve which phase to test based on parameters.""" 
    phase = super_params.test_phase
    if phase not in ["unet", "resnet", "gsn"]:
        print(f"Warning: Unsupported phase '{phase}'")
        return None
    return phase


def _configure_dataset_params(super_params, dataset_info, modal):
    """Configure dataset parameters and return original values."""
    original_params = {}
    
    if modal == "mr":
        original_params['mr_data_dir'] = getattr(super_params, 'mr_data_dir', None)
        original_params['mr_json_dir'] = getattr(super_params, 'mr_json_dir', None)
        
        super_params.mr_data_dir = dataset_info["data_dir"]
        super_params.mr_json_dir = dataset_info["json"]
        print(f"[DATASET CONFIG] {dataset_info['json']} -> {dataset_info['data_dir']}")
    else:
        original_params['ct_data_dir'] = getattr(super_params, 'ct_data_dir', None)
        original_params['ct_json_dir'] = getattr(super_params, 'ct_json_dir', None)
        
        super_params.ct_data_dir = dataset_info["data_dir"]
        super_params.ct_json_dir = dataset_info["json"]
        print(f"[DATASET CONFIG] {dataset_info['json']} -> {dataset_info['data_dir']}")
    
    return original_params


def _restore_dataset_params(super_params, original_params):
    """Restore original dataset parameters."""
    for param, value in original_params.items():
        if value is not None:
            setattr(super_params, param, value)


def _run_unet_test(validator, test_loader, save_on, super_params):
    """
    Run segmentation testing using the dedicated test method.
    
    Args:
        validator: MorphiNetValidator instance
        test_loader: Test data loader
        save_on: Test dataset ('ct' or 'mr')
        super_params: Configuration parameters
    """
    # Limit number of batches if max_samples is specified
    if super_params.max_samples > 0:
        test_loader = _limit_dataloader(test_loader, super_params.max_samples)
    
    # Use the dedicated test method, which handles its own logging
    validator.test_unet(test_loader, save_on)


def _run_resnet_test(validator, test_loader, save_on, super_params):
    """
    Run ResNet testing using the dedicated test method.
    
    Args:
        validator: MorphiNetValidator instance
        test_loader: Test data loader
        save_on: Test dataset ('ct' or 'mr')
        super_params: Configuration parameters
    """
    # Limit number of batches if max_samples is specified
    if super_params.max_samples > 0:
        test_loader = _limit_dataloader(test_loader, super_params.max_samples)
    
    # Use the dedicated test method, which handles its own logging
    validator.test_resnet(test_loader, save_on)


def _run_gsn_test(validator, test_loader, save_on, super_params, dataset):
    """
    Run GSN testing and log results to WandB.
    
    Args:
        validator: MorphiNetValidator instance
        test_loader: Test data loader
        save_on: Test dataset ('ct' or 'mr')
        super_params: Configuration parameters
        dataset: Name of the dataset being tested
    """
    # Limit number of batches if max_samples is specified
    if super_params.max_samples > 0:
        test_loader = _limit_dataloader(test_loader, super_params.max_samples)
    
    # Use the dedicated full pipeline test method
    results = validator.test_full_pipeline(test_loader, save_on)
    
    # Log metrics to WandB
    summary_metrics = {
        f'gsn_{dataset}_mesh_dice': results['mesh_dice'],
        f'gsn_{dataset}_df_mse': results['df_mse']
    }
    wandb.log(summary_metrics)
    
    print("GSN test results summary:")
    for metric, value in summary_metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")


def _limit_dataloader(dataloader, max_samples):
    """Create a limited version of the dataloader based on max_samples."""
    class LimitedDataLoader:
        def __init__(self, original_loader, max_samples):
            self.original_loader = original_loader
            self.max_samples = max_samples
            
        def __iter__(self):
            sample_count = 0
            for batch in self.original_loader:
                if sample_count >= self.max_samples:
                    break
                yield batch
                sample_count += len(next(iter(batch.values())))  # Get batch size from first item
                
        def __len__(self):
            # Estimate length based on batch size
            if len(self.original_loader) > 0:
                estimated_batches = max(1, self.max_samples // 1)  # Assume batch size of 1
                return min(len(self.original_loader), estimated_batches)
            return 0
    
    return LimitedDataLoader(dataloader, max_samples)