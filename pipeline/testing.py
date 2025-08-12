"""
Streamlined MorphiNet Testing Module

Compact, GSN-centric testing with score computation and result exports.
Testing APIs are implemented here (moved from training/validators.py).
"""

import os
import numpy as np
import torch
import wandb
import nibabel as nib
import trimesh
from scipy.ndimage import binary_dilation
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms.utils import distance_transform_edt
from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes
 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


class MorphiNetTester:
    """Testing runner for MorphiNet full pipeline (UNet + ResNet + GSN)."""

    def __init__(self, super_params, models, dataloaders, preprocessor, mesh_ops, inference, orchestrator=None):
        self.super_params = super_params
        self.orchestrator = orchestrator
        self.encoder_mr = models['encoder_mr']
        self.encoder_ct = models['encoder_ct']
        self.decoder = models['decoder']
        self.GSN = models['GSN']
        self.dataloader_manager = dataloaders
        self.preprocessor = preprocessor
        self.mesh_ops = mesh_ops
        self.inference = inference
        self.pred_transform = None
        self.sigmoid_scale_factor = super_params.sigmoid_scale_factor
        self.mask_threshold = super_params.mask_threshold
        self.rasterizer = getattr(mesh_ops, 'rasterizer', None)

    def _ensure_pred_transform(self):
        if self.pred_transform is None:
            from monai.transforms import Compose, AsDiscrete, KeepLargestConnectedComponent
            self.pred_transform = Compose([
                AsDiscrete(argmax=True),
                KeepLargestConnectedComponent(is_onehot=True, independent=False, connectivity=3),
            ])

    @torch.no_grad()
    def test_full(self, test_loader, modal):
        self._ensure_pred_transform()
        if modal == "ct":
            encoder = self.encoder_ct
            roi_size = self.super_params.crop_window_size
        else:
            encoder = self.encoder_mr
            roi_size = self.super_params.crop_window_size[:2]
        encoder.eval(); self.decoder.eval(); self.GSN.eval()

        unet_dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        resnet_dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        msh_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")

        dataset_name = getattr(self.orchestrator, 'dataset', 'unknown') or 'unknown'
        # Map internal dataset identifiers to historical export folder names
        dataset_export_name = {'scotheart': 'sct'}.get(dataset_name, dataset_name)
        output_root = getattr(self.super_params, 'output_root', '/mnt/data/Experiment/MorphiNet/Result/')
        export_dir = os.path.join(output_root, dataset_export_name, 'MorphiNet', 'myo', 'f0')
        os.makedirs(export_dir, exist_ok=True)
        export_ablation = dataset_name in {"cap", "scotheart"}
        ablation_dir = os.path.join(output_root, 'ablation', 'MorphiNet', 'myo', 'f0')
        if export_ablation:
            os.makedirs(os.path.join(output_root, 'ablation'), exist_ok=True)

        for step, data in enumerate(test_loader):
            case_id = data.get(f"{modal}_case_id", [])[0].strip('_0000')
            img, seg_true = (
                data[f"{modal}_image"].to(DEVICE),
                data[f"{modal}_label"].to(DEVICE),
            )
            if modal == 'mr':
                roi = roi_size[:2]
            else:
                roi = roi_size

            seg_pred = sliding_window_inference(img, roi_size=roi, sw_batch_size=8, predictor=encoder, overlap=0.5, mode="gaussian")

            seg_pred_onehot_unet = self.inference._convert_to_onehot(seg_pred, self.super_params.num_classes, is_prediction=True)
            seg_true_onehot_unet = self.inference._convert_to_onehot(seg_true, self.super_params.num_classes, is_prediction=False)
            unet_dice_metric(seg_pred_onehot_unet, seg_true_onehot_unet)

            seg_pred_ds = self.preprocessor._memory_efficient_post_transform(seg_pred, seg_true, modal, to_gpu=True, decoder_size=False)

            if export_ablation and seg_pred_ds.ndim == 5:
                try:
                    unet_preds = torch.argmax(seg_pred_ds, dim=1, keepdim=True)
                    assert unet_preds.shape[0] == 1, "UNet prediction should have 1 batch element"
                    upscale_ratio = getattr(self.super_params, "upscale_ratio", 2)
                    unet_preds = torch.nn.functional.interpolate(
                        unet_preds.float(), scale_factor=upscale_ratio, mode="nearest"
                    ).to(unet_preds.dtype)
                    pred_arr = unet_preds[0, 0].cpu().numpy().astype("uint8")
                    nii_img = nib.Nifti1Image(pred_arr, np.eye(4))
                    save_dir = ablation_dir.replace("MorphiNet/myo/f0", f"ResNet_before-{modal}/myo/f0")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{case_id}_pred.nii.gz")
                    nib.save(nii_img, save_path)
                except Exception as e:
                    print(f"Warning: UNet prediction export failed: {e}")

            seg_pred_ds_decoder_size = self.preprocessor._memory_efficient_post_transform(seg_pred, seg_true, modal, to_gpu=True, decoder_size=True)
            binary_mask_pred = (torch.argmax(seg_pred_ds_decoder_size, dim=1, keepdim=True) == 0)
            dist_map_pred = (-distance_transform_edt(binary_mask_pred.squeeze(1)) + distance_transform_edt(~binary_mask_pred.squeeze(1))).unsqueeze(1)
            mask = torch.sigmoid(dist_map_pred * self.sigmoid_scale_factor + 1).detach()
            mask = mask * binary_mask_pred
            mask[mask < self.mask_threshold] = 0
            seg_pred_ds_padded, pad_info = self.inference._apply_resnet_padding(seg_pred_ds)
            resnet_output_padded = self.decoder(seg_pred_ds_padded)
            resnet_output = self.inference._remove_resnet_padding(resnet_output_padded, pad_info)
            seg_pred_ds = seg_pred_ds_decoder_size + mask * resnet_output
            seg_pred_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ds])

            seg_true_ds = self.preprocessor._generate_downsampled_gt(seg_true, modal, decoder_size=True)
            seg_pred_onehot_resnet = self.inference._convert_to_onehot(seg_pred_ds, self.super_params.num_classes, is_prediction=False)
            seg_true_onehot_resnet = self.inference._convert_to_onehot(seg_true_ds, self.super_params.num_classes, is_prediction=False)
            resnet_dice_metric(seg_pred_onehot_resnet, seg_true_onehot_resnet)

            if export_ablation and seg_pred_ds.ndim == 5:
                try:
                    pred_arr = seg_pred_ds[0, 0].cpu().numpy().astype("uint8")
                    nii_img = nib.Nifti1Image(pred_arr, np.eye(4))
                    save_dir = ablation_dir.replace("MorphiNet/myo/f0", f"ResNet_after-{modal}/myo/f0")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{case_id}_pred.nii.gz")
                    nib.save(nii_img, save_path)

                    true_arr = seg_true_ds[0, 0].cpu().numpy().astype("uint8")
                    nii_img = nib.Nifti1Image(true_arr, np.eye(4))
                    save_dir = ablation_dir.replace("MorphiNet/myo/f0", f"ResNet_gt-{modal}/myo/f0")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{case_id}_true.nii.gz")
                    nib.save(nii_img, save_path)
                except Exception as e:
                    print(f"Warning: ResNet prediction export failed: {e}")

            foreground = seg_pred_ds > 0
            lv = (seg_pred_ds == 1)
            rv = (seg_pred_ds == 3)
            myo = (seg_pred_ds == 2)
            df_pred = torch.stack([
                distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0])
                for i in [foreground, lv, rv, myo]], dim=1)

            template_mesh = self.mesh_ops.warp_template_mesh(df_pred)
            if export_ablation:
                try:
                    verts = template_mesh.verts_packed()
                    faces = template_mesh.faces_packed()
                    os.makedirs(ablation_dir.replace("MorphiNet/myo/f0", f"level_0/myo/f0"), exist_ok=True)
                    save_obj(
                        os.path.join(
                            ablation_dir.replace("MorphiNet/myo/f0", f"level_0/myo/f0"),
                            f"{case_id}.obj"
                        ),
                        verts.to(torch.float32),
                        faces.to(torch.int64)
                    )

                    # Convert PyTorch3D Meshes -> Trimesh, apply Loop subdivision, then convert back
                    # Extract vertices/faces to numpy for Trimesh
                    _verts_np = template_mesh.verts_packed().detach().cpu().numpy()
                    _faces_np = template_mesh.faces_packed().detach().cpu().numpy()
                    _tri_mesh = trimesh.Trimesh(vertices=_verts_np, faces=_faces_np, process=False)
                    _tri_mesh = _tri_mesh.subdivide_loop(iterations=2)
                    _device = template_mesh.verts_packed().device
                    _verts_t = torch.from_numpy(_tri_mesh.vertices).to(device=_device, dtype=torch.float32)
                    _faces_t = torch.from_numpy(_tri_mesh.faces).to(device=_device, dtype=torch.int64)
                    template_mesh_loo = Meshes(verts=[_verts_t], faces=[_faces_t])

                    verts = template_mesh_loo.verts_packed()
                    faces = template_mesh_loo.faces_packed()
                    os.makedirs(ablation_dir.replace("MorphiNet/myo/f0", f"loop/myo/f0"), exist_ok=True)
                    save_obj(
                        os.path.join(
                            ablation_dir.replace("MorphiNet/myo/f0", f"loop/myo/f0"),
                            f"{case_id}.obj"
                        ),
                        verts.to(torch.float32),
                        faces.to(torch.int64)
                    )
                except Exception as e:
                    print(f"Warning: Warped template mesh export failed: {e}")

            all_levels = self.GSN(template_mesh, self.mesh_ops.subdivided_faces.faces_levels, df_pred, self.mesh_ops.subdivided_faces.labels_levels)
            subdiv_mesh = all_levels[-1]
            if export_ablation:
                try:
                    for lvl_idx, lvl_mesh in enumerate(all_levels):
                        os.makedirs(ablation_dir.replace("MorphiNet/myo/f0", f"level_{lvl_idx+1}/myo/f0"), exist_ok=True)
                        verts = lvl_mesh.verts_packed()
                        faces = lvl_mesh.faces_packed()
                        save_obj(
                            os.path.join(
                                ablation_dir.replace("MorphiNet/myo/f0", f"level_{lvl_idx+1}/myo/f0"), 
                                f"{case_id}.obj"
                                ),
                            verts.to(torch.float32), 
                            faces.to(torch.int64)
                            )
                except Exception as e:
                    print(f"Warning: GSN multi-level export failed: {e}")

            voxeld_mesh = torch.cat([
                self.rasterizer(pred_mesh.verts_padded(), pred_mesh.faces_padded())
                for pred_mesh in subdiv_mesh
            ], dim=0)

            seg_true_ds = (seg_true_ds == 2).to(torch.float32)
            msh_metric_batch(voxeld_mesh, seg_true_ds)

            verts = subdiv_mesh.verts_packed()
            faces = subdiv_mesh.faces_packed()
            save_obj(os.path.join(export_dir, f"{case_id}.obj"),
                        verts.to(torch.float32), 
                        faces.to(torch.int64))

        def _avg(d):
            return d.mean().item() if d is not None and not torch.isnan(d.mean()) else 0.0
        unet_avg = _avg(unet_dice_metric.aggregate())
        resnet_avg = _avg(resnet_dice_metric.aggregate())
        mesh_dice = _avg(msh_metric_batch.aggregate())
        return {"mesh_dice": mesh_dice, "unet_dice": unet_avg, "resnet_dice": resnet_avg}


def run_full_test(pipeline, super_params):
    """
    Single-pass MorphiNet testing using the full pipeline (UNet+ResNet+GSN).
    Always runs the complete testing stage; per-phase branches are removed.
    
    Args:
        pipeline: MorphiNetPipeline instance
        super_params: Configuration parameters
    """
    print("="*80)
    print("STREAMLINED MORPHINET TESTING")
    print("="*80)
    # Phase argument removed; always run full pipeline
    print(f"Testing dataset: {super_params.test_dataset}")
    
    # Resolve dataset to test
    dataset_to_test = _resolve_dataset(super_params)
    
    if not dataset_to_test:
        print("Invalid dataset specified. Aborting test.")
        return

    tester = MorphiNetTester(
        super_params=super_params,
        models=pipeline.orchestrator.models,
        dataloaders=pipeline.orchestrator.dataloader_manager,
        preprocessor=pipeline.orchestrator.preprocessor,
        mesh_ops=pipeline.orchestrator.mesh_ops,
        inference=pipeline.orchestrator.inference,
        orchestrator=pipeline.orchestrator,
    )
    
    # Test the specified dataset and phase
    print(f"\n--- Testing FULL PIPELINE on {dataset_to_test.upper()} ---")
    
    # Configure dataset for testing
    dataset_info = DATASET_REGISTRY[dataset_to_test]
    modal = dataset_info["modality"]
    
    # Temporarily configure dataset parameters
    original_params = _configure_dataset_params(super_params, dataset_info, modal)
    
    try:
        # Prepare test dataloaders
        pipeline.orchestrator.prepare_dataloaders(
            data_types=["test"],
            phase="gsn",
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
        
        # Run full pipeline once (UNet, ResNet, and GSN handled inside)
        save_on = "mr" if modal == "mr" else "ct"
        _run_gsn_test(tester, test_loader, save_on, super_params, dataset_to_test)
            
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


# phase selection has been removed; full pipeline only


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


# UNet/ResNet standalone runners removed; they are executed inside the full pipeline


def _run_gsn_test(tester, test_loader, save_on, super_params, dataset):
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
    results = tester.test_full(test_loader, save_on)
    
    # Log metrics to WandB
    summary_metrics = {
        f'unet_{dataset}_dice': results['unet_dice'],
        f'resnet_{dataset}_dice': results['resnet_dice'],
        f'gsn_{dataset}_dice': results['mesh_dice'],
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