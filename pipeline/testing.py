"""
Streamlined MorphiNet Testing Module

Compact, GSN-centric testing with score computation and result exports.
Testing APIs are implemented here (moved from training/validators.py).
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import nibabel as nib
import trimesh
import time
from scipy.ndimage import binary_dilation
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms.utils import distance_transform_edt, generate_spatial_bounding_box
from monai.transforms import (
    Compose, AsDiscrete, KeepLargestConnectedComponent, Lambda, EnsureType
    )
from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes
from einops import rearrange

# Import new metrics and export functionality
from utils.xlsx_exporter import export_ablation_study_to_xlsx
from utils.path_config import get_dataset_registry, get_path_default

# Import refactored testing modules
from utils.testing.basic_testing import (
    create_template_mesh_variants, compute_basic_metrics, 
    compute_comprehensive_metrics, export_metrics_xlsx, create_ground_truth_mesh
)
from utils.testing.ablation_testing import (
    compute_ablation_dice_scores_all_labels, compute_ablation_hausdorff_distances,
    compute_volume_differences, extract_phase_from_case_id
)
from utils.testing.mr_testing import (
    create_mr_ground_truth_mask, compute_mr_mesh_metrics, 
    export_mr_metrics_xlsx
)
from utils.testing.visualization_testing import (
    create_surface_distance_visualization, create_mr_slice_visualization
)
 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Dataset registry mapping dataset identifiers to metadata
DATASET_REGISTRY = get_dataset_registry()


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
        self.pred_transform = Compose([
            AsDiscrete(argmax=True),
            KeepLargestConnectedComponent(is_onehot=False, independent=False, connectivity=3),
            EnsureType(data_type="tensor", dtype=torch.int8)
        ])
        self.rasterizer = getattr(mesh_ops, 'rasterizer', None)

    @torch.no_grad()
    def test_full(self, test_loader, modal):
        if modal == "ct":
            encoder = self.encoder_ct
            roi_size = self.super_params.crop_window_size
        else:
            encoder = self.encoder_mr
            roi_size = self.super_params.crop_window_size[:2]
        encoder.eval(); self.decoder.eval(); self.GSN.eval()

        unet_dice_metric = DiceMetric(include_background=False, reduction="none")
        resnet_dice_metric = DiceMetric(include_background=False, reduction="none") 
        msh_metric_batch = DiceMetric(include_background=False, reduction="none")
        hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="none", percentile=75)
        
        all_case_ids = []
        all_gt_meshes = []
        all_seg_true = []
        all_mesh_variants = {}
        all_mr_metrics = []  # Store MR-specific metrics for separate SAX/LAX reporting

        dataset_name = getattr(self.orchestrator, 'dataset', 'unknown') or 'unknown'
        dataset_export_name = {'scotheart': 'sct'}.get(dataset_name, dataset_name)  # Map internal dataset identifiers to historical export folder names
        output_root = getattr(self.super_params, 'output_root', get_path_default('MORPHINET_OUTPUT_ROOT'))
        export_dir = os.path.join(output_root, dataset_export_name, 'MorphiNet', 'myo', 'f0')
        os.makedirs(export_dir, exist_ok=True)
        export_ablation = dataset_name in {"cap", "scotheart"}
        ablation_dir = os.path.join(output_root, 'ablation', 'MorphiNet', 'myo', 'f0')
        if export_ablation:
            os.makedirs(os.path.join(output_root, 'ablation'), exist_ok=True)
            
        # Ablation study data collection - enhanced with detailed metrics
        ablation_metrics = {
            'case_ids': [],
            'phases': [],
            'modal': modal,
            'dataset_name': dataset_name,
            # Dice scores for each label
            'before_resnet_dice': {'LV': [], 'MYO': [], 'RV': []},
            'after_resnet_dice': {'LV': [], 'MYO': [], 'RV': []},
            # Hausdorff distances for each label  
            'before_resnet_hausdorff': {'LV': [], 'MYO': [], 'RV': []},
            'after_resnet_hausdorff': {'LV': [], 'MYO': [], 'RV': []},
            # Volume differences (absolute and percentage)
            'volume_differences': {'LV': [], 'MYO': [], 'RV': []}
        }

        # Initialize timing statistics
        timing_stats = {
            'unet_inference': 0.0,
            'resnet_processing': 0.0,
            'distance_transform': 0.0,
            'template_warping': 0.0,
            'gsn_processing': 0.0,
            'case_count': 0
        }

        for _, data in enumerate(test_loader):
            case_id = data.get(f"{modal}_case_id", [])[0].strip('_0000')

            img, seg_true = (
                data[f"{modal}_image"].to(DEVICE),
                data[f"{modal}_label"].to(DEVICE),
            )
            
            # Extract world coordinate parameters for visualization (specific CT cases only)
            world_params = None
            if modal == 'ct' and case_id in ['130352_CE-ED', 'ct_train_1007']:
                # Extract affine matrix from seg_true metadata
                if hasattr(data[f"{modal}_label"], 'meta') and 'original_affine' in data[f"{modal}_label"].meta:
                    world_affine = data[f"{modal}_label"].meta['original_affine'].numpy()
                elif hasattr(data[f"{modal}_label"], 'meta') and 'affine' in data[f"{modal}_label"].meta:
                    world_affine = data[f"{modal}_label"].meta['affine'].numpy()
                else:
                    # Fallback: construct affine or use identity
                    world_affine = np.eye(4)
                    print(f"Warning: No affine matrix found for {case_id}, using identity matrix")
                
                # Extract bounding box parameters from original segmentation for scaling
                seg_array = seg_true[0, 0].cpu().numpy()
                bbox = np.array(generate_spatial_bounding_box(seg_array[None]))
                bbox[[-2, -1]] = bbox[[-1, -2]] # swap w and d
                world_pixdim = [np.linalg.norm(world_affine[:3, i]) for i in range(3)]
                world_scale = bbox.ptp(0).max()

                world_params = (world_pixdim, world_scale)
            
            # Initialize MR-specific variables
            mr_masks = None
            
            if modal == 'mr':
                mr_masks = create_mr_ground_truth_mask(data, case_id, dataset_name, self.super_params)
                roi = roi_size[:2]
            else:
                roi = roi_size

            # Time UNet inference
            start_time = time.time()
            seg_pred = sliding_window_inference(img, roi_size=roi, sw_batch_size=8, predictor=encoder, overlap=0.5, mode="gaussian")
            timing_stats['unet_inference'] += time.time() - start_time

            if modal == "mr":
                seg_pred_ = rearrange(seg_pred, "d (b c) h w -> b c h w d", b=1)
                seg_true_ = rearrange(seg_true, "d (b c) h w -> b c h w d", b=1)
            else:
                seg_pred_ = seg_pred.clone()
                seg_true_ = seg_true.clone()
            seg_pred_onehot_unet = self.inference._convert_to_onehot(seg_pred_, self.super_params.num_classes, is_prediction=True)
            seg_true_onehot_unet = self.inference._convert_to_onehot(seg_true_, self.super_params.num_classes, is_prediction=False)
            unet_dice_metric(seg_pred_onehot_unet, seg_true_onehot_unet)

            seg_pred_ds = self.preprocessor._memory_efficient_post_transform(seg_pred, seg_true, modal, to_gpu=True, decoder_size=False)

            if export_ablation:
                seg_pred_ = rearrange(seg_pred, "d (b c) h w -> b c h w d", b=1) if modal == 'mr' else seg_pred
                seg_pred_ = torch.stack([self.pred_transform(i) for i in seg_pred_])
                seg_pred_ = rearrange(seg_pred_, "b c h w d -> d (b c) h w") if modal == 'mr' else seg_pred_
                pred_arr_unet = self.preprocessor._memory_efficient_post_transform(seg_pred_, seg_true, modal, to_gpu=True, decoder_size=True, nearest_interpolate=True)

                pred_arr_unet = pred_arr_unet[0, 0].cpu().numpy().astype("uint8")
                nii_img = nib.Nifti1Image(pred_arr_unet, np.eye(4))
                save_dir = ablation_dir.replace("MorphiNet/myo/f0", f"ResNet_before-{modal}/myo/f0")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{case_id}_pred.nii.gz")
                nib.save(nii_img, save_path)
                
                # Store case metadata for ablation metrics
                ablation_metrics['case_ids'].append(case_id)
                ablation_metrics['phases'].append(extract_phase_from_case_id(case_id, dataset_name))
                # Store UNet prediction for later comprehensive metrics computation
                setattr(self, '_temp_unet_pred', pred_arr_unet)

            # Time ResNet processing
            start_time = time.time()
            seg_pred_ds_decoder_size = self.preprocessor._memory_efficient_post_transform(seg_pred, seg_true, modal, to_gpu=True, decoder_size=True)
            binary_mask_pred = (torch.argmax(seg_pred_ds_decoder_size, dim=1, keepdim=True) > 0)
            mask = torch.zeros_like(binary_mask_pred)
            seg_np = binary_mask_pred[0, 0].cpu().numpy().astype(bool)
            dilated_np = binary_dilation(seg_np, iterations=20)
            mask[0, 0] = torch.from_numpy(dilated_np.astype(np.float32)).to(binary_mask_pred.device)
            mask[binary_mask_pred == 1] = 0

            resnet_output = self.decoder(seg_pred_ds)
            seg_pred_ds = seg_pred_ds_decoder_size + mask * resnet_output
            seg_pred_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ds])
            timing_stats['resnet_processing'] += time.time() - start_time

            seg_true_ds = self.preprocessor._generate_downsampled_gt(seg_true, modal, decoder_size=True)
            seg_pred_onehot_resnet = self.inference._convert_to_onehot(seg_pred_ds, self.super_params.num_classes, is_prediction=False)
            seg_true_onehot_resnet = self.inference._convert_to_onehot(seg_true_ds, self.super_params.num_classes, is_prediction=False)
            resnet_dice_metric(seg_pred_onehot_resnet, seg_true_onehot_resnet)

            if export_ablation:
                mask_arr = mask[0, 0].cpu().numpy().astype("uint8")
                nii_img = nib.Nifti1Image(mask_arr, np.eye(4))
                save_dir = ablation_dir.replace("MorphiNet/myo/f0", f"ResNet_mask-{modal}/myo/f0")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{case_id}_mask.nii.gz")
                nib.save(nii_img, save_path)

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
                
                # Compute comprehensive ablation metrics for this case
                if hasattr(self, '_temp_unet_pred'):
                    pred_arr_unet = self._temp_unet_pred

                    seg_np = (true_arr > 0).astype(bool)
                    dilated_mask = binary_dilation(seg_np, iterations=2)
                    pred_arr = pred_arr * dilated_mask
                    
                    # Dice scores for all labels
                    before_dice_scores = compute_ablation_dice_scores_all_labels(pred_arr_unet, true_arr, self.super_params.num_classes, self.inference)
                    after_dice_scores = compute_ablation_dice_scores_all_labels(pred_arr, true_arr, self.super_params.num_classes, self.inference)

                    # Hausdorff distances for all labels
                    before_hausdorff = compute_ablation_hausdorff_distances(pred_arr_unet, true_arr, self.super_params.num_classes, self.inference)
                    after_hausdorff = compute_ablation_hausdorff_distances(pred_arr, true_arr, self.super_params.num_classes, self.inference)
                    
                    # Volume differences between UNet and ResNet predictions
                    volume_diffs = compute_volume_differences(pred_arr_unet, pred_arr)
                    
                    # Store all metrics
                    for label in ['LV', 'MYO', 'RV']:
                        ablation_metrics['before_resnet_dice'][label].append(before_dice_scores[label])
                        ablation_metrics['after_resnet_dice'][label].append(after_dice_scores[label])
                        ablation_metrics['before_resnet_hausdorff'][label].append(before_hausdorff[label])
                        ablation_metrics['after_resnet_hausdorff'][label].append(after_hausdorff[label])
                        ablation_metrics['volume_differences'][label].append(volume_diffs[label])
                    
                    # Clean up temporary storage
                    delattr(self, '_temp_unet_pred')

            # Time distance transform
            start_time = time.time()
            foreground = seg_pred_ds > 0
            lv = (seg_pred_ds == 1)
            rv = (seg_pred_ds == 3)
            myo = (seg_pred_ds == 2)
            df_pred = torch.stack([
                distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0])
                for i in [foreground, lv, rv, myo]], dim=1)
            timing_stats['distance_transform'] += time.time() - start_time

            # Time template mesh warping
            start_time = time.time()
            template_mesh = self.mesh_ops.warp_template_mesh(
                F.interpolate(df_pred, size=(32, 32, 32), mode="trilinear", align_corners=False)
            )
            timing_stats['template_warping'] += time.time() - start_time

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
                        faces.to(torch.int32)
                    )

                    # Convert PyTorch3D Meshes -> Trimesh, apply Loop subdivision, then convert back
                    # Extract vertices/faces to numpy for Trimesh
                    _verts_np = template_mesh.verts_packed().detach().cpu().numpy()
                    _faces_np = template_mesh.faces_packed().detach().cpu().numpy()
                    _tri_mesh = trimesh.Trimesh(vertices=_verts_np, faces=_faces_np, process=False)
                    _tri_mesh = _tri_mesh.subdivide_loop(iterations=2)
                    _device = template_mesh.verts_packed().device
                    _verts_t = torch.from_numpy(_tri_mesh.vertices).to(device=_device, dtype=torch.float32)
                    _faces_t = torch.from_numpy(_tri_mesh.faces).to(device=_device, dtype=torch.int32)
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
                        faces.to(torch.int32)
                    )
                except Exception as e:
                    print(f"Warning: Warped template mesh export failed: {e}")

            # Time GSN processing
            start_time = time.time()
            all_levels = self.GSN(template_mesh, self.mesh_ops.subdivided_faces.faces_levels, df_pred, self.mesh_ops.subdivided_faces.labels_levels)
            timing_stats['gsn_processing'] += time.time() - start_time
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
                            faces.to(torch.int32)
                            )
                except Exception as e:
                    print(f"Warning: GSN multi-level export failed: {e}")

            # Create mesh variants for ablation study
            mesh_variants = create_template_mesh_variants(template_mesh, all_levels, self.super_params.template_mesh_dir, export_ablation)
            
            # Compute metrics for the final GSN mesh
            seg_true_ds = self.preprocessor._generate_downsampled_gt(seg_true, modal, decoder_size=128) # increase decoder size to 128 for more accurate measures
            dilated_mask = torch.zeros_like(seg_true_ds)
            seg_np = (seg_true_ds == 2)[0, 0].cpu().numpy().astype(bool)
            dilated_np = binary_dilation(seg_np, iterations=2)
            dilated_mask[0, 0] = torch.from_numpy(dilated_np.astype(np.float32)).to(seg_true_ds.device)
            voxeld_mesh = torch.cat([
                self.rasterizer(pred_mesh.verts_padded(), pred_mesh.faces_padded()) # ensure the raster_size in orchestrator.py is the same as the decoder size
                for pred_mesh in all_levels[-1]
            ], dim=0)
            voxeld_mesh_masked = voxeld_mesh * dilated_mask

            # Compute mesh metrics based on dataset type
            if modal == 'mr' and mr_masks is not None:
                # Separate combined masks for metrics from individual masks for visualization
                combined_masks = {k: v for k, v in mr_masks.items() if k in ['sax', 'lax']}
                individual_masks = mr_masks.get('individual_slices', {})

                # Process combined masks for metrics computation (backward compatibility)
                processed_combined_masks = {
                    key: self.preprocessor._generate_downsampled_gt(value, modal, decoder_size=128) 
                    for key, value in combined_masks.items()
                }

                case_mr_metrics = compute_mr_mesh_metrics(
                    voxeld_mesh, processed_combined_masks
                )
                # Store MR metrics for this case
                all_mr_metrics.append({
                    'case_id': case_id,
                    'metrics': case_mr_metrics
                })
                
                # Create MR slice visualization for specific cases
                if case_id in ['RT3DE_002-10_ES', 'patient002_frame12']:
                    print(f"Creating MR slice visualization for case: {case_id}")
                    export_path = export_dir.replace("myo/f0", "")
                    
                    # Process individual masks for visualization
                    processed_individual_masks = {
                        key: self.preprocessor._generate_downsampled_gt(value, modal, decoder_size=128)
                        for key, value in individual_masks.items()
                    } if individual_masks else {}

                    create_mr_slice_visualization(
                        case_id=case_id,
                        dataset_name=dataset_export_name,
                        predicted_mesh=all_levels[0],  # Use GSN first subdivision level
                        mr_masks=processed_individual_masks,
                        export_dir=export_path,
                        voxeld_mesh=voxeld_mesh  # Pass voxelized mesh for slice-plane comparisons
                    )

            msh_metric_batch(voxeld_mesh_masked, (seg_true_ds == 2).to(torch.float32))
            hausdorff_metric(voxeld_mesh_masked, (seg_true_ds == 2).to(torch.float32))

            # Store meshes and case IDs for comprehensive metrics computation
            all_case_ids.append(case_id)
            
            # Store all mesh variants for this case
            for variant_name, variant_mesh in mesh_variants.items():
                if variant_name not in all_mesh_variants:
                    all_mesh_variants[variant_name] = []
                all_mesh_variants[variant_name].append(variant_mesh)
            
            # Create ground truth mesh from segmentation using modular function
            gt_mesh = create_ground_truth_mesh(seg_true_ds)
            all_gt_meshes.append(gt_mesh)
            
            # Store ground truth segmentation for coordinate transformation
            all_seg_true.append(seg_true_ds)

            verts = all_levels[-1].verts_packed()
            faces = all_levels[-1].faces_packed()
            save_obj(os.path.join(export_dir, f"{case_id}.obj"),
                        verts.to(torch.float32), 
                        faces.to(torch.int32))

            # Increment case counter for timing
            timing_stats['case_count'] += 1

            # Create surface distance visualization for specific CT cases
            if modal == 'ct' and case_id in ['130352_CE-ED', 'ct_train_1007']:
                print(f"\nCreating surface distance visualization for case: {case_id}")
                export_path = export_dir.replace("myo/f0", "")
                create_surface_distance_visualization(
                    gt_mesh=gt_mesh,
                    pred_mesh=all_levels[0],  # Use GSN level 1 (first subdivision level)
                    case_id=case_id,
                    dataset_name=dataset_export_name,
                    export_dir=export_path,
                    world_params=world_params
                )

        # Compute basic metrics using modular function
        basic_metrics = compute_basic_metrics(unet_dice_metric, resnet_dice_metric, msh_metric_batch)
        
        if export_ablation:
            # Full ablation study - evaluate all mesh variants
            print("Starting ablation study evaluation for all mesh variants...")
            
            for variant_name, variant_meshes in all_mesh_variants.items():
                print(f"\nEvaluating variant: {variant_name}")
                
                # Create separate metric trackers for this variant
                variant_dice_metric = DiceMetric(include_background=False, reduction="none")
                variant_hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="none", percentile=75)
                
                # Re-compute voxelized meshes and metrics for this variant
                for i, variant_mesh in enumerate(variant_meshes):
                    # Apply the same voxelization and masking as the original pipeline
                    voxeld_mesh = torch.cat([
                        self.rasterizer(pred_mesh.verts_padded(), pred_mesh.faces_padded())
                        for pred_mesh in variant_mesh
                    ], dim=0)
                    
                    # Use the same segmentation ground truth and dilated mask from the original loop
                    seg_true_case = all_seg_true[i]
                    dilated_mask = torch.zeros_like(seg_true_case)
                    seg_np = (seg_true_case == 2)[0, 0].cpu().numpy().astype(bool)
                    dilated_np = binary_dilation(seg_np, iterations=2)
                    dilated_mask[0, 0] = torch.from_numpy(dilated_np.astype(np.float32)).to(seg_true_case.device)
                    
                    voxeld_mesh_masked = voxeld_mesh * dilated_mask
                    
                    # Compute metrics for this variant
                    variant_dice_metric(voxeld_mesh_masked, (seg_true_case == 2).to(torch.float32))
                    variant_hausdorff_metric(voxeld_mesh_masked, (seg_true_case == 2).to(torch.float32))
                
                # Compute comprehensive metrics for this variant
                dice_scores, hausdorff_scores, mesh_metrics = compute_comprehensive_metrics(
                    variant_dice_metric, variant_hausdorff_metric, 
                    variant_meshes, all_gt_meshes, all_seg_true, all_case_ids, dataset_name
                )
                
                # Export XLSX for this variant
                export_metrics_xlsx(
                    variant_name, all_case_ids, dice_scores, hausdorff_scores, 
                    mesh_metrics, export_dir, dataset_name
                )
            
            print("Ablation study evaluation completed for all variants.")
            
        else:
            # Standard evaluation - evaluate level_1, gsn_final variants
            for variant_name, variant_meshes in all_mesh_variants.items():
                if variant_name in ['gsn_level_1', 'gsn_final']:
                    dice_scores, hausdorff_scores, mesh_metrics = compute_comprehensive_metrics(
                        msh_metric_batch, hausdorff_metric, 
                        variant_meshes, all_gt_meshes, all_seg_true, all_case_ids, dataset_name
                    )
                    
                    # Export XLSX for this variant
                    export_metrics_xlsx(
                        variant_name, all_case_ids, dice_scores, hausdorff_scores, 
                        mesh_metrics, export_dir, dataset_name
                    )
        
        # Export ablation study metrics if collected
        if export_ablation and ablation_metrics['case_ids']:
            print(f"Exporting ablation study metrics for {len(ablation_metrics['case_ids'])} cases...")
            export_path = export_dir.replace("myo/f0", "")
            export_ablation_study_to_xlsx(ablation_metrics, export_path, dataset_export_name)
        
        # Export MR-specific metrics to XLSX files if available
        if modal == 'mr' and all_mr_metrics:
            export_path = export_dir.replace("myo/f0", "")
            export_mr_metrics_xlsx(all_mr_metrics, all_case_ids, export_path, dataset_export_name)
        
        # Print averaged timing results
        if timing_stats['case_count'] > 0:
            print("\n" + "="*60)
            print("TIMING SUMMARY (Average per case)")
            print("="*60)
            print(f"UNet Inference:      {timing_stats['unet_inference'] / timing_stats['case_count']:.4f}s")
            print(f"ResNet Processing:   {timing_stats['resnet_processing'] / timing_stats['case_count']:.4f}s")
            print(f"Distance Transform:  {timing_stats['distance_transform'] / timing_stats['case_count']:.4f}s")
            print(f"Template Warping:    {timing_stats['template_warping'] / timing_stats['case_count']:.4f}s")
            print(f"GSN Processing:      {timing_stats['gsn_processing'] / timing_stats['case_count']:.4f}s")
            print(f"Total per case:      {sum(timing_stats[k] for k in timing_stats if k != 'case_count') / timing_stats['case_count']:.4f}s")
            print(f"Number of cases:     {timing_stats['case_count']}")
            print("="*60)
        
        # Return the basic metrics (using final GSN mesh as reference)
        return {"mesh_dice": basic_metrics['mesh_dice'], 
                "unet_dice": basic_metrics['unet_dice'], 
                "resnet_dice": basic_metrics['resnet_dice']}


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