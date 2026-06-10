"""TestBench-compatible mesh and metric helpers for MorphiNet testing."""

import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import trimesh
from pytorch3d.ops import taubin_smoothing
from pytorch3d.structures import Meshes

from utils.mesh_metrics import compute_mesh_metrics
from utils.xlsx_exporter import export_metrics_to_xlsx
from utils.path_config import get_dataset_registry

DATASET_DATA_DIRS = {key: value["data_dir"] for key, value in get_dataset_registry().items()}
DATASET_DATA_DIRS["sct"] = DATASET_DATA_DIRS["scotheart"]

DATASET_MODAL = {
    "acdc": "mr",
    "cap": "mr",
    "mmwhs": "ct",
    "scotheart": "ct",
    "sct": "ct",
}


def _metric_values(metric) -> np.ndarray:
    """Return flattened per-case values from a MONAI metric accumulator."""
    aggregate = metric.aggregate()
    if isinstance(aggregate, torch.Tensor):
        values = aggregate.detach().cpu().float().numpy()
    else:
        values = np.asarray(aggregate, dtype=np.float32)
    return np.nan_to_num(values.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def _metric_mean(metric) -> float:
    values = _metric_values(metric)
    return float(values.mean()) if values.size else 0.0


def _single_part_mesh(mesh: Meshes) -> Meshes:
    """Collapse a possibly batched Meshes object into the MYO part expected by TestBench."""
    if len(mesh) == 1:
        return mesh
    return Meshes(verts=[mesh.verts_packed()], faces=[mesh.faces_packed()]).to(mesh.device)


def create_template_mesh_variants(template_mesh: Meshes, all_levels: List[Meshes], template_mesh_dir: str, export_ablation: bool) -> Dict[str, Meshes]:
    """Create TestBench-style mesh variants for standard and ablation evaluation."""
    variants: Dict[str, Meshes] = {}

    if all_levels:
        variants["gsn_level_1"] = all_levels[0]
        variants["gsn_final"] = all_levels[-1]

    if export_ablation:
        variants["level_0"] = template_mesh
        for index, mesh in enumerate(all_levels, start=1):
            variants[f"level_{index}"] = mesh
        try:
            tri_mesh = trimesh.Trimesh(
                vertices=template_mesh.verts_packed().detach().cpu().numpy(),
                faces=template_mesh.faces_packed().detach().cpu().numpy(),
                process=False,
            ).subdivide_loop(iterations=2)
            device = template_mesh.device
            variants["loop"] = Meshes(
                verts=[torch.as_tensor(tri_mesh.vertices, dtype=torch.float32, device=device)],
                faces=[torch.as_tensor(tri_mesh.faces, dtype=torch.int64, device=device)],
            )
        except Exception as exc:
            print(f"Warning: Could not create loop subdivision variant: {exc}")

    return variants


def compute_basic_metrics(unet_dice_metric, resnet_dice_metric, msh_metric_batch) -> Dict[str, float]:
    """Aggregate headline UNet, ResNet, and final mesh Dice metrics."""
    return {
        "unet_dice": _metric_mean(unet_dice_metric),
        "resnet_dice": _metric_mean(resnet_dice_metric),
        "mesh_dice": _metric_mean(msh_metric_batch),
    }


def _part_metric_dict(values: Iterable[float], part: str = "MYO") -> Dict[str, List[float]]:
    return {part: [float(v) for v in values]}


def _smooth_meshes(meshes: List[Meshes]) -> List[Meshes]:
    """Apply the Taubin smoothing used by TestBench before geometric metrics."""
    smoothed = []
    for mesh in meshes:
        try:
            smoothed.append(taubin_smoothing(_single_part_mesh(mesh), 0.77, -0.34, 10))
        except Exception as exc:
            print(f"Warning: Taubin smoothing failed, using unsmoothed mesh: {exc}")
            smoothed.append(_single_part_mesh(mesh))
    return smoothed


def compute_comprehensive_metrics(
    dice_metric,
    hausdorff_metric,
    pred_meshes: List[Meshes],
    gt_meshes: List[Meshes],
    seg_true_list: List[torch.Tensor],
    case_ids: List[str],
    dataset_name: str,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, Dict[str, np.ndarray]]]:
    """Compute TestBench-style MYO metrics for a mesh variant."""
    dice_scores = _part_metric_dict(_metric_values(dice_metric)[: len(case_ids)])
    hausdorff_scores = _part_metric_dict(_metric_values(hausdorff_metric)[: len(case_ids)])

    dataset_key = {"sct": "scotheart"}.get((dataset_name or "").lower(), (dataset_name or "").lower())
    data_dir = DATASET_DATA_DIRS.get(dataset_key)
    modal = DATASET_MODAL.get(dataset_key)

    try:
        mesh_metrics = compute_mesh_metrics(
            gt_meshes_parts=[_single_part_mesh(mesh) for mesh in gt_meshes],
            pred_meshes_parts=_smooth_meshes(pred_meshes),
            parts=["MYO"],
            seg_true_list=seg_true_list,
            case_ids=case_ids,
            dataset=dataset_key,
            modal=modal,
            data_dir=data_dir,
        )
    except Exception as exc:
        print(f"Warning: Comprehensive mesh metrics failed: {exc}")
        zeros = np.zeros(len(case_ids), dtype=np.float32)
        mesh_metrics = {
            "MYO": {
                "asd": zeros,
                "aspect_ratio": zeros,
                "skew": zeros,
                "jacobian": zeros,
                "jacobian_ratio_low": zeros,
                "normal_consistency": zeros,
                "nm_face_ratio": zeros,
                "chamfer": zeros,
            }
        }

    return dice_scores, hausdorff_scores, mesh_metrics


def export_metrics_xlsx(
    variant_name: str,
    case_ids: List[str],
    dice_scores: Dict[str, List[float]],
    hausdorff_scores: Dict[str, List[float]],
    mesh_metrics: Dict[str, Dict[str, np.ndarray]],
    export_dir: str,
    dataset_name: str,
) -> str:
    """Export a TestBench-compatible workbook for one MorphiNet variant."""
    root_dir = export_dir.replace(os.path.join("myo", "f0"), "").rstrip(os.sep)
    os.makedirs(root_dir, exist_ok=True)
    output_path = os.path.join(root_dir, f"{dataset_name}_{variant_name}_metrics.xlsx")
    export_metrics_to_xlsx(
        case_ids=case_ids,
        dice_scores=dice_scores,
        hausdorff_scores=hausdorff_scores,
        mesh_metrics=mesh_metrics,
        output_path=output_path,
        method_name=f"MorphiNet_{variant_name}",
    )
    return output_path


def create_ground_truth_mesh(seg_true_ds: torch.Tensor) -> Meshes:
    """Create a MYO mesh in normalized TestBench/MorphiNet coordinates from labels."""
    volume = (seg_true_ds.detach().cpu().numpy().squeeze() == 2).astype(np.uint8)
    device = seg_true_ds.device

    if volume.ndim != 3 or not np.any(volume):
        verts = torch.zeros((0, 3), dtype=torch.float32, device=device)
        faces = torch.zeros((0, 3), dtype=torch.int64, device=device)
        return Meshes(verts=[verts], faces=[faces])

    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(volume)
    mesh.vertices = mesh.vertices[:, [1, 0, 2]]
    center = mesh.bounding_box.centroid
    extent = mesh.bounding_box.extents.max()
    mesh.apply_translation(-center)
    if extent > 0:
        mesh.apply_scale(2.0 / extent)
    try:
        from trimesh.smoothing import filter_laplacian

        mesh = filter_laplacian(mesh, lamb=0.13, iterations=10)
    except Exception:
        pass

    verts = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(mesh.faces, dtype=torch.int64, device=device)
    return Meshes(verts=[verts], faces=[faces])
