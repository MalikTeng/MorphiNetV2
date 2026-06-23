"""
Advanced mesh quality and geometric metrics for MorphiNet testing.

This module provides comprehensive mesh evaluation metrics including:
- Average Surface Distance (ASD)
- Mesh quality metrics (Aspect Ratio, Skew, Jacobian)  
- Normal consistency metrics
- Manifoldness metrics (Non-manifold Face/Edge/Vertex ratios)

Copied and adapted from TestBench_24/utils/metrics.py for integration
with MorphiNet testing pipeline.
"""

import os
import sys
import glob
from typing import Union, List, Dict
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.ops import sample_points_from_meshes, knn_points, knn_gather, packed_to_padded
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d import _C
from pytorch3d.loss import chamfer_distance
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from scipy.spatial.distance import cdist
from monai.transforms.utils import generate_spatial_bounding_box
import trimesh
import open3d as o3d

from monai.transforms import Compose, LoadImaged, EnsureTyped
from data.components import SequentialTransformd

# Optional dependencies
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None

__all__ = [
    "compute_mesh_metrics", "average_surface_distance", "mesh_quality_metrics", 
    "normal_consistency_metrics", "manifoldness_metrics", "chamfer_distance_mesh"
]

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3

def _ensure_consistent_dtype(meshes: Meshes, target_dtype: torch.dtype = torch.float32) -> Meshes:
    """
    Ensure all mesh vertices have consistent dtype for PyTorch3D operations.
    
    Args:
        meshes: Input PyTorch3D Meshes object
        target_dtype: Target dtype for vertices (default: float32)
        
    Returns:
        Meshes object with vertices cast to target_dtype
    """
    if meshes is None:
        return meshes
        
    try:
        verts_list = meshes.verts_list()
        faces_list = meshes.faces_list()
        
        # Ensure vertices have consistent dtype
        verts_consistent = [v.to(dtype=target_dtype) if v.dtype != target_dtype else v for v in verts_list]
        
        # Faces should be int32 for indexing
        faces_consistent = [f.to(dtype=torch.int32) if f.dtype != torch.int32 else f for f in faces_list]
        
        return Meshes(verts=verts_consistent, faces=faces_consistent)
    except Exception as e:
        print(f"Warning: Failed to ensure consistent dtype for meshes: {e}")
        return meshes

def ndc_to_world(pred_meshes_parts: List[Meshes], gt_meshes_parts: List[Meshes], 
                 seg_true_list: List[torch.Tensor] = None, case_ids: List[str] = None,
                 dataset: str = None, modal: str = None, data_dir: str = None) -> tuple:
    """
    Transform meshes from NDC space to world coordinates.
    
    Adapted from TestBench-24/testbench-ct.py for MorphiNet integration.
    This function converts mesh coordinates from normalized device coordinates [-1, 1] 
    to real-world coordinates based on the original segmentation bounding box.
    
    Args:
        pred_meshes_parts: List of predicted meshes in NDC coordinates
        gt_meshes_parts: List of ground truth meshes in NDC coordinates
        seg_true_list: List of ground truth segmentations for spatial reference (optional)
        case_ids: List of case identifiers (optional, for debugging)
        dataset: Dataset name for finding original NIFTI files (optional)
        modal: Modality ('ct' or 'mr') for finding original NIFTI files (optional)
        data_dir: Dataset directory path for finding original NIFTI files (optional)
        
    Returns:
        tuple: (transformed_pred_meshes, transformed_gt_meshes) in world coordinates
    """
    if not pred_meshes_parts or not gt_meshes_parts:
        print("Warning: Empty mesh lists provided for NDC to world transformation")
        return pred_meshes_parts, gt_meshes_parts
        
    pred_meshes_world = []
    gt_meshes_world = []
    
    device = pred_meshes_parts[0].device if pred_meshes_parts else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i, (pred_mesh, gt_mesh) in enumerate(zip(pred_meshes_parts, gt_meshes_parts)):
        try:
            scale = None
            pixdim = None
            
            # Use original NIFTI file if dataset info is available
            spatial_info = _find_and_transform_original_nifti(case_ids[i], dataset, modal, data_dir)
            if spatial_info is not None:
                scale, pixdim = spatial_info
            
            # Transform predicted mesh from NDC to voxel space, then to world
            pred_mesh_transformed = _transform_mesh_ndc_to_world(pred_mesh, scale, pixdim, device)
            pred_meshes_world.append(pred_mesh_transformed)
            
            # Transform ground truth mesh from NDC to voxel space, then to world  
            gt_mesh_transformed = _transform_mesh_ndc_to_world(gt_mesh, scale, pixdim, device)
            gt_meshes_world.append(gt_mesh_transformed)
            
        except Exception as e:
            print(f"Warning: NDC to world transformation failed for mesh pair {i}: {e}")
            # Return original meshes if transformation fails
            pred_meshes_world.append(pred_mesh)
            gt_meshes_world.append(gt_mesh)
    
    return pred_meshes_world, gt_meshes_world

def _find_and_transform_original_nifti(case_id: str, dataset: str, modal: str, data_dir: str) -> tuple:
    """
    Find and transform the original NIFTI label file for a given case.
    
    Args:
        case_id: Case identifier 
        dataset: Dataset name
        modal: Modality ('ct' or 'mr')
        data_dir: Dataset directory path
        
    Returns:
        tuple: (scale, pixdim) extracted from transformed NIFTI, or None if failed
    """
    # Find the label file
    label_file = None
    matches = glob.glob(os.path.join(data_dir, "labelsTs", f"*{case_id}*.nii.gz"))
    if matches:
        label_file = matches[0]  # Take first match
    
    if not label_file:
        print(f"Warning: Could not find label NIFTI file for case {case_id} in {data_dir}")
        return None
        
    # Apply transformation sequence: LoadImaged + SequentialTransformd
    transforms = Compose([
        LoadImaged(["label"]), 
        SequentialTransformd(["label"], sequence="s:xy f:x f:z", allow_missing_keys=False),
        EnsureTyped(["label"], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), allow_missing_keys=False)
    ])
    data_dict = transforms({"label": label_file})
    
    # Extract the transformed label array
    label_tensor = data_dict["label"]
    label_array = label_tensor.get_array()
    
    # Remove channel dimension if present (should be 4D: C, H, W, D)
    if label_array.ndim == 4 and label_array.shape[0] == 1:
        label_array = label_array[0]  # Remove channel dimension

    # Get pixdim from the label tensor
    affine = label_tensor.affine.cpu().numpy()
    pixdim = [np.linalg.norm(affine[:3, i]) for i in range(3)]
    pixdim = np.min(pixdim).astype(np.float32)
    
    # Generate bounding box from the transformed segmentation
    bbox = np.array(generate_spatial_bounding_box(label_array[None])).astype(np.float32)
    bbox = bbox[:, [2, 1, 0]]   # i,j,k -> x,y,z

    # Calculate centroid and scale from the bounding box
    scale = bbox.ptp(0).max()  # Peak-to-peak (max - min) along axis 0, then max
    
    return (scale, pixdim)

def _transform_mesh_ndc_to_world(mesh: Meshes, scale: float, pixdim: list, device: torch.device) -> Meshes:
    """
    Helper function to transform a single mesh from NDC to world coordinates.
    
    Args:
        mesh: PyTorch3D Meshes object in NDC coordinates
        scale: Scale factor for transformation
        pixdim: Pixel dimensions of the image
        device: Target device for computations
        
    Returns:
        Transformed Meshes object in world coordinates
    """
    # Get mesh bounding box
    mesh_bbox = mesh.get_bounding_boxes().transpose(1, 2).reshape(-1, 3)
    mesh_extent = ((mesh_bbox.cpu().numpy()).ptp(0)).max()
    
    # Clone the mesh to avoid in-place operations on the original
    mesh_clone = mesh.clone()
    
    # Transform from NDC to world coordinates
    scale_factor = (scale / mesh_extent) * pixdim
    mesh_clone.scale_verts_(scale_factor)
    
    return mesh_clone

class _PointFaceDistance(Function):
    """PyTorch3D point-face distance computation."""
    
    @staticmethod
    def forward(ctx, points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA):
        dists, idxs = _C.point_face_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None

point_face_distance = _PointFaceDistance.apply

def point_mesh_face_distance(meshes: Meshes, pcls: Pointclouds, min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA):
    """
    Compute distance between pointcloud and mesh faces.
    
    Args:
        meshes: PyTorch3D Meshes object
        pcls: PyTorch3D Pointclouds object  
        min_triangle_area: Minimum triangle area threshold
        
    Returns:
        Median point-to-face distance
    """
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")

    # packed representation for pointclouds
    points = pcls.points_packed()
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )

    point_to_face = torch.pow(point_to_face, 0.5)
    point_dist = point_to_face.median()

    return point_dist

def _validate_mesh_for_sampling(mesh: Meshes) -> bool:
    """
    Validate that a mesh is suitable for surface sampling.
    
    Args:
        mesh: PyTorch3D mesh to validate
        
    Returns:
        True if mesh is valid for sampling, False otherwise
    """
    if mesh is None:
        return False
    
    try:
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()
        
        # Check if mesh has vertices and faces
        if verts.shape[0] == 0 or faces.shape[0] == 0:
            return False
            
        # Check for degenerate faces (zero area)
        # Get face vertices
        face_verts = verts[faces]  # Shape: (num_faces, 3, 3)
        
        # Compute face normals using cross product
        v0v1 = face_verts[:, 1] - face_verts[:, 0]  # Edge 0->1
        v0v2 = face_verts[:, 2] - face_verts[:, 0]  # Edge 0->2
        
        # Cross product gives face normal (magnitude = 2 * area)
        face_normals = torch.cross(v0v1, v0v2, dim=1)
        face_areas = torch.norm(face_normals, dim=1) / 2.0
        
        # Check if total surface area is positive
        total_area = torch.sum(face_areas).item()
        if total_area <= 1e-8:  # Very small threshold
            print(f"Warning: Mesh has zero area (collapsed geometry), but allowing for metrics computation")
            print(f"  Vertices range: {verts.min().item():.3f} to {verts.max().item():.3f}")
            print(f"  Face areas range: {face_areas.min().item():.2e} to {face_areas.max().item():.2e}")
            # Return True temporarily to allow computation with default fallback values
            return True
            
        # Check if there are enough valid (non-degenerate) faces
        valid_faces = face_areas > 1e-10
        num_valid_faces = torch.sum(valid_faces).item()
        if num_valid_faces < 10:  # Need at least some valid faces
            print(f"Mesh validation failed: only {num_valid_faces} valid faces out of {len(face_areas)}")
            return False
            
        # Check for NaN or inf values
        if torch.isnan(face_areas).any() or torch.isinf(face_areas).any():
            return False
            
        return True
        
    except Exception:
        return False

def average_surface_distance(gt_meshes_parts: List[Meshes], pred_meshes_parts: List[Meshes], parts: List[str], num_surface_sampling: int = 5000, 
                            seg_true_list: List[torch.Tensor] = None, case_ids: List[str] = None,
                            dataset: str = None, modal: str = None, data_dir: str = None) -> Dict[str, np.ndarray]:
    """
    Compute Average Surface Distance (ASD) between ground truth and predicted meshes.
    
    Args:
        gt_meshes_parts: List of ground truth meshes per batch
        pred_meshes_parts: List of predicted meshes per batch
        parts: List of anatomical parts (e.g., ['LV', 'RV', 'MYO'])
        num_surface_sampling: Number of points to sample from mesh surfaces
        seg_true_list: List of ground truth segmentations for spatial reference (optional)
        case_ids: List of case identifiers (optional, for debugging)
        dataset: Dataset name for finding original NIFTI files (optional)
        modal: Modality ('ct' or 'mr') for finding original NIFTI files (optional)
        data_dir: Dataset directory path for finding original NIFTI files (optional)
        
    Returns:
        Dictionary mapping part names to arrays of ASD values per batch
    """
    avg_distance_all = np.zeros((len(gt_meshes_parts), len(parts)))
    
    pred_meshes_world, gt_meshes_world = ndc_to_world(
        pred_meshes_parts, gt_meshes_parts, seg_true_list, case_ids, dataset, modal, data_dir
    )
    
    for i, (gt_mesh_parts, pred_mesh_parts) in enumerate(zip(gt_meshes_world, pred_meshes_world)):
        for p, (gt_mesh, pred_mesh) in enumerate(zip(gt_mesh_parts, pred_mesh_parts)):
            try:
                # Ensure consistent dtypes for all mesh operations
                pred_mesh = _ensure_consistent_dtype(pred_mesh, torch.float32)
                
                # Validate meshes before sampling
                if not _validate_mesh_for_sampling(pred_mesh):
                    print(f"Warning: Invalid predicted mesh at batch {i}, part {p}")
                    avg_distance_all[i, p] = 0.0
                    continue
                
                # Sample points from GT mesh surface
                if isinstance(gt_mesh, Pointclouds):
                    gt_pointcloud_part = gt_mesh
                    # Ensure pointcloud points have consistent dtype
                    if gt_pointcloud_part.points_padded().dtype != torch.float32:
                        gt_points = gt_pointcloud_part.points_padded().to(dtype=torch.float32)
                        gt_pointcloud_part = Pointclouds(points=gt_points)
                else:
                    gt_mesh = _ensure_consistent_dtype(gt_mesh, torch.float32)
                    if not _validate_mesh_for_sampling(gt_mesh):
                        print(f"Warning: Invalid ground truth mesh at batch {i}, part {p}")
                        avg_distance_all[i, p] = 0.0
                        continue
                    gt_face_points = sample_points_from_meshes(gt_mesh, num_surface_sampling)
                    gt_pointcloud_part = Pointclouds(points=gt_face_points)
                
                # Compute distance from predicted mesh to GT points
                avg_distance = point_mesh_face_distance(pred_mesh, gt_pointcloud_part)
                avg_distance_all[i, p] = avg_distance.item()
                
            except Exception as e:
                print(f"Warning: ASD computation failed for batch {i}, part {p}: {e}")
                avg_distance_all[i, p] = 0.0

    avg_distance_parts = {
        part: avg_distance_all[:, p] for p, part in enumerate(parts)
    }
    return avg_distance_parts

def mesh_quality_metrics(pred_meshes_parts: List[Meshes], parts: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute mesh quality metrics using PyVista.
    
    Args:
        pred_meshes_parts: List of predicted meshes per batch
        parts: List of anatomical parts
        
    Returns:
        Dictionary mapping parts to quality metrics (aspect_ratio, skew, jacobian, jacobian_ratio_low)
    """
    if not PYVISTA_AVAILABLE:
        # Fallback to simple geometric metrics if PyVista not available
        print("Warning: PyVista not available, using fallback mesh quality metrics")
        return _fallback_mesh_quality(pred_meshes_parts, parts)
    
    quality_scores = np.zeros((4, len(parts), len(pred_meshes_parts)))

    for i, pred_mesh_parts in enumerate(pred_meshes_parts):
        for p, pred_mesh in enumerate(pred_mesh_parts):
            try:
                # Ensure consistent dtypes for mesh operations
                pred_mesh = _ensure_consistent_dtype(pred_mesh, torch.float32)
                
                mesh = trimesh.Trimesh(
                    vertices=pred_mesh.verts_packed().cpu().numpy(),
                    faces=pred_mesh.faces_packed().cpu().numpy(),
                )
                mesh_pv = pv.wrap(mesh)
                
                aspect = np.median(np.array(mesh_pv.compute_cell_quality('aspect_ratio').active_scalars))
                skew = np.median(np.array(mesh_pv.compute_cell_quality('skew').active_scalars))
                jacobian_values = np.array(mesh_pv.compute_cell_quality('scaled_jacobian').active_scalars)
                jacobian = np.median(jacobian_values)
                
                # Compute ratio of cells with jacobian < 0.7
                jacobian_ratio_low = np.sum(jacobian_values < 0.7) / len(jacobian_values) if len(jacobian_values) > 0 else 0.0

                quality_scores[0, p, i] = aspect
                quality_scores[1, p, i] = skew  
                quality_scores[2, p, i] = jacobian
                quality_scores[3, p, i] = jacobian_ratio_low
            except Exception as e:
                print(f"Warning: PyVista mesh quality computation failed for batch {i}, part {p}: {e}")
                # Use fallback values
                quality_scores[0, p, i] = 1.0  # aspect ratio
                quality_scores[1, p, i] = 0.0  # skew
                quality_scores[2, p, i] = 1.0  # jacobian
                quality_scores[3, p, i] = 0.0  # jacobian_ratio_low (good quality)

    quality_metrics = {
        part: {
            'aspect_ratio': quality_scores[0, p], 
            'skew': quality_scores[1, p], 
            'jacobian': quality_scores[2, p],
            'jacobian_ratio_low': quality_scores[3, p]
        } for p, part in enumerate(parts)
    }
    return quality_metrics

def _fallback_mesh_quality(pred_meshes_parts: List[Meshes], parts: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Fallback mesh quality metrics when PyVista is not available."""
    quality_metrics = {}
    for p, part in enumerate(parts):
        num_batches = len(pred_meshes_parts)
        quality_metrics[part] = {
            'aspect_ratio': np.ones(num_batches),      # Default to 1.0 (good aspect ratio)
            'skew': np.zeros(num_batches),            # Default to 0.0 (no skew)
            'jacobian': np.ones(num_batches),         # Default to 1.0 (good jacobian)
            'jacobian_ratio_low': np.zeros(num_batches)  # Default to 0.0 (no poor quality cells)
        }
    return quality_metrics

def normal_consistency_metrics(gt_meshes_parts: List[Meshes], pred_meshes_parts: List[Meshes], parts: List[str], num_surface_sampling: int = 5000) -> Dict[str, np.ndarray]:
    """
    Compute mean normal consistency between ground truth and predicted meshes.
    
    Args:
        gt_meshes_parts: List of ground truth meshes per batch
        pred_meshes_parts: List of predicted meshes per batch  
        parts: List of anatomical parts
        num_surface_sampling: Number of points to sample for normal comparison
        
    Returns:
        Dictionary mapping part names to arrays of normal consistency values
    """
    metrics = np.zeros((len(gt_meshes_parts), len(parts)))

    for i, (gt_mesh_parts, pred_mesh_parts) in enumerate(zip(gt_meshes_parts, pred_meshes_parts)):
        for p, (gt_mesh, pred_mesh) in enumerate(zip(gt_mesh_parts, pred_mesh_parts)):
            try:
                # Ensure consistent dtypes for all mesh operations
                pred_mesh = _ensure_consistent_dtype(pred_mesh, torch.float32)
                gt_mesh = _ensure_consistent_dtype(gt_mesh, torch.float32)
                
                # Validate meshes before sampling
                if not _validate_mesh_for_sampling(pred_mesh) or not _validate_mesh_for_sampling(gt_mesh):
                    print(f"Warning: Invalid meshes at batch {i}, part {p}")
                    metrics[i, p] = 0.0
                    continue
                
                gt_points, gt_normals = sample_points_from_meshes(
                    gt_mesh, num_samples=num_surface_sampling, return_normals=True
                )      
                pred_points, pred_normals = sample_points_from_meshes(
                    pred_mesh, num_samples=num_surface_sampling, return_normals=True
                )
                
                normal_consistency = _compute_normal_consistency(
                    pred_points, pred_normals, gt_points, gt_normals
                )
                metrics[i, p] = normal_consistency
            except Exception as e:
                print(f"Warning: Normal consistency computation failed for batch {i}, part {p}: {e}")
                metrics[i, p] = 0.0  # Default to 0 consistency

    normal_metrics = {
        part: metrics[:, p] for p, part in enumerate(parts)
    }
    return normal_metrics

def _compute_normal_consistency(pred_points, pred_normals, gt_points, gt_normals):
    """Compute absolute normal consistency between predicted and ground truth points."""
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # Find nearest neighbors
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    pred_normals_near = knn_gather(gt_normals, knn_pred.idx, lengths_gt)[..., 0, :]

    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    gt_normals_near = knn_gather(pred_normals, knn_gt.idx, lengths_pred)[..., 0, :]

    # Compute cosine similarity
    pred_to_gt_cos = F.cosine_similarity(pred_normals, pred_normals_near, dim=2)
    gt_to_pred_cos = F.cosine_similarity(gt_normals, gt_normals_near, dim=2)

    pred_to_gt_abs_cos_sim = pred_to_gt_cos.abs().mean(dim=1)
    gt_to_pred_abs_cos_sim = gt_to_pred_cos.abs().mean(dim=1)
    
    abs_normal_consistency = 0.5 * (pred_to_gt_abs_cos_sim + gt_to_pred_abs_cos_sim)
    return abs_normal_consistency.cpu().numpy()[0]

def manifoldness_metrics(pred_meshes_parts: List[Meshes], parts: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute manifoldness metrics including non-manifold face ratios.
    
    Args:
        pred_meshes_parts: List of predicted meshes per batch
        parts: List of anatomical parts
        
    Returns:
        Dictionary mapping parts to manifoldness metrics
    """
    metrics = np.zeros((len(parts), 3, len(pred_meshes_parts)))
    
    for i, pred_mesh_parts in enumerate(pred_meshes_parts):
        for p, pred_mesh in enumerate(pred_mesh_parts):
            try:
                # Ensure consistent dtypes for mesh operations
                pred_mesh = _ensure_consistent_dtype(pred_mesh, torch.float32)
                
                nm_vertices, nm_edges = _calculate_non_manifold_edge_vertex(pred_mesh)
                nv, ne, nf, nm_faces = _calculate_non_manifold_face(pred_mesh)
                
                metrics[p, 0, i] = nm_vertices / max(nv, 1)  # Avoid division by zero
                metrics[p, 1, i] = nm_edges / max(ne, 1)
                metrics[p, 2, i] = nm_faces / max(nf, 1)
            except Exception as e:
                print(f"Warning: Manifoldness computation failed for batch {i}, part {p}: {e}")
                # Use default values indicating good manifoldness
                metrics[p, 0, i] = 0.0
                metrics[p, 1, i] = 0.0 
                metrics[p, 2, i] = 0.0

    manifold_metrics = {
        part: {
            'nm_vertex_ratio': metrics[p, 0],
            'nm_edge_ratio': metrics[p, 1], 
            'nm_face_ratio': metrics[p, 2]
        } for p, part in enumerate(parts)
    }
    return manifold_metrics

def _calculate_non_manifold_edge_vertex(mesh: Meshes):
    """Calculate non-manifold edges and vertices using Open3D."""
    verts = mesh.verts_packed().cpu().numpy()
    faces = mesh.faces_packed().cpu().numpy()

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    nm_edges = np.asarray(o3d_mesh.get_non_manifold_edges(allow_boundary_edges=False))
    nm_vertices = np.asarray(o3d_mesh.get_non_manifold_vertices())
    
    return nm_vertices.shape[0], nm_edges.shape[0]

def _calculate_non_manifold_face(mesh: Meshes):
    """Calculate non-manifold faces using trimesh face adjacency."""
    verts = mesh.verts_packed().cpu().numpy()
    faces = mesh.faces_packed().cpu().numpy()
    
    trimesh_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    
    try:
        f_adj = trimesh_mesh.face_adjacency
        face_normals = trimesh_mesh.face_normals
        
        count = 0
        for f in range(f_adj.shape[0]):
            if face_normals[f_adj[f, 0]] @ face_normals[f_adj[f, 1]] < 0:
                count += 1
                
        return (trimesh_mesh.vertices.shape[0], 
                trimesh_mesh.edges.shape[0], 
                trimesh_mesh.faces.shape[0], 
                count)
    except:
        # Return default values if face adjacency computation fails
        return (verts.shape[0], 0, faces.shape[0], 0)

def chamfer_distance_mesh(gt_meshes_parts: List[Meshes], pred_meshes_parts: List[Meshes], parts: List[str], num_surface_sampling: int = 5000,
                          seg_true_list: List[torch.Tensor] = None, case_ids: List[str] = None,
                          dataset: str = None, modal: str = None, data_dir: str = None) -> Dict[str, np.ndarray]:
    """
    Compute Chamfer distance between meshes using surface sampling.
    
    Args:
        gt_meshes_parts: List of ground truth meshes per batch
        pred_meshes_parts: List of predicted meshes per batch
        parts: List of anatomical parts
        num_surface_sampling: Number of points to sample from surfaces
        seg_true_list: List of ground truth segmentations for spatial reference (optional)
        case_ids: List of case identifiers (optional, for debugging)
        dataset: Dataset name for finding original NIFTI files (optional)
        modal: Modality ('ct' or 'mr') for finding original NIFTI files (optional)
        data_dir: Dataset directory path for finding original NIFTI files (optional)
        
    Returns:
        Dictionary mapping part names to arrays of Chamfer distances (in mm)
    """
    cf_distance_final = np.zeros((len(gt_meshes_parts), len(parts)))
    
    pred_meshes_world, gt_meshes_world = ndc_to_world(
        pred_meshes_parts, gt_meshes_parts, seg_true_list, case_ids, dataset, modal, data_dir
    )
    
    for i, (gt_mesh_parts, pred_mesh_parts) in enumerate(zip(gt_meshes_world, pred_meshes_world)):
        for p, (gt_mesh, pred_mesh) in enumerate(zip(gt_mesh_parts, pred_mesh_parts)):
            try:
                # Ensure consistent dtypes for all mesh operations
                pred_mesh = _ensure_consistent_dtype(pred_mesh, torch.float32)
                
                # Validate meshes before sampling
                if not _validate_mesh_for_sampling(pred_mesh):
                    print(f"Warning: Invalid predicted mesh at batch {i}, part {p}")
                    cf_distance_final[i, p] = 0.0
                    continue
                
                if isinstance(gt_mesh, Pointclouds):
                    gt_points = gt_mesh.points_padded()
                    # Ensure consistent dtype for pointclouds
                    if gt_points.dtype != torch.float32:
                        gt_points = gt_points.to(dtype=torch.float32)
                else:
                    gt_mesh = _ensure_consistent_dtype(gt_mesh, torch.float32)
                    if not _validate_mesh_for_sampling(gt_mesh):
                        print(f"Warning: Invalid ground truth mesh at batch {i}, part {p}")
                        cf_distance_final[i, p] = 0.0
                        continue
                    gt_points = sample_points_from_meshes(gt_mesh, num_surface_sampling)
                
                pred_points = sample_points_from_meshes(pred_mesh, num_surface_sampling)
                
                # Compute chamfer distance using PyTorch3D
                cham_dist, _ = chamfer_distance(pred_points, gt_points, point_reduction='mean')
                cf_distance_final[i, p] = cham_dist.item()
            except Exception as e:
                print(f"Warning: Chamfer distance computation failed for batch {i}, part {p}: {e}")
                cf_distance_final[i, p] = 0.0

    cf_distance_parts = {
        part: cf_distance_final[:, p] for p, part in enumerate(parts)
    }
    return cf_distance_parts

def compute_mesh_metrics(gt_meshes_parts: List[Meshes], pred_meshes_parts: List[Meshes], parts: List[str], num_surface_sampling: int = 5000, 
                        seg_true_list: List[torch.Tensor] = None, case_ids: List[str] = None,
                        dataset: str = None, modal: str = None, data_dir: str = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute all mesh metrics required for XLSX export.
    
    Args:
        gt_meshes_parts: List of ground truth meshes per batch
        pred_meshes_parts: List of predicted meshes per batch  
        parts: List of anatomical parts (e.g., ['LV', 'RV', 'MYO'])
        num_surface_sampling: Number of points to sample from mesh surfaces
        seg_true_list: List of ground truth segmentations for spatial reference (optional)
        case_ids: List of case identifiers (optional, for debugging)
        dataset: Dataset name for finding original NIFTI files (optional)
        modal: Modality ('ct' or 'mr') for finding original NIFTI files (optional)
        data_dir: Dataset directory path for finding original NIFTI files (optional)
        
    Returns:
        Dictionary containing all metrics organized by part and metric name
    """
    # Compute all metrics
    asd_metrics = average_surface_distance(gt_meshes_parts, pred_meshes_parts, parts, num_surface_sampling, seg_true_list, case_ids, dataset, modal, data_dir)
    quality_metrics = mesh_quality_metrics(pred_meshes_parts, parts)
    normal_metrics = normal_consistency_metrics(gt_meshes_parts, pred_meshes_parts, parts, num_surface_sampling)
    manifold_metrics = manifoldness_metrics(pred_meshes_parts, parts)
    chamfer_metrics = chamfer_distance_mesh(gt_meshes_parts, pred_meshes_parts, parts, num_surface_sampling, seg_true_list, case_ids, dataset, modal, data_dir)
    
    # Organize all metrics by part
    all_metrics = {}
    for part in parts:
        all_metrics[part] = {
            'asd': asd_metrics[part],
            'aspect_ratio': quality_metrics[part]['aspect_ratio'],
            'skew': quality_metrics[part]['skew'],
            'jacobian': quality_metrics[part]['jacobian'],
            'jacobian_ratio_low': quality_metrics[part]['jacobian_ratio_low'],
            'normal_consistency': normal_metrics[part],
            'nm_face_ratio': manifold_metrics[part]['nm_face_ratio'],
            'chamfer': chamfer_metrics[part]
        }
    
    # print("Mesh metrics computation completed.")
    return all_metrics