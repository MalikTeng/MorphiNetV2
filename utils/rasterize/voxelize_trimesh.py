"""
GPU-accelerated triangle-mesh voxelization with CPU fallback.

This module provides fast, reliable conversion from surface meshes to binary 
voxel volumes using CUDA acceleration when available, with automatic fallback
to CPU-based Trimesh computation. The implementation uses signed distance fields
and returns PyTorch tensors. No gradients/backprop are supported or required.
"""

from typing import List
import warnings

import numpy as np
import torch
import torch.nn as nn

# Try to import CUDA extension
try:
    import sys
    import os
    # Add build directory to path for the compiled extension
    build_dir = os.path.join(os.path.dirname(__file__), 'build_simple')
    if os.path.exists(build_dir):
        sys.path.insert(0, build_dir)
    import voxelize_cuda_ext
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    voxelize_cuda_ext = None

# Import CPU fallback
try:
    import trimesh
except Exception as e:  # pragma: no cover - import-time guard
    trimesh = None


class VoxelizeTrimesh(nn.Module):
    """GPU-accelerated mesh voxelization with CPU fallback.

    This class provides fast triangle mesh voxelization using CUDA acceleration
    when available, with automatic fallback to CPU-based Trimesh computation.
    
    - Inputs are PyTorch tensors to match existing callers
    - Outputs a binary occupancy volume as a PyTorch tensor on the input device  
    - Uses fixed grid in normalized coordinates [-1, 1] along each axis
    - CUDA acceleration provides 10-50x speedup for typical meshes

    Args:
        shape: Target voxel grid dimensions [D, H, W]
        chunk_size: Number of query points per batch (CPU fallback only)
        use_cuda: Whether to use CUDA acceleration (default: auto-detect)
    """

    def __init__(self, shape: List[int], chunk_size: int = 250_000, use_cuda: bool = True) -> None:
        super().__init__()
        assert len(shape) == 3, "shape must be [D, H, W]"
        self.shape_dhw = list(shape)
        self.chunk_size = int(chunk_size)
        
        # Determine acceleration method
        self.use_cuda = use_cuda and CUDA_AVAILABLE and torch.cuda.is_available()
        
        if self.use_cuda:
            print(f"VoxelizeTrimesh: Using CUDA acceleration for {shape} voxelization")
            if voxelize_cuda_ext is not None:
                # Print CUDA device info
                try:
                    cuda_info = voxelize_cuda_ext.cuda_info()
                    for info_line in cuda_info[:3]:  # Print first 3 lines
                        print(f"  {info_line}")
                except:
                    pass
        else:
            if trimesh is None:
                raise ImportError(
                    "Neither CUDA extension nor trimesh is available. "
                    "Install trimesh with `pip install trimesh` or build CUDA extension.")
            
            fallback_reason = []
            if not CUDA_AVAILABLE:
                fallback_reason.append("CUDA extension not built")
            if not torch.cuda.is_available():
                fallback_reason.append("CUDA not available")
            if not use_cuda:
                fallback_reason.append("CUDA disabled")
                
            reason_str = " (" + ", ".join(fallback_reason) + ")" if fallback_reason else ""
            print(f"VoxelizeTrimesh: Using CPU fallback{reason_str} for {shape} voxelization")

    def _make_grid_points(self, device: torch.device) -> torch.Tensor:
        """Create voxel-center grid points in normalized coordinates [-1, 1].

        Returns:
            [N, 3] tensor with columns ordered as (x, y, z) to match mesh coords.
        """
        D, H, W = self.shape_dhw
        d_coords = torch.linspace(-1, 1, D, device=device)
        h_coords = torch.linspace(-1, 1, H, device=device)
        w_coords = torch.linspace(-1, 1, W, device=device)

        grid_d, grid_h, grid_w = torch.meshgrid(d_coords, h_coords, w_coords, indexing="ij")
        # Match previous convention: stack as (x, y, z) = (w, h, d)
        grid_points = torch.stack([grid_w, grid_h, grid_d], dim=-1).reshape(-1, 3)
        return grid_points

    @torch.no_grad()
    def forward(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Convert batched meshes to occupancy grids.

        Args:
            vertices: [B, V, 3] float tensor in normalized coords [-1, 1]
            faces: [B, F, 3] long/int tensor of triangle indices

        Returns:
            volume: [B, 1, D, H, W] float tensor with 1.0 inside and 0.0 outside
        """
        assert vertices.ndim == 3 and faces.ndim == 3, "vertices/faces must be batched"
        
        # Use CUDA acceleration if available and data is on GPU
        if self.use_cuda and vertices.is_cuda and faces.is_cuda:
            return self._forward_cuda(vertices, faces)
        else:
            return self._forward_cpu(vertices, faces)
    
    def _forward_cuda(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated voxelization using CUDA kernels."""
        device = vertices.device
        D, H, W = self.shape_dhw
        
        # Generate grid points on GPU
        grid_points = self._make_grid_points(device)
        shape_tensor = torch.tensor(self.shape_dhw, dtype=torch.int32, device="cpu")
        
        try:
            # Call CUDA extension
            volume = voxelize_cuda_ext.voxelize_cuda_forward(
                vertices.contiguous(), 
                faces.contiguous().int(),  # Ensure int32
                grid_points.contiguous(),
                shape_tensor
            )
            return volume
            
        except Exception as e:
            warnings.warn(
                f"CUDA voxelization failed: {e}. Falling back to CPU implementation.",
                RuntimeWarning
            )
            return self._forward_cpu(vertices, faces)
    
    def _forward_cpu(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """CPU fallback using Trimesh signed distance computation."""
        if trimesh is None:
            raise RuntimeError("CPU fallback requires trimesh package")
            
        batch_size = vertices.shape[0]
        device = vertices.device
        D, H, W = self.shape_dhw

        # Prepare output
        volume = torch.zeros(batch_size, 1, D, H, W, dtype=torch.float32, device=device)

        # Prepare sampling grid on the target device; we'll convert to numpy per-batch
        grid_points = self._make_grid_points(device=device)
        num_points = grid_points.shape[0]

        for b in range(batch_size):
            # Move mesh data to CPU numpy for Trimesh
            v_np = vertices[b].detach().to("cpu", dtype=torch.float32).numpy()
            f_np = faces[b].detach().to("cpu", dtype=torch.int32).numpy()

            if f_np.size == 0 or v_np.size == 0:
                continue

            mesh = trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)
            
            # Check if mesh has inverted normals (negative volume)
            # If so, fix the normals to ensure consistent SDF behavior
            if mesh.volume < 0:
                # Fix inverted normals - this flips face winding to make volume positive
                mesh.fix_normals()
            
            # Signed distance queries in chunks to limit memory
            sdfs: List[np.ndarray] = []
            pts_np = grid_points.detach().to("cpu", dtype=torch.float32).numpy()
            for start in range(0, num_points, self.chunk_size):
                end = min(start + self.chunk_size, num_points)
                sdf_chunk = trimesh.proximity.signed_distance(mesh, pts_np[start:end])
                sdfs.append(sdf_chunk)

            sdf = np.concatenate(sdfs, axis=0)
            
            # With fixed normals, use Trimesh's documented convention:
            # NEGATIVE distances are OUTSIDE, POSITIVE distances are INSIDE
            # (This is opposite to the common graphics convention)
            inside = (sdf > 0.0).reshape(D, H, W)

            volume[b, 0] = torch.from_numpy(inside.astype(np.float32)).to(device)

        return volume


