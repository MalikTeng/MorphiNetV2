"""
Robust triangle-mesh voxelization using Open3D with interior filling.

This module provides noise-free conversion from surface meshes to binary 
voxel volumes using Open3D's solid voxelization methods (voxel carving and 
raycasting occupancy), eliminating both floating-point precision noise and 
the surface-only limitation of basic triangle intersection.

Key improvements:
- Interior volume filling for surface meshes
- Multiple robust algorithms (voxel carving, raycasting occupancy)
- Mesh preprocessing for watertight surfaces
- Noise-free triangle intersection
"""

from typing import List, Literal
import warnings

import numpy as np
import torch
import torch.nn as nn

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None


class VoxelizeOpen3D(nn.Module):
    """
    Solid mesh voxelization using Open3D with interior volume filling.
    
    This class provides multiple algorithms for converting surface meshes into 
    solid voxel volumes, addressing the shell-vs-volume issue:
    
    - 'carving': Open3D's voxel carving (official method for solid voxelization)
    - 'occupancy': Raycasting-based occupancy computation (faster, GPU-compatible)
    - 'surface': Triangle intersection only (for comparison, creates shells)
    
    Args:
        shape: Target voxel grid dimensions [D, H, W] 
        method: Voxelization method ('carving', 'occupancy', 'surface')
        voxel_size: Size of each voxel (auto-calculated if None)
    """
    
    def __init__(self, shape: List[int], method: Literal['carving', 'occupancy', 'surface'] = 'occupancy', voxel_size: float = None) -> None:
        super().__init__()
        assert len(shape) == 3, "shape must be [D, H, W]"
        assert method in ['carving', 'occupancy', 'surface'], f"Invalid method: {method}"
        
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D not available. Install with: pip install open3d")
            
        self.shape_dhw = list(shape)
        self.method = method
        # Auto-calculate voxel size to fit normalized coordinates [-1, 1]  
        self.voxel_size = voxel_size or (2.0 / max(shape))
        
        print(f"VoxelizeOpen3D: Using '{method}' method for {shape} solid voxelization")
        if method == 'carving':
            print("  → Voxel carving: Fills interior using multi-view depth simulation")
        elif method == 'occupancy':  
            print("  → Raycasting occupancy: Fills interior using ray-mesh intersection")
        else:
            print("  → Surface only: Creates shell representation (for comparison)")
        print(f"  → Voxel size: {self.voxel_size:.4f}")
        
    @torch.no_grad()
    def forward(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """
        Convert batched surface meshes to solid occupancy grids.

        Args:
            vertices: [B, V, 3] float tensor in normalized coords [-1, 1]
            faces: [B, F, 3] long/int tensor of triangle indices

        Returns:
            volume: [B, 1, D, H, W] float tensor with 1.0 inside and 0.0 outside
        """
        if self.method == 'carving':
            return self._voxelize_carving(vertices, faces)
        elif self.method == 'occupancy':
            return self._voxelize_occupancy(vertices, faces) 
        else:  # surface
            return self._voxelize_surface(vertices, faces)
    
    def _voxelize_carving(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Solid voxelization using Open3D's voxel carving method."""
        batch_size = vertices.shape[0]
        device = vertices.device
        D, H, W = self.shape_dhw

        volume = torch.zeros(batch_size, 1, D, H, W, dtype=torch.float32, device=device)

        for b in range(batch_size):
            verts_np = vertices[b].detach().cpu().float().numpy()
            faces_np = faces[b].detach().cpu().long().numpy()

            if faces_np.size == 0 or verts_np.size == 0:
                continue

            try:
                # Create and clean mesh
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts_np)
                mesh.triangles = o3d.utility.Vector3iVector(faces_np)
                
                # Clean mesh for better carving results
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles() 
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                mesh.compute_vertex_normals()
                
                # Voxel carving for solid interior filling
                cubic_size = 2.2  # Slightly larger than [-1,1] bounds
                voxel_resolution = float(max(self.shape_dhw))
                
                try:
                    # Open3D's official solid voxelization method
                    voxel_grid, _, _ = o3d.geometry.VoxelGrid.voxel_carving(
                        mesh, cubic_size, voxel_resolution
                    )
                    
                    # Convert sparse voxel grid to dense array
                    if len(voxel_grid.get_voxels()) > 0:
                        voxel_volume = self._sparse_to_dense(voxel_grid, D, H, W)
                        volume[b, 0] = torch.from_numpy(voxel_volume).to(device)
                        
                except Exception as carving_error:
                    warnings.warn(f"Voxel carving failed for batch {b}: {carving_error}. Trying occupancy fallback.", RuntimeWarning)
                    # Fallback to occupancy method
                    volume[b:b+1] = self._voxelize_occupancy(vertices[b:b+1], faces[b:b+1])
                
            except Exception as e:
                warnings.warn(f"Mesh processing failed for batch {b}: {e}", RuntimeWarning)
                continue

        return volume
    
    def _voxelize_occupancy(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Solid voxelization using raycasting occupancy computation."""
        batch_size = vertices.shape[0] 
        device = vertices.device
        D, H, W = self.shape_dhw

        volume = torch.zeros(batch_size, 1, D, H, W, dtype=torch.float32, device=device)
        
        # Create query points grid 
        query_points = self._create_query_points(device)

        for b in range(batch_size):
            verts_np = vertices[b].detach().cpu().float().numpy()
            faces_np = faces[b].detach().cpu().long().numpy()

            if faces_np.size == 0 or verts_np.size == 0:
                continue

            try:
                # Create mesh
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts_np)
                mesh.triangles = o3d.utility.Vector3iVector(faces_np)
                
                # Create raycasting scene for occupancy queries
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
                
                # Query occupancy at all grid points
                query_np = query_points.cpu().numpy().astype(np.float32)
                occupancy = scene.compute_occupancy(query_np).numpy()
                
                # Reshape to target dimensions
                occupancy_grid = occupancy.reshape(D, H, W).astype(np.float32)
                volume[b, 0] = torch.from_numpy(occupancy_grid).to(device)
                
            except Exception as e:
                warnings.warn(f"Occupancy computation failed for batch {b}: {e}", RuntimeWarning)
                continue

        return volume
    
    def _voxelize_surface(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Surface-only voxelization (creates shells, for comparison)."""
        batch_size = vertices.shape[0]
        device = vertices.device  
        D, H, W = self.shape_dhw

        volume = torch.zeros(batch_size, 1, D, H, W, dtype=torch.float32, device=device)

        for b in range(batch_size):
            verts_np = vertices[b].detach().cpu().float().numpy()
            faces_np = faces[b].detach().cpu().long().numpy()

            if faces_np.size == 0 or verts_np.size == 0:
                continue

            try:
                # Create mesh
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts_np)
                mesh.triangles = o3d.utility.Vector3iVector(faces_np)
                
                # Surface voxelization (triangle intersection only)
                voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
                    mesh, voxel_size=self.voxel_size
                )
                
                # Convert to dense array
                if len(voxel_grid.get_voxels()) > 0:
                    voxel_volume = self._sparse_to_dense(voxel_grid, D, H, W)
                    volume[b, 0] = torch.from_numpy(voxel_volume).to(device)
                
            except Exception as e:
                warnings.warn(f"Surface voxelization failed for batch {b}: {e}", RuntimeWarning)
                continue

        return volume
    
    def _create_query_points(self, device: torch.device) -> torch.Tensor:
        """Create voxel center grid points in normalized coordinates [-1, 1]."""
        D, H, W = self.shape_dhw
        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device) 
        z = torch.linspace(-1, 1, D, device=device)
        
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        return points
    
    def _sparse_to_dense(self, voxel_grid, D: int, H: int, W: int) -> np.ndarray:
        """Convert Open3D sparse voxel grid to dense numpy array."""
        # Get voxel coordinates and map to target grid
        voxels = voxel_grid.get_voxels()
        if len(voxels) == 0:
            return np.zeros((D, H, W), dtype=np.float32)
            
        voxel_coords = np.asarray([voxel_grid.get_voxel_center_coordinate(v.grid_index) 
                                 for v in voxels])
        
        # Map world coordinates [-1,1] to grid indices [0, size-1]
        grid_indices = ((voxel_coords + 1.0) / 2.0 * np.array([D-1, H-1, W-1])).astype(int)
        grid_indices = np.clip(grid_indices, 0, [D-1, H-1, W-1])
        
        # Create dense volume
        voxel_volume = np.zeros((D, H, W), dtype=np.float32)
        for idx in grid_indices:
            voxel_volume[idx[0], idx[1], idx[2]] = 1.0
        
        return voxel_volume


class VoxelizeOpen3DDense(nn.Module):
    """
    Alternative Open3D implementation using dense grid approach.
    
    More compatible with existing pipeline but potentially slower for sparse meshes.
    """
    
    def __init__(self, shape: List[int]) -> None:
        super().__init__()
        assert len(shape) == 3, "shape must be [D, H, W]"
        
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D not available")
            
        self.shape_dhw = list(shape)
        
    def _create_query_points(self, device: torch.device) -> torch.Tensor:
        """Create grid points in normalized coordinates [-1, 1]."""
        D, H, W = self.shape_dhw
        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        z = torch.linspace(-1, 1, D, device=device)
        
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        return points
        
    @torch.no_grad()
    def forward(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Dense grid voxelization using Open3D raycasting."""
        batch_size = vertices.shape[0]
        device = vertices.device
        D, H, W = self.shape_dhw

        volume = torch.zeros(batch_size, 1, D, H, W, dtype=torch.float32, device=device)
        query_points = self._create_query_points(device)

        for b in range(batch_size):
            verts_np = vertices[b].detach().cpu().float().numpy()
            faces_np = faces[b].detach().cpu().long().numpy()

            if faces_np.size == 0 or verts_np.size == 0:
                continue

            try:
                # Create mesh
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts_np)
                mesh.triangles = o3d.utility.Vector3iVector(faces_np)
                
                # Create raycasting scene
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
                
                # Query points
                query_np = query_points.cpu().numpy().astype(np.float32)
                occupancy = scene.compute_occupancy(query_np).numpy()
                
                # Reshape and convert
                occupancy_grid = occupancy.reshape(D, H, W).astype(np.float32)
                volume[b, 0] = torch.from_numpy(occupancy_grid).to(device)
                
            except Exception as e:
                warnings.warn(f"Open3D dense voxelization failed for batch {b}: {e}", RuntimeWarning)
                continue

        return volume
