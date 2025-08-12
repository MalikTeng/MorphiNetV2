"""
Rasterization module - Trimesh backend only.

This package now exclusively uses the Trimesh signed-distance field
approach for robust mesh voxelization with correct binary output.
"""

from .voxelize_trimesh import VoxelizeTrimesh

__all__ = ["VoxelizeTrimesh"]