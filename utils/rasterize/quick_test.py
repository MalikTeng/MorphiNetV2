#!/usr/bin/env python3
"""
Quick validation test for voxelization (CPU only, small grids).
"""

import torch
import numpy as np
from voxelize_trimesh import VoxelizeTrimesh

def create_simple_cube():
    """Create a simple cube mesh."""
    vertices = torch.tensor([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],  # bottom
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]     # top
    ], dtype=torch.float32).unsqueeze(0)
    
    faces = torch.tensor([
        # Proper winding order (counter-clockwise when viewed from outside)
        [0, 2, 1], [0, 3, 2],  # bottom
        [4, 5, 6], [4, 6, 7],  # top  
        [0, 1, 5], [0, 5, 4],  # front
        [2, 7, 6], [2, 3, 7],  # back
        [0, 4, 7], [0, 7, 3],  # left
        [1, 2, 6], [1, 6, 5]   # right
    ], dtype=torch.int64).unsqueeze(0)
    
    return vertices, faces

def main():
    print("Quick Voxelization Test")
    print("=" * 30)
    
    # Create test mesh
    vertices, faces = create_simple_cube()
    print(f"Test mesh: {vertices.shape[1]} vertices, {faces.shape[1]} faces")
    
    # Test with small grid (8x8x8)
    voxelizer = VoxelizeTrimesh(shape=[8, 8, 8], use_cuda=False)
    print("\nVoxelizing 8³ grid...")
    
    try:
        volume = voxelizer(vertices, faces)
        print(f"✅ Success! Volume shape: {volume.shape}")
        
        # Check center voxel (should be inside = 1.0)
        center_val = volume[0, 0, 4, 4, 4].item()
        print(f"Center voxel value: {center_val}")
        
        # Check corner voxel (should be outside = 0.0)
        corner_val = volume[0, 0, 0, 0, 0].item()
        print(f"Corner voxel value: {corner_val}")
        
        # Count total voxels inside
        total_inside = volume.sum().item()
        print(f"Total voxels inside: {total_inside}")
        
        if center_val > 0.5 and corner_val < 0.5 and total_inside > 0:
            print("✅ Basic voxelization working correctly!")
            return True
        else:
            print("❌ Voxelization results seem incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)