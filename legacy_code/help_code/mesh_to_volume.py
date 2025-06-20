"""
Mesh to Volume Converter (Single Mesh, Binary Output)
=====================================================

This script converts a single 3D triangle mesh (.obj file) into a binary 
volumetric representation (voxel grid) and saves it as a NIFTI file (.nii.gz).

Purpose:
--------
The primary purpose of this script is to convert a surface-based 3D mesh model into
a binary volumetric representation, which can be used for various analyses, deep learning
training, or visualization purposes.

The script processes a single mesh, converts it to a binary volume, resizes/pads it,
and saves it as a volumetric representation.

Dependencies:
------------
- PyTorch: For tensor operations and GPU acceleration
- PyTorch3D: For mesh handling and operations
- Trimesh: For loading and manipulating mesh files
- NumPy: For numerical operations
- Nibabel: For NIFTI file I/O
- Custom CUDA-accelerated rasterization module (utils.rasterize.rasterize)

Workflow:
---------
1. Load a single triangle mesh from an .obj file using Trimesh
2. Convert the mesh to PyTorch3D's Meshes format
3. Rasterize the mesh into a binary volume using a custom CUDA-accelerated rasterizer
4. Resize/pad the binary volume to the desired output dimensions
5. Save the volumetric representation as a NIFTI file

Usage:
------
- Run directly: python mesh_to_volume.py (uses default paths and parameters)
- Import and use the mesh_to_volume function with custom parameters

Example:
    >>> from mesh_to_volume import mesh_to_volume
    >>> mesh_to_volume("output_binary_volume.nii.gz", [256, 256, 256])
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
import trimesh
from trimesh import load
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Scale

# Add the parent directory to the path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rasterize.rasterize import Rasterize

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def load_mesh_to_pytorch3d(mesh_path):
    """
    Loads a mesh from a file and converts it to PyTorch3D format
    
    Args:
        mesh_path (str): Path to the mesh OBJ file
    
    Returns:
        pytorch3d.structures.Meshes: The mesh in PyTorch3D format
    """
    print(f"Loading mesh from {mesh_path}")
    
    # Load the mesh using trimesh
    mesh = load(mesh_path)
    
    # Apply Loop subdivision twice
    print(f"Applying Loop subdivision (2 iterations)...")
    subdivided_vertices, subdivided_faces = trimesh.remesh.subdivide_loop(
        vertices=mesh.vertices, faces=mesh.faces, iterations=2
    )
    # Update the mesh object with subdivided data
    mesh = trimesh.Trimesh(vertices=subdivided_vertices, faces=subdivided_faces)
    print(f"Mesh subdivided. New vertex count: {len(mesh.vertices)}, face count: {len(mesh.faces)}")
    
    # Convert to PyTorch3D mesh
    # Wrap vertices and faces in a list since PyTorch3D expects batched inputs
    mesh_pytorch3d = Meshes(
        verts=[torch.tensor(mesh.vertices, dtype=torch.float32)],
        faces=[torch.tensor(mesh.faces, dtype=torch.int64)]
    ).to(DEVICE)
    
    print(f"Mesh loaded with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    
    return mesh_pytorch3d

def translate_to_ndc(mesh):
    """
    Translates a mesh to NDC (Normalized Device Coordinates) space.
    NDC space is a coordinate system where coordinates are normalized between -1 and 1.
    
    Note: This function is not used in the current workflow as the meshes
    are already in NDC space, but is kept for reference or potential future use.
    
    Args:
        mesh (pytorch3d.structures.Meshes): The mesh to translate
    
    Returns:
        pytorch3d.structures.Meshes: The translated mesh
    """
    # Get the vertices
    verts = mesh.verts_padded()
    
    # Calculate the center of the bounding box
    bbox_min = torch.min(verts, dim=1)[0]
    bbox_max = torch.max(verts, dim=1)[0]
    center = (bbox_min + bbox_max) / 2
    
    # Calculate the scale factor to fit in [-1, 1]
    scale_factor = torch.max((bbox_max - bbox_min) / 2)
    
    # Normalize vertices to [-1, 1]
    verts = (verts - center.unsqueeze(1)) / scale_factor
    
    # Create a new mesh with the normalized vertices
    return Meshes(
        verts=verts,
        faces=mesh.faces_padded()
    )

def voxelize_mesh(mesh, rasterizer):
    """
    Voxelizes a single mesh using the provided rasterizer
    
    Args:
        mesh (pytorch3d.structures.Meshes): The mesh to voxelize
        rasterizer (utils.rasterize.Rasterize): The rasterizer to use
    
    Returns:
        torch.Tensor: Binary volume representing the voxelized mesh
    """
    with torch.no_grad():  # No gradients needed for inference
        voxelized = rasterizer(
            mesh.verts_padded(), 
            mesh.faces_padded()
        )
    
    # Extract volume from the rasterized output
    volume = voxelized[0, 0]  # Shape: [D, H, W]
    
    # Convert to a binary volume
    binary_volume = torch.where(volume > 0, torch.tensor(1, device=DEVICE), torch.tensor(0, device=DEVICE))
    
    return binary_volume

def mesh_to_volume(output_nii_path, crop_window_size=[128, 128, 128], output_size=[512, 512, 248]):
    """
    Converts a single 3D triangle mesh into a binary volumetric representation
    and saves it as a NIFTI file.
    
    This function performs the following operations:
    1. Loads a single mesh from an aligned OBJ file
    2. Converts the mesh to PyTorch3D format
    3. Rasterizes the mesh using a CUDA-accelerated custom rasterizer into a binary volume
    4. Resizes/pads the volume to the desired output dimensions
    5. Saves the volumetric representation as a NIFTI file
    
    Args:
        output_nii_path (str): Path where the output NIFTI file will be saved
        crop_window_size (list): Dimensions of the processing volume as [depth, height, width]
                                Default is [128, 128, 128]
        output_size (list): Dimensions of the final output volume as [depth, height, width]
                           Default is [512, 512, 248]
    
    Returns:
        None: The function saves the voxelized mesh to the specified output path
    """
    # Define the mesh path - looking in the template folder
    template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "template")
    mesh_path = os.path.join(template_dir, "template_mesh-lv_myo.obj") # Path to the single mesh
    
    # Create a rasterizer with the specified output dimensions
    # This uses a custom CUDA-accelerated rasterizer for efficient voxelization
    rasterizer = Rasterize(crop_window_size)
    
    # Load the mesh to PyTorch3D format - assumes mesh is already in NDC space
    print(f"Processing mesh from {mesh_path}...")
    mesh = load_mesh_to_pytorch3d(mesh_path)
        
    # Voxelize the mesh
    print(f"Voxelizing the mesh...")
    binary_volume = voxelize_mesh(mesh, rasterizer)
    
    # Convert to numpy array
    volume = binary_volume.cpu().numpy().astype(np.int16) # Use int16 for NIFTI
    
    print(f"Initial volume shape: {volume.shape}, data type: {volume.dtype}")
    
    # Resize the volume to the output size using nearest neighbor interpolation
    # to preserve label values (0 or 1)
    if crop_window_size != output_size:
        print(f"Resizing volume from {crop_window_size} to {output_size}...")
        from scipy.ndimage import zoom
        
        # First zoom to [248, 248, 248]
        intermediate_size = [248, 248, 248]
        zoom_factors = [int_dim / in_dim for int_dim, in_dim in zip(intermediate_size, crop_window_size)]
        
        # Use order=0 for nearest neighbor interpolation to preserve label values
        volume = zoom(volume, zoom_factors, order=0)
        print(f"Intermediate volume shape after zoom: {volume.shape}")
        
        # Now pad to reach the final size [512, 512, 248]
        # Calculate symmetric padding for each dimension
        pad_dim0 = output_size[0] - intermediate_size[0]
        pad_dim1 = output_size[1] - intermediate_size[1]
        pad_dim2 = output_size[2] - intermediate_size[2]
        
        # Distribute padding evenly on both sides
        # For odd padding amounts, put the extra pad on the right side
        pad_width = [
            (pad_dim0 // 2, pad_dim0 - pad_dim0 // 2),
            (pad_dim1 // 2, pad_dim1 - pad_dim1 // 2),
            (pad_dim2 // 2, pad_dim2 - pad_dim2 // 2)
        ]
        
        # Pad with zeros
        volume = np.pad(volume, pad_width, mode='constant', constant_values=0)
        
        print(f"Final volume shape after padding: {volume.shape}")
        print(f"Padding applied: {pad_width}")
    
    # Save as NIFTI
    # Create a basic affine transformation matrix
    # The identity matrix means no spatial transformation is applied
    affine = np.eye(4)
    
    # Create the NIFTI image
    nii_img = nib.Nifti1Image(volume, affine)
    
    # Save the NIFTI file
    nib.save(nii_img, output_nii_path)
    print(f"Saved voxelized mesh to {output_nii_path}")
    print(f"Volume contains binary labels (0 or 1).")

if __name__ == "__main__":
    # Set output path
    # Construct paths relative to the script location for better portability
    output_nii_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "template_myo_voxelized.nii.gz") # Updated output name
    
    # Processing size and final output size
    crop_window_size = [128, 128, 128]  # Size for voxelization
    output_size = [512, 512, 248]       # Size for final output
    
    # Run the conversion
    mesh_to_volume(output_nii_path, crop_window_size, output_size) 