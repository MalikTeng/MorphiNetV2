import os
import sys
import numpy as np
import torch
import nibabel as nib
from pytorch3d.io import save_obj, load_obj
from pytorch3d.ops.marching_cubes import marching_cubes
from pytorch3d.structures import Meshes
from pytorch3d.ops import taubin_smoothing
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse


def extract_lv_myocardium_mesh(input_path, output_path, label_value=1, smoothing=True, isolevel=0.5):
    """
    Extract the left ventricle myocardium from a segmentation volume and convert to mesh
    
    Args:
        input_path: Path to the input volume file (.nii.gz)
        output_path: Path to save the output mesh (.obj)
        label_value: Label value for the left ventricle in the segmentation (default: 1)
        smoothing: Whether to apply Taubin smoothing to the mesh
        isolevel: Isolevel for marching cubes algorithm
    """
    # Load the volume file using nibabel
    print(f"Loading volume from {input_path}")
    try:
        volume = nib.load(input_path)
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
        
    volume_data = volume.get_fdata()
    
    # Convert to torch tensor
    volume_tensor = torch.from_numpy(volume_data).float()
    
    # Select left ventricle myocardium with the specified label value
    print(f"Selecting left ventricle with label value {label_value}")
    lv_myocardium = (volume_tensor == label_value).float()
    
    # Check if any voxels match the specified label value
    if torch.sum(lv_myocardium) == 0:
        print(f"Warning: No voxels with label value {label_value} (left ventricle) found in the volume.")
        print(f"Please ensure the input volume contains segmentation with left ventricle labeled as {label_value}.")
        sys.exit(1)
    
    # Prepare for marching cubes (needs 4D tensor in pytorch3d)
    # The expected shape is (batch, depth, height, width)
    lv_myocardium = lv_myocardium.unsqueeze(0)
    
    # Rearrange dimensions if needed (depending on the original data format)
    # lv_myocardium = lv_myocardium.permute(0, 3, 1, 2)  # Uncomment if needed
    
    # Extract surface mesh using marching cubes
    print("Extracting surface mesh using marching cubes")
    try:
        verts, faces = marching_cubes(
            lv_myocardium,
            isolevel=isolevel,
            return_local_coords=True
        )
    except Exception as e:
        print(f"Error in marching cubes algorithm: {e}")
        sys.exit(1)
    
    # Create a mesh object
    mesh = Meshes(verts=verts, faces=faces)
    
    # Apply Taubin smoothing if requested
    if smoothing:
        print("Applying Taubin smoothing")
        mesh = taubin_smoothing(mesh, 0.77, -0.34, 30)
    
    # Save the mesh as an OBJ file
    print(f"Saving mesh to {output_path}")
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only try to create directory if path has a directory component
            os.makedirs(output_dir, exist_ok=True)
        save_obj(output_path, mesh.verts_packed(), mesh.faces_packed())
    except PermissionError:
        print(f"Error: Permission denied when creating directory or writing to '{output_path}'")
        print("Please check if you have write permissions to this location or use a different path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error saving mesh: {e}")
        sys.exit(1)
    
    print(f"Successfully exported mesh with {mesh.verts_packed().shape[0]} vertices and {mesh.faces_packed().shape[0]} faces")
    return mesh


def visualize_meshes(extracted_mesh, template_path, output_html_path):
    """
    Create a 3D visualization with both the extracted mesh and template mesh
    
    Args:
        extracted_mesh: PyTorch3D mesh object of the extracted LV myocardium
        template_path: Path to the template mesh file (.obj)
        output_html_path: Path to save the HTML visualization
    """
    print(f"Loading template mesh from {template_path}")
    template_verts, template_faces, _ = load_obj(template_path)
    
    # Get the vertices and faces from the extracted mesh
    extracted_verts = extracted_mesh.verts_packed().detach().cpu().numpy()
    extracted_faces = extracted_mesh.faces_packed().detach().cpu().numpy()
    
    # Get the vertices and faces from the template mesh
    template_verts = template_verts.detach().cpu().numpy()
    template_faces = template_faces.verts_idx.detach().cpu().numpy()
    
    # Create a plotly figure
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])
    
    # Set up the layout with proper 3D scene configuration
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',  # This ensures proper scaling
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=20, r=20, t=30, b=20)  # Tighter margins for better rendering
    )
    
    # Add the extracted mesh
    x, y, z = extracted_verts[:, 0], extracted_verts[:, 1], extracted_verts[:, 2]
    i, j, k = extracted_faces[:, 0], extracted_faces[:, 1], extracted_faces[:, 2]
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='pink',
        opacity=0.7,
        name="Extracted LV Myocardium"
    ))
    
    # Add the template mesh
    x, y, z = template_verts[:, 0], template_verts[:, 1], template_verts[:, 2]
    i, j, k = template_faces[:, 0], template_faces[:, 1], template_faces[:, 2]
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='blue',
        opacity=0.3,
        name="Template Mesh"
    ))
    
    # Save the figure as an HTML file
    print(f"Saving visualization to {output_html_path}")
    fig.write_html(output_html_path)
    print(f"Visualization saved successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LV myocardium mesh from segmentation volume")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Path to input segmentation volume (.nii.gz). Must contain left ventricle labeled with the specified label value.")
    parser.add_argument("--output", "-o", type=str, required=True, 
                        help="Path to output mesh file (.obj). Example: './results/lv_myocardium.obj'")
    parser.add_argument("--label", "-l", type=int, default=1,
                        help="Label value for the left ventricle in the segmentation (default: 1)")
    parser.add_argument("--template", "-t", type=str, default="template/template_mesh-myo.obj", 
                        help="Path to template mesh file (.obj)")
    parser.add_argument("--no-smoothing", action="store_false", dest="smoothing", 
                        help="Disable Taubin smoothing")
    parser.add_argument("--isolevel", type=float, default=0.5, 
                        help="Isolevel for marching cubes (default: 0.5)")
    parser.add_argument("--visualize", "-v", action="store_true", 
                        help="Create visualization with template mesh")
    
    args = parser.parse_args()
    
    # Validate input and output paths
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist or is not a file.")
        print("Please provide a valid path to a NIFTI segmentation volume (.nii.gz).")
        sys.exit(1)
    
    # Extract and save the mesh
    extracted_mesh = extract_lv_myocardium_mesh(
        args.input, 
        args.output, 
        label_value=args.label,
        smoothing=args.smoothing, 
        isolevel=args.isolevel
    )
    
    # Create visualization if requested
    if args.visualize:
        # Use current directory for visualization output to avoid permission issues
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_html_path = os.path.join(script_dir, "lv_myocardium_visualization.html")
        visualize_meshes(extracted_mesh, args.template, output_html_path) 