import os
import trimesh
import argparse

def subdivide_mesh(input_path, iterations=2):
    """
    Load a mesh, clean it up, apply Loop subdivision, and save the result
    
    Args:
        input_path (str): Path to the input mesh OBJ file
        iterations (int): Number of subdivision iterations
    """
    # Load the mesh
    print(f"Loading mesh from {input_path}")
    mesh = trimesh.load(input_path)
    
    # Get original vertex and face count
    original_vertices = len(mesh.vertices)
    original_faces = len(mesh.faces)
    print(f"Original mesh: {original_vertices} vertices, {original_faces} faces")
    
    # Clean up the mesh
    print("Cleaning up the mesh...")
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.fix_normals()
    mesh.fill_holes()
    mesh.remove_unreferenced_vertices()
    
    print(f"Cleaned mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Apply Loop subdivision
    print(f"Applying Loop subdivision ({iterations} iterations)...")
    subdivided_vertices, subdivided_faces = trimesh.remesh.subdivide_loop(
        vertices=mesh.vertices, faces=mesh.faces, iterations=iterations
    )
    
    # Create new mesh with subdivided data
    subdivided_mesh = trimesh.Trimesh(vertices=subdivided_vertices, faces=subdivided_faces)
    print(f"Subdivided mesh: {len(subdivided_mesh.vertices)} vertices, {len(subdivided_mesh.faces)} faces")
    
    # Generate output filename
    base_path, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(base_path, f"{name}_subdivided{ext}")
    
    # Save the subdivided mesh
    subdivided_mesh.export(output_path)
    print(f"Saved subdivided mesh to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subdivide a mesh using Loop subdivision")
    parser.add_argument("input_path", help="Path to the input mesh OBJ file")
    parser.add_argument("--iterations", type=int, default=2, help="Number of subdivision iterations (default: 2)")
    
    args = parser.parse_args()
    subdivide_mesh(args.input_path, args.iterations)
