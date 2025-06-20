'''
    voxelise the template mesh to a matrix of size 128 x 128 x 128 for DeformNet data preparation
'''
import trimesh
import numpy as np
import os, sys
sys.path.extend([
    os.path.join(os.path.dirname(__file__), '../template'),
])
import plotly.graph_objects as go


# Load the 3D mesh from an OBJ file
mesh = trimesh.load_mesh('template/template_mesh-myo.obj')

# Transform the mesh to [0, 1] bounding box
mesh.vertices -= mesh.bounds[0]
mesh.vertices /= mesh.extents

# Transform the mesh to be compatible with DeformNet
mesh.vertices = np.array([[-1.0, -1.0, 1.0]]) * mesh.vertices[:, [1, 0, 2]]

# Export the mesh as an obj file
mesh.export('template.obj')

# Export subdivided mesh as an obj file
for _ in range(2):
    mesh = mesh.subdivide()
mesh.export('template_manifold.obj')