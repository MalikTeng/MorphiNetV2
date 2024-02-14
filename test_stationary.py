"""
    read a sample from the CTA dataset and convert the voxel segmentation into a triangle mesh, and save the mesh as a CTA_mesh.obj file
"""
import os
import torch
import numpy as np
import nibabel as nib
from pytorch3d.structures import Meshes
from pytorch3d.ops.marching_cubes import marching_cubes
from pytorch3d.io import save_obj

# load the sample
sample = nib.load('/mnt/data/Experiment/Data/SCOTHEART/labelsTr/110002_CE-ED.nii.gz')
sample = sample.get_fdata()[None]
sample = (sample == 2) | (sample == 4) # select the segmentation labels
sample = torch.tensor(sample, dtype=torch.float32, device=torch.device('cuda:0'))

# convert the voxel segmentation into a triangle mesh
verts, faces = marching_cubes(sample.permute(0, 3, 1, 2), return_local_coords=False)
# reorder the vertices' coordinates from y, x, z to x, y, z
verts = [vert[:, [1, 0, 2]] for vert in verts]
cta_mesh = Meshes(verts, faces)

# save the mesh as a CTA_mesh.obj file
save_obj('template/CTA_mesh.obj', cta_mesh.verts_packed(), cta_mesh.faces_packed())
