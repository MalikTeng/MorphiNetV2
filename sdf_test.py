"""
    test if F.grid_sample works for sampling signed distance field
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from trimesh.base import Trimesh
from trimesh import load_mesh
from trimesh.voxel.ops import matrix_to_marching_cubes
from monai.transforms.utils import distance_transform_edt
from monai.transforms import (
    Compose,
    AsDiscrete,
    KeepLargestConnectedComponent,
    Spacing,
    CropForeground,
    Resize,
    SpatialPad,
    EnsureType
)

from data.transform import pre_transform

# ct data test
# Load a label sample from the training dataset and process it with the pre_transform function
dataset_root = "/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset020_SCOTHEART"
label_repo = json.load(open("dataset/dataset_task20_f0.json", 'r'))["train_fold0"]
transform = pre_transform(["ct_image", "ct_label"], "ct", "valid", False, [128, 128, 128], [4, 4, 4], 2.0)
# data augmentation for resizing the segmentation prediction into crop window size
post_transform = [
    Spacing([2.0, 2.0, 2.0], mode="nearest"),
    CropForeground(),
    Resize(128, size_mode="longest", mode="nearest-exact"),
    SpatialPad(128, method="end", mode="constant"),
    EnsureType(),
]
post_transform = Compose(post_transform[1:])

label = np.random.choice(label_repo)
label = transform({
    "ct_image": os.path.join(dataset_root, label["image"]),
    "ct_label": os.path.join(dataset_root, label["label"])
})["ct_label"]
label = post_transform(label)
label = label == 2   # select the label

# # mr data test
# dataset_root = "/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX"
# label_repo = json.load(open("dataset/dataset_task11_f0.json", 'r'))["train_fold0"]
# transform = pre_transform(["mr_image", "mr_label"], "mr", "valid", False, [128, 128, 128], [1.367, 1.367, 1.367], 1)
# # data augmentation for resizing the segmentation prediction into crop window size
# post_transform = [
#     Spacing([1, -1, -1],  mode="nearest"),
#     CropForeground(),
#     Resize(128, size_mode="longest", mode="nearest-exact"),
#     SpatialPad(128, method="end", mode="constant"),
#     EnsureType(),
# ]
# post_transform = Compose(post_transform)

# label = np.random.choice(label_repo)
# label = transform({
#     "mr_image": os.path.join(dataset_root, label["image"]),
#     "mr_label": os.path.join(dataset_root, label["label"])
# })["mr_label"]
# label = post_transform(label[0].unsqueeze(0))
# label = label == 2   # select the label

# # # mr imageTr and imageTs comparison
# dataset_root = "/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX"
# image_repo = json.load(open("dataset/dataset_task11_f0.json", 'r'))["test"]
# transform = pre_transform(["mr_image", "mr_label"], "mr", "valid", False, [128, 128, 128], [None, None, None], 1)

# # Load a label sample from the training dataset and process it with the pre_transform function
# image = np.random.choice(image_repo)
# image = transform({
#     "mr_image": os.path.join(dataset_root, image["image"]),
# })["mr_image"]
# # plot the image (3d to 2d image) as point clouds using matplotlib
# image = image[0]
# image = image[image.shape[0] // 2]
# plt.imshow(image, cmap="gray")
# plt.savefig("image.png")

# Calculate the distance field of the label and the gradient of the distance field (i, j, k)
label_df = distance_transform_edt(1 - label.float()) + distance_transform_edt(label.float())
label_grad = torch.gradient(-label_df, dim=(1, 2, 3), edge_order=1)     # i j k <--> y x z
label_grad = torch.stack(label_grad, dim=1)
label_grad /= torch.norm(label_grad, dim=1, keepdim=True)
# mute any nan/inf values in label_grad
label_grad[torch.isnan(label_grad)] = 0
label_grad[torch.isinf(label_grad)] = 0

# Visualise the sample and the gradients with plotly in the NDC space from world space (128, 128, 128)
label_meshed = matrix_to_marching_cubes(label.squeeze(0))
verts, faces = label_meshed.vertices, label_meshed.faces
verts = 2 * (verts / 128 - 0.5)
y, x, z = verts.T
I, J, K = faces.T
fig = go.Figure(data=[
    go.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        color='red', opacity=0.5
        )
    ])
x, y, z = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128), np.linspace(-1, 1, 128), indexing='xy')

# Load the control mesh
control_mesh = load_mesh("/home/yd21/Documents/MorphiNet/template/initial_mesh-myo.obj")
# visualise the control_mesh
vertices, faces = control_mesh.vertices, control_mesh.faces
x, y, z = vertices.T
I, J, K = faces.T
fig.add_trace(
    go.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        color='blue', opacity=0.5
    )
)
# # update the vertices with new x, y, z and save the control mesh
# control_mesh.vertices = np.stack([x, y, z], axis=-1)
# control_mesh.export("/home/yd21/Documents/MorphiNet/template/initial_mesh-myo_new.obj")


# visualise the gradients sampled at the vertices position
control_mesh = Meshes(
    verts=[torch.tensor(control_mesh.vertices, dtype=torch.float32)],
    faces=[torch.tensor(control_mesh.faces, dtype=torch.int64)]
)
label_grad *= label_df.unsqueeze(1)
mesh_grad = F.grid_sample(
    label_grad.permute(0, 1, 4, 2, 3).float(),                  # shape: (N, C: yxz, D, H, W)
    control_mesh.verts_padded().unsqueeze(1).unsqueeze(1),      # shape: (N, 1, 1, V, 3: xyz)
    align_corners=False
).view(1, 3, -1).transpose(-1, -2)[..., [1, 0, 2]]              # shape: (N, C: yxz, 1, 1, V) -> (N, V, C: xyz) -> (N, V, 3: xyz
x, y, z = control_mesh.verts_packed().numpy().T
u, v, w = mesh_grad[0].numpy().T
fig.add_trace(
    go.Cone(
        x=x, y=y, z=z,
        u=u, v=v, w=w,
        colorscale='Viridis', sizemode='scaled', sizeref=1, showscale=True
    )
)
fig.write_html("seg_mesh_gradient.html")
fig.data = []

# Update the control_mesh vertices by warping
# step one -- rigid registration
verts = control_mesh.verts_padded()
offset = torch.stack([2 * (torch.nonzero(df <= 1).float().mean(0) / label_df.shape[-1] - 0.5) for df in label_df])[:, [1, 0, 2]] - verts.mean(1)    # i, j, k -> x, y, z
offset = offset.unsqueeze(0).expand(1, control_mesh._V, -1)
verts += offset
# step two -- localised deformation
verts = (verts + 1) * 64
verts += mesh_grad

# visualise the warped control_mload_meshesh and the label
x, y, z = verts[0].numpy().T
I, J, K = control_mesh.faces_packed().numpy().T
fig = go.Figure(data=[
    go.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        color='blue', opacity=0.5
    )
])
verts, faces = label_meshed.vertices, label_meshed.faces
y, x, z = verts.T
I, J, K = faces.T
fig.add_trace(
    go.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        color='red', opacity=0.5
        )
    )

# save the plotly figure as a html file
fig.write_html("seg_warped_mesh.html")

