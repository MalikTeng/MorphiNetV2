import json
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from monai.data import MetaTensor
from monai.transforms.utils import distance_transform_edt
from pytorch3d.structures import Meshes


__all__ = ["slice_silhouette_loss"]


@torch.no_grad()
def distance_field(dots: torch.Tensor, affine: torch.Tensor, image_shape: list, get_outline: bool = False) -> torch.Tensor:
    """Generate distance transform.

    Args:
        dots: input outline dots of segmentation as (N, V).
        affine: affine matrix transform the dots to image coordinate as (4, 4).
        image_shape: shape of the image as (H, W, D).
        get_outline: whether to get the outline of the segmentation.
    Returns:
        np.ndarray: Distance field.
    """
    dots = dots.detach()
    pixels = torch.matmul(affine,
                        torch.cat([dots, torch.ones_like(dots[:1])], axis=0))[:3]
    pixels_idx = torch.bucketize(pixels[:2], torch.arange(max(image_shape)).to(dots.device), right=True)
    pixels_idx[0] = torch.clamp(pixels_idx[0], 0, image_shape[0] - 1)
    pixels_idx[1] = torch.clamp(pixels_idx[1], 0, image_shape[1] - 1)

    field = torch.zeros(tuple(image_shape[:2]), dtype=torch.float, device=dots.device)
    field[pixels_idx[0], pixels_idx[1]] = 1
    if get_outline:
        field = sample_outline(field)
        pixels_idx = torch.where(field > 0)

    fg_dist = distance_transform_edt(field[None, None]).squeeze()
    bg_dist = distance_transform_edt(1 - field[None, None]).squeeze()
    field = fg_dist - bg_dist
    field = field[pixels_idx[0], pixels_idx[1]]

    if get_outline:
        pixels_idx = torch.stack([*pixels_idx, torch.zeros_like(pixels_idx[0])], axis=0)
        pixels_idx = torch.cat([pixels_idx, torch.ones_like(pixels_idx[:1])], axis=0).float()
        pixels_idx = torch.matmul(torch.inverse(affine), pixels_idx)[:3]

    return field, pixels_idx

def sample_outline(image):
    a = F.max_pool2d(image[None, None].float(), 
                     kernel_size=(3, 1), stride=1, padding=(1, 0))[0]
    b = F.max_pool2d(image[None, None].float(), 
                     kernel_size=(1, 3), stride=1, padding=(0, 1))[0]
    border, _ = torch.max(torch.cat([a, b], dim=0), dim=0)
    outline = border - image.float()
    return outline


def slice_silhouette_loss(subdiv_mesh: Meshes, seg_true: MetaTensor, slice_info: str, scale_downsample: int, seg_indices: int or tuple, device: torch.device, alpha: float=2.0, reduction: str="mean") -> Tensor:
    """
        slicer find the silhouette of the mesh at a given image-view plane. follwoing steps are performed:

        1. load the mesh and image-view plane parameters -- data array shape and affine matrix transforming the array from voxel space to patient space (world coordinates).
        2. locate the long- and short-axes planes in the voxel space.
        3. find faces from the mesh intersect with the long- and short-axes planes.
        4. find the silhouette of the mesh by projecting the barycentric coordinates of the faces onto the image-view plane.
        5. convert the silhouette to the original voxel space and save it as a binary mask.

        input:
            subdiv_mesh: subdivided meshes in voxel space
            seg_true: list of input segmentation labels
            slice_info: slice_info json file includes the data shape and affine matrix for every CMR slices
            scale_downsample: downsample scale factor calculated from the crop_window_size and pixdim
            seg_indices: index to select the class label in segmentation matching with the mesh
            device: choose either 'cuda' or 'cpu'
            alpha: reference in paper, Karimi, D. et. al. (2019) Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks, IEEE Transactions on medical imaging, 39(2), 499-513
            reduction: choose whether to return the average or sum the distance error of points on all image-views
        output:
            loss: torch.Tensor
    """
    b = len(seg_true)
    silhouette_error = 0.0
    for i in range(b):
        # load segmentation parameters
        volume_affine = torch.linalg.inv(seg_true[i].affine).float() # from patient space to voxel space
        h, w, d = seg_true[i].shape[1:]
        bbox = torch.meshgrid(*[torch.arange(s) for s in seg_true[i].shape[1:]])
        bbox = torch.stack(bbox, axis=0).view(3, -1).to(device)
        bbox = bbox[:, seg_true[i].flatten() > 0]
        bbox_min, bbox_max = bbox.min(axis=1)[0], bbox.max(axis=1)[0]
        bbox_extent = bbox_max - bbox_min
        bbox_centre = bbox_min + bbox_extent / 2

        view_error = 0.0
        # load view parameters
        view_param = json.load(open(slice_info[i], "r"))
        for view in view_param.keys():
            view_affine = torch.tensor(view_param[view]["affine_matrix"])
            view_shape = list(view_param[view]["data_shape"])
            # create a grid of the voxel space
            coord = torch.meshgrid(*[torch.arange(s) for s in view_shape])
            coord = torch.stack(coord, axis=0).view(3, -1).float()
            # locate the long- and short-axes planes in the voxel space
            normal = volume_affine @ view_affine @ torch.tensor([0, 0, 1, 0], dtype=torch.float)
            normal = normal[:3] / torch.linalg.norm(normal[:3])
            coord = torch.matmul(volume_affine @ view_affine,
                                torch.cat([coord, torch.ones_like(coord)[:1]], axis=0))[:3]
            coord = torch.round(coord).long()
            coord = coord[:, (coord[0] >= 0) & (coord[0] < h) & \
                            (coord[1] >= 0) & (coord[1] < w) & \
                                (coord[2] >= 0) & (coord[2] < d)]
            normal = normal.to(device)
            coord = coord.to(device)
            
            # rescale and translate the mesh to match with the segmentation label
            verts = subdiv_mesh[i].verts_packed()
            faces = subdiv_mesh[i].faces_packed()
            verts = (verts - scale_downsample // 2) * bbox_extent / 2 + bbox_centre
            # verts = verts * bbox_extent / 2 + bbox_centre

            # find the centroid of the faces
            centroid = 1/3 * verts[faces].sum(dim=1)
            # find the distance from the centroid to the plane
            dist = torch.dot(normal, coord[:, 0].float())
            p = torch.matmul(centroid, normal) - dist
            # find the silhouette of the mesh as the centroid of faces projected on the plane, tolerance is set to 1.5 mm
            p_pred = centroid - p[:, None] * normal[None]
            p_pred = p_pred[torch.where(abs(p) < 1.5)[0]].T
            # p = p[abs(p) < 1.5]
            # find the silhouette of the segmentation on the same plane
            label = seg_true[i].as_tensor()[0, coord[0], coord[1], coord[2]]
            p_label = coord[:, 
                            (label == seg_indices[0]) | (label == seg_indices[1]) 
                            if isinstance(seg_indices, list) else (label == seg_indices)].float()

            # calculate the difference matrix between p_pred and p_label and applies the distance weights calculated from distance_transform_edt, referencing monai.losses.HausdorffDTLoss
            slice_affine = torch.linalg.inv(volume_affine @ view_affine).to(device)
            pred_edt, _ = distance_field(p_pred, slice_affine, view_shape)
            label_edt, p_label = distance_field(p_label, slice_affine, view_shape, get_outline=True)

            distance = pred_edt[:, None] ** alpha + label_edt[None] ** alpha
            error = torch.softmax((p_pred[..., None] - p_label[:, None]) ** 2, dim=0) * distance
            
            if reduction == "mean":
                view_error += error.mean()
            elif reduction == "sum":
                view_error += error.sum()

        silhouette_error += view_error / len(view_param.keys())
    
    silhouette_error /= b

    return silhouette_error

