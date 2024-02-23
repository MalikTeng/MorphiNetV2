from typing import Union, List
import json
import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from monai.data import MetaTensor

from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import cot_laplacian, packed_to_padded
from pytorch3d.ops.knn import knn_points


__all__ = ["mesh_laplacian_smoothing", "skeleton_loss_fn", "face_score"]


# ----------------------- point to face distance -----------------------

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3


# FacePointDistance
class _FacePointDistance(Function):
    """
    Torch autograd Function wrapper FacePointDistance Cuda implementation
    """

    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_tris,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_tris: Scalar equal to maximum number of faces in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(T,)`, where `dists[t]` is the squared
                euclidean distance of `t`-th triangular face to the closest point in the
                corresponding example in the batch
            idxs: LongTensor of shape `(T,)` indicating the closest point in the
                corresponding example in the batch.

            `dists[t] = d(points[idxs[t]], tris[t, 0], tris[t, 1], tris[t, 2])`,
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`.
        """
        dists, idxs = _C.face_point_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.face_point_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


face_point_distance = _FacePointDistance.apply

@torch.no_grad()
def face_score(
    meshes: Meshes,
    pcls: Pointclouds,
    min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be face_point(mesh, pcl)

    face_point(mesh, pcl): Computes the squared distance of each triangular face in
        mesh to the closest point in pcl.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.

    Returns:
        face_point(mesh, pcl) distance: between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if meshes._N != pcls._N:
        raise ValueError("meshes and pointclouds must be equal sized batches")

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    # max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # face to point distance: shape (T,)
    face_to_point = face_point_distance(
        points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
    )
    face_dist = face_to_point.view(meshes._N, -1).mean(0)

    # group subdivied faces in set of four as result of subdivide original faces, and calculate a possibility to get rid of subdvided faces in that set
    face_dist = torch.softmax(face_dist.view(-1, 6).mean(1), dim=0)

    return face_dist


def mesh_laplacian_smoothing(meshes, method: str = "uniform"):
    r"""
    Computes the laplacian smoothing objective for a batch of meshes.
    This function supports three variants of Laplacian smoothing,
    namely with uniform weights("uniform"), with cotangent weights ("cot"),
    and cotangent curvature ("cotcurv").For more details read [1, 2].

    Args:
        meshes: Meshes object with a batch of meshes.
        method: str specifying the method for the laplacian.
    Returns:
        loss: Average laplacian smoothing loss across the batch.
        Returns 0 if meshes contains no meshes or all empty meshes.

    Consider a mesh M = (V, F), with verts of shape Nx3 and faces of shape Mx3.
    The Laplacian matrix L is a NxN tensor such that LV gives a tensor of vectors:
    for a uniform Laplacian, LuV[i] points to the centroid of its neighboring
    vertices, a cotangent Laplacian LcV[i] is known to be an approximation of
    the surface normal, while the curvature variant LckV[i] scales the normals
    by the discrete mean curvature. For vertex i, assume S[i] is the set of
    neighboring vertices to i, a_ij and b_ij are the "outside" angles in the
    two triangles connecting vertex v_i and its neighboring vertex v_j
    for j in S[i], as seen in the diagram below.

    .. code-block:: python

               a_ij
                /\
               /  \
              /    \
             /      \
        v_i /________\ v_j
            \        /
             \      /
              \    /
               \  /
                \/
               b_ij

        The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
        For the uniform variant,    w_ij = 1 / |S[i]|
        For the cotangent variant,
            w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
        For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
        where A[i] is the sum of the areas of all triangles containing vertex v_i.

    There is a nice trigonometry identity to compute cotangents. Consider a triangle
    with side lengths A, B, C and angles a, b, c.

    .. code-block:: python

               c
              /|\
             / | \
            /  |  \
         B /  H|   \ A
          /    |    \
         /     |     \
        /a_____|_____b\
               C

        Then cot a = (B^2 + C^2 - A^2) / 4 * area
        We know that area = CH/2, and by the law of cosines we have

        A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a

        Putting these together, we get:

        B^2 + C^2 - A^2     2BC cos a
        _______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
           4 * area            2CH


    [1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
    and curvature flow", SIGGRAPH 1999.

    [2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
    """

    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                # pyre-fixme[58]: `/` is not supported for operand types `float` and
                #  `Tensor`.
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    L = L.to_sparse_csr()
    if method == "uniform":
        loss = L.mm(verts_packed)
    elif method == "cot":
        # pyre-fixme[61]: `norm_w` is undefined, or not always defined.
        loss = L.mm(verts_packed) * norm_w - verts_packed
    elif method == "cotcurv":
        # pyre-fixme[61]: `norm_w` may not be initialized here.
        loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
    loss = loss.norm(dim=1)

    loss = loss * weights
    return loss.sum() / N


def skeleton_loss_fn(subdiv_mesh: Meshes, seg_true: List[MetaTensor], slice_info: str, seg_indices: Union[int, str], device: torch.device, t: float=2.73, reduction: str="mean") -> Tensor:
    """
        collect ground-truth segmentation labels from all image-views as label skeleton and calculate the attenuated k-nn distance error between the surface centroid of subdivided mesh and the skeleton.

        input:
            subdiv_mesh: subdivided meshes in voxel space
            seg_true: list of input segmentation labels
            slice_info: slice_info json file includes the data shape and affine matrix for every CMR slices
            seg_indices: index to select the class label in segmentation matching with the mesh
            device: choose either 'cuda' or 'cpu'
            t: temperature parameter in the exponential function
            reduction: choose whether to return the average or sum the distance error of points on batch
        output:
            loss: torch.tensor
    """
    point_label, bbox_centre, bbox_extent = [], [], []
    with torch.no_grad():
        for b, vol in enumerate(seg_true):
            # load segmentation parameters
            label = vol.as_tensor() > 0 if isinstance(seg_indices, str) else vol.as_tensor() == seg_indices
            bbox = torch.nonzero(label, as_tuple=False)
            bbox = torch.stack([bbox.min(axis=0).values, bbox.max(axis=0).values]).T.to(device)
            bbox_centre.append(bbox.float().mean(1))
            bbox_extent.append(bbox.float().diff(1).max())

            coord = torch.meshgrid(*[torch.arange(s) for s in vol.shape])
            coord = torch.stack(coord, axis=0).view(3, -1).float()

            # load view parameters
            view_param = json.load(open(slice_info[b], "r"))
            point_view = []
            for view in view_param.values():
                normal_start, normal_end = view["normal_vector"]
                normal = torch.tensor(normal_end) - torch.tensor(normal_start)
                d_view = torch.dot(normal, torch.tensor(normal_start))
                coord_idx_view = torch.matmul(normal, coord) - d_view < 1e-6
                coord_view = coord[:, coord_idx_view[0] & label.flatten()]
                point_view.append(coord_view.T)
            point_view = torch.cat(point_view, axis=0)
            point_label.append(point_view)
        length_label = torch.LongTensor([i.shape[0] for i in point_label]).to(device)
        point_label = packed_to_padded(
            torch.cat(point_label, dim=0), 
            first_idxs=torch.LongTensor([0] + [i.shape[0] for i in point_label[:-1]]).cumsum(0),
            max_size=max([i.shape[0] for i in point_label])
            ).to(device)
        bbox_extent = torch.stack(bbox_extent, axis=0)
        bbox_centre = torch.stack(bbox_centre, axis=0)

    subdiv_mesh.scale_verts_(bbox_extent / 2)
    subdiv_mesh.offset_verts_(bbox_centre.unsqueeze(1).expand(subdiv_mesh._N, subdiv_mesh._V, -1).reshape(-1, 3))
    verts, faces = subdiv_mesh.verts_packed(), subdiv_mesh.faces_packed()
    point_subdiv = 1/3 * verts[faces].sum(dim=1)

    # knn distance error
    point_subdiv = point_subdiv.view(subdiv_mesh._N, -1, 3)
    nn = knn_points(point_subdiv, point_label, lengths1=None, lengths2=length_label, norm=2, K=1)
    error = nn.dists[..., 0]
    error = error * torch.exp(-error / t)

    if reduction == "mean":
        loss = error.mean()
    elif reduction == "sum":
        loss = error.mean(1).sum()

    return loss



# @torch.no_grad()
# def distance_field(dots: torch.tensor, affine: torch.tensor, image_shape: list, get_outline: bool = False) -> torch.tensor:
#     """Generate distance transform.

#     Args:
#         dots: input outline dots of segmentation as (N, V).
#         affine: affine matrix transform the dots to image coordinate as (4, 4).
#         image_shape: shape of the image as (H, W, D).
#         get_outline: whether to get the outline of the segmentation.
#     Returns:
#         np.ndarray: Distance field.
#     """
#     dots = dots.detach()
#     pixels = torch.matmul(affine,
#                         torch.cat([dots, torch.ones_like(dots[:1])], axis=0))[:3]
#     pixels_idx = torch.bucketize(pixels[:2], torch.arange(max(image_shape)).to(dots.device), right=True)
#     pixels_idx[0] = torch.clamp(pixels_idx[0], 0, image_shape[0] - 1)
#     pixels_idx[1] = torch.clamp(pixels_idx[1], 0, image_shape[1] - 1)

#     field = torch.zeros(tuple(image_shape[:2]), dtype=torch.float32, device=dots.device)
#     field[pixels_idx[0], pixels_idx[1]] = 1.0
#     if get_outline:
#         field = sample_outline(field)
#         pixels_idx = torch.where(field > 0)

#     fg_dist = distance_transform_edt(field[None, None]).squeeze()
#     bg_dist = distance_transform_edt(1 - field[None, None]).squeeze()
#     field = fg_dist + bg_dist
#     field = field[pixels_idx[0], pixels_idx[1]]

#     if get_outline:
#         pixels_idx = torch.stack([*pixels_idx, torch.zeros_like(pixels_idx[0])], axis=0)
#         pixels_idx = torch.cat([pixels_idx, torch.ones_like(pixels_idx[:1])], axis=0)
#         pixels_idx = torch.matmul(torch.inverse(affine), pixels_idx.float())[:3]

#     return field, pixels_idx

# def sample_outline(image):
#     a = F.max_pool2d(image[None, None], 
#                      kernel_size=(3, 1), stride=1, padding=(1, 0))[0]
#     b = F.max_pool2d(image[None, None], 
#                      kernel_size=(1, 3), stride=1, padding=(0, 1))[0]
#     border, _ = torch.max(torch.cat([a, b], dim=0), dim=0)
#     outline = border - image
#     return outline


# def slice_silhouette_loss(subdiv_mesh: Meshes, seg_true: MetaTensor, scale_downsample: int, slice_info: str, seg_indices: Union[int, str], device: torch.device, alpha: float=2.0, reduction: str="mean") -> Tensor:
#     """
#         slicer find the silhouette of the mesh at a given image-view plane. follwoing steps are performed:

#         1. load the mesh and image-view plane parameters -- data array shape and affine matrix transforming the array from voxel space to patient space (world coordinates).
#         2. locate the long- and short-axes planes in the voxel space.
#         3. find faces from the mesh intersect with the long- and short-axes planes.
#         4. find the silhouette of the mesh by projecting the barycentric coordinates of the faces onto the image-view plane.
#         5. convert the silhouette to the original voxel space and save it as a binary mask.

#         input:
#             subdiv_mesh: subdivided meshes in voxel space
#             seg_true: list of input segmentation labels
#             slice_info: slice_info json file includes the data shape and affine matrix for every CMR slices
#             scale_downsample: downsample scale factor calculated from the crop_window_size and pixdim
#             seg_indices: index to select the class label in segmentation matching with the mesh
#             device: choose either 'cuda' or 'cpu'
#             alpha: reference in paper, Karimi, D. et. al. (2019) Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks, IEEE Transactions on medical imaging, 39(2), 499-513
#             reduction: choose whether to return the average or sum the distance error of points on all image-views
#         output:
#             loss: torch.tensor
#     """
#     b = len(seg_true)
#     silhouette_error = 0.0
#     for i in range(b):
#         with torch.no_grad():
#             # load segmentation parameters
#             volume_affine = torch.linalg.inv(seg_true[i].affine.float()) # from patient space to voxel space
#             h, w, d = seg_true[i].shape[1:]
#             bbox = torch.meshgrid(*[torch.arange(s) for s in seg_true[i].shape[1:]])
#             bbox = torch.stack(bbox, axis=0).view(3, -1).to(device)
#             bbox = bbox[:, seg_true[i].as_tensor().flatten() > 0]
#             bbox_min, bbox_max = bbox.min(axis=1)[0], bbox.max(axis=1)[0]
#             bbox_extent = bbox_max - bbox_min
#             bbox_centre = bbox_min + bbox_extent / 2
#             # load view parameters
#             view_param = json.load(open(slice_info[i], "r"))

#         view_error = 0.0
#         for view in view_param.keys():
#             with torch.no_grad():
#                 view_affine = torch.tensor(view_param[view]["affine_matrix"])
#                 view_shape = list(view_param[view]["data_shape"])
#                 # create a grid of the voxel space
#                 coord = torch.meshgrid(*[torch.arange(s) for s in view_shape])
#                 coord = torch.stack(coord, axis=0).view(3, -1)
#                 # locate the long- and short-axes planes in the voxel space
#                 normal = volume_affine @ view_affine @ torch.tensor([0, 0, 1, 0], dtype=torch.float32)
#                 normal = normal[:3] / torch.linalg.norm(normal[:3])
#                 coord = torch.matmul(volume_affine @ view_affine,
#                                     torch.cat([coord.float(), torch.ones_like(coord)[:1]], axis=0))[:3]
#                 coord = torch.round(coord)
#                 coord = coord[:, (coord[0] >= 0) & (coord[0] < h) & \
#                                 (coord[1] >= 0) & (coord[1] < w) & \
#                                     (coord[2] >= 0) & (coord[2] < d)]
#                 normal = normal.to(device)
#                 coord = coord.to(device)
#                 # find the silhouette of the segmentation on the same plane
#                 label = seg_true[i].as_tensor()[0, coord[0].long(), coord[1].long(), coord[2].long()]
#                 p_label = coord[:, 
#                                 (label > 0) if isinstance(seg_indices, str) 
#                                 else (label == seg_indices)]
#                 if p_label.shape[1] == 0:
#                     continue
            
#             # rescale and translate the mesh to match with the segmentation label
#             mesh = subdiv_mesh[i]
#             mesh.scale_verts_(bbox_extent / 2)
#             mesh.offset_verts_(bbox_centre)
#             verts, faces = mesh.verts_packed(), mesh.faces_packed()
#             # find the centroid of the faces
#             centroid = 1/3 * verts[faces].sum(dim=1)
#             # find the distance from the centroid to the plane
#             dist = torch.dot(normal, coord[:, 0])
#             p = torch.matmul(centroid, normal) - dist
#             # find the silhouette of the mesh as the centroid of faces projected on the plane, tolerance is set to 1.5 mm
#             p_pred = centroid - p[:, None] * normal[None]
#             p_pred = p_pred[torch.where(abs(p) < 1.5)[0]].T

#             # calculate the difference matrix between p_pred and p_label and applies the distance weights calculated from distance_transform_edt, referencing monai.losses.HausdorffDTLoss
#             slice_affine = torch.linalg.inv(volume_affine @ view_affine).to(device)
#             pred_edt, _ = distance_field(p_pred, slice_affine, view_shape)
#             label_edt, p_label = distance_field(p_label, slice_affine, view_shape, get_outline=True)

#             distance = pred_edt[:, None] ** alpha + label_edt[None] ** alpha
#             error = torch.softmax((p_pred[..., None] - p_label[:, None]) ** 2, dim=0) * distance
            
#             if reduction == "mean":
#                 view_error += error.mean()
#             elif reduction == "sum":
#                 view_error += error.sum()

#         silhouette_error += view_error / len(view_param.keys())
    
#     silhouette_error /= b

#     return silhouette_error
