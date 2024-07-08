from typing import Union, List
import json
import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from monai.data import MetaTensor

from trimesh.convex import convex_hull

from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.ops import cot_laplacian, packed_to_padded, sample_points_from_meshes
from pytorch3d.ops.knn import knn_points

import plotly.graph_objects as go


__all__ = ["surface_crossing_loss", "mesh_laplacian_smoothing"]


# ----------------------- point to face distance -----------------------

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3


# PointFaceDistance
class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
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
            max_points: Scalar equal to maximum number of points in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.

            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
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
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


point_face_distance = _PointFaceDistance.apply


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


def surface_crossing_loss(
    meshes: Meshes,
    pcls: Pointclouds,
    meshes_labels: torch.LongTensor,
    min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
):
    """
    Computes the distance between surface point clouds from a mesh and faces of another mesh in a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    # if len(meshes) != len(pcls):
    #     raise ValueError("meshes and pointclouds must be equal sized batches")
    # N = len(meshes)

    penalty = 0.0

    for label_pairs in [[1, 0], [2, 1], [0, 2]]:

        # get the label as mask of mesh faces
        points_label = torch.nonzero(meshes_labels == label_pairs[0])[:, 0]
        meshes_label = torch.nonzero(meshes_labels == label_pairs[1]).T
        # get the subset of pointclouds and meshes
        pcls_subset = Pointclouds(points=pcls[:, points_label])
        meshes_subset = meshes.submeshes([meshes_label])

        # # create a plot drawing the pcls_subset and meshes_subset in the same scene using plotly
        # fig = go.Figure()
        # fig.add_trace(
        #     go.Scatter3d(
        #         x=pcls_subset.points_packed()[:, 0].cpu().numpy(),
        #         y=pcls_subset.points_packed()[:, 1].cpu().numpy(),
        #         z=pcls_subset.points_packed()[:, 2].cpu().numpy(),
        #         mode="markers",
        #         marker=dict(size=2),
        #     )
        # )
        # verts = meshes_subset.verts_packed().detach().cpu().numpy()
        # faces = meshes_subset.faces_packed().detach().cpu().numpy()
        # fig.add_trace(
        #     go.Mesh3d(
        #         x=verts[:, 0],
        #         y=verts[:, 1],
        #         z=verts[:, 2],
        #         i=faces[:, 0],
        #         j=faces[:, 1],
        #         k=faces[:, 2],
        #         opacity=0.5,
        #     )
        # )
        # fig.update_layout(scene=dict(aspectmode="data"))
        # fig.write_html(f"points_surface-{label_pairs[0]}_{label_pairs[1]}.html")

        # packed representation for pointclouds
        points = pcls_subset.points_packed()  # (P, 3)
        points_first_idx = pcls_subset.cloud_to_packed_first_idx()
        max_points = pcls_subset.num_points_per_cloud().max().item()

        # packed representation for faces
        verts_packed = meshes_subset.verts_packed()
        faces_packed = meshes_subset.faces_packed()
        tris = verts_packed[faces_packed]  # (T, 3, 3)
        tris_first_idx = meshes_subset.mesh_to_faces_packed_first_idx()
        max_tris = meshes_subset.num_faces_per_mesh().max().item()

        # point to face distance: shape (P,)
        point_to_face = point_face_distance(
            points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
        )

        # weight each example by the inverse of number of points in the example
        point_to_cloud_idx = pcls_subset.packed_to_cloud_idx()  # (sum(P_i),)
        num_points_per_cloud = pcls_subset.num_points_per_cloud()  # (N,)
        weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
        weights_p = 1.0 / weights_p.float()
        point_to_face = point_to_face * weights_p

        # face to point distance: shape (T,)
        face_to_point = face_point_distance(
            points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
        )

        # weight each example by the inverse of number of faces in the example
        tri_to_mesh_idx = meshes_subset.faces_packed_to_mesh_idx()  # (sum(T_n),)
        num_tris_per_mesh = meshes_subset.num_faces_per_mesh()  # (N, )
        weights_t = num_tris_per_mesh.gather(0, tri_to_mesh_idx)
        weights_t = 1.0 / weights_t.float()
        face_to_point = face_to_point * weights_t


        # the penalty is applied to where distance is smaller than d_min in the NDC space.
        point_to_face_penalty = torch.clamp(1 / 128 - point_to_face, min=0)

        face_to_point_penalty = torch.clamp(1 / 128 - face_to_point, min=0)

        penalty += point_to_face_penalty.sum() + face_to_point_penalty.sum()

    return penalty


# ----------------------- laplacian smoothing -----------------------

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
