"""Plotly visualizations adapted from TestBench visualise-kit scripts."""

import os
from typing import Dict

import numpy as np
import torch
import trimesh
import plotly.graph_objects as go
from pytorch3d.structures import Meshes


def _mesh_to_trimesh(mesh: Meshes) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=mesh.verts_packed().detach().cpu().numpy(),
        faces=mesh.faces_packed().detach().cpu().numpy(),
        process=False,
    )


def _write_figure(fig: go.Figure, output_base: str) -> str:
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    html_path = f"{output_base}.html"
    fig.write_html(html_path)
    try:
        fig.write_image(f"{output_base}.svg", scale=3)
        return f"{output_base}.svg"
    except Exception:
        return html_path


def _add_mesh_trace(fig: go.Figure, mesh: trimesh.Trimesh, name: str, intensity=None, showscale=False):
    fig.add_trace(go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        name=name,
        opacity=1.0,
        hoverinfo="none",
        intensity=intensity,
        colorscale="jet" if intensity is not None else None,
        showscale=showscale,
        color=None if intensity is not None else "rgb(158,202,225)",
        lighting=dict(ambient=0.8),
        lightposition=dict(x=0, y=0, z=2),
    ))


def create_surface_distance_visualization(gt_mesh, pred_mesh, case_id: str, dataset_name: str, export_dir: str, world_params=None) -> str:
    """Create the CT surface-distance view used by TestBench for MorphiNet."""
    gt_tri = _mesh_to_trimesh(gt_mesh)
    pred_tri = _mesh_to_trimesh(pred_mesh)
    try:
        distances = np.abs(trimesh.proximity.signed_distance(gt_tri, pred_tri.vertices))
    except Exception:
        distances = np.zeros(len(pred_tri.vertices), dtype=np.float32)

    fig = go.Figure()
    _add_mesh_trace(fig, pred_tri, "MorphiNet", intensity=distances, showscale=True)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=-1.0, y=-1.0, z=1.0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
        ),
        showlegend=False,
        template="seaborn",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return _write_figure(fig, os.path.join(export_dir, f"{case_id}_{dataset_name}_surface_distance"))


def _outline_coordinates(array: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(array != 0))
    if coords.size == 0:
        return coords.reshape(0, 3)
    return coords[:, [2, 0, 1]]


def _mesh_points_for_slice(voxeld_mesh: np.ndarray, label_points: np.ndarray) -> np.ndarray:
    if label_points.size == 0:
        return label_points.reshape(0, 3)
    valid = []
    shape = np.asarray(voxeld_mesh.shape)
    for point in label_points:
        if np.all(point >= 0) and np.all(point < shape) and voxeld_mesh[tuple(point)] > 0:
            valid.append(point)
    return np.asarray(valid, dtype=np.int64).reshape(-1, 3)


def create_mr_slice_visualization(case_id: str, dataset_name: str, predicted_mesh, mr_masks: Dict[str, torch.Tensor], export_dir: str, voxeld_mesh=None) -> str:
    """Create TestBench-style MR slice overlays of label points and voxelized mesh points."""
    fig = go.Figure()
    colors = ["#fc8d59", "#ffffbf", "#91cf60"]
    voxel_np = None
    if voxeld_mesh is not None:
        voxel_np = voxeld_mesh.detach().cpu().numpy().squeeze()

    for idx, (name, tensor) in enumerate(mr_masks.items()):
        if name == "individual_slices":
            continue
        arrays = tensor.detach().cpu().numpy()
        arrays = np.asarray(arrays).reshape((-1,) + arrays.shape[-3:])
        for slice_idx, array in enumerate(arrays):
            label_points = np.argwhere(array == 2)
            if label_points.size:
                fig.add_trace(go.Scatter3d(
                    x=label_points[:, 0], y=label_points[:, 1], z=label_points[:, 2],
                    mode="markers", name=f"{name}-{slice_idx} label",
                    marker=dict(size=1.0, color=colors[(idx + slice_idx) % len(colors)], opacity=0.55),
                ))
            if voxel_np is not None:
                mesh_points = _mesh_points_for_slice(voxel_np, np.argwhere(array != 0))
                if mesh_points.size:
                    fig.add_trace(go.Scatter3d(
                        x=mesh_points[:, 0], y=mesh_points[:, 1], z=mesh_points[:, 2],
                        mode="markers", name=f"{name}-{slice_idx} mesh",
                        marker=dict(size=1.0, color="#1019E3", opacity=1.0),
                    ))

    try:
        tri = _mesh_to_trimesh(predicted_mesh)
        vertices = 128.0 / max(tri.extents.max(), 1e-6) * tri.vertices + 64.0
        vertices = vertices[:, [2, 1, 0]]
        tri = trimesh.Trimesh(vertices=vertices, faces=tri.faces, process=False)
        _add_mesh_trace(fig, tri, "MorphiNet")
    except Exception:
        pass

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="cube", camera=dict(eye=dict(x=-1, y=1, z=-1), center=dict(x=0, y=0, z=0), up=dict(x=0, y=1, z=0)),
        ),
        showlegend=False,
        template="seaborn",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return _write_figure(fig, os.path.join(export_dir, f"{case_id}_{dataset_name}_mr_slices"))
