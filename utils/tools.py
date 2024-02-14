from typing import Union
from matplotlib import axis
from monai.utils.misc import str2list
import torch
import numpy as np
import pandas as pd
from argparse import Namespace
from torch.nn.functional import one_hot
from torch import Tensor
from pytorch3d.structures import Meshes, Pointclouds
from trimesh.voxel.ops import matrix_to_marching_cubes
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go


__all__ = ["draw_plotly", "draw_train_loss", "draw_eval_score"]

def draw_plotly(
    image: Union[Tensor, None] = None, seg_true: Union[Tensor, None] = None, 
    seg_pred: Union[Tensor, None] = None, mesh_pred: Union[Meshes, None] = None 
    ):
    fig = make_subplots(rows=1, cols=1)
    if image is not None:
        # Add image as cross sections, perpendicular to each other.
        assert image.shape[0] == 1, "Only support one image at a time."
        image = image.squeeze()   # remove batch and channel dimension
        X, Y, Z = image.shape
        _, _, z_grid = np.mgrid[0:X, 0:Y, 0:Z]
        fig.add_traces([
            go.Surface(
                z=Z // 2 * np.ones((X, Y)), surfacecolor=image[..., Z // 2].T, 
                colorscale="Gray", cmin=0, cmax=1,
                showscale=False
                ),
            go.Surface(
                z=z_grid[0], x=X // 2 * np.ones((Y, Z)), surfacecolor=image[X // 2], 
                colorscale="Gray", cmin=0, cmax=1,
                showscale=False
                ),
        ])

    if seg_true is not None:
        assert seg_true.shape[0] == 1, "Only support one seg_true at a time."
        num_classes = torch.unique(seg_true).shape[0]
        if num_classes == 2:
            mesh = matrix_to_marching_cubes(seg_true.squeeze().cpu().numpy())
            x, y, z = mesh.vertices.T
            I, J, K = mesh.faces.T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color="pink",
                opacity=0.25,
                name="seg_true"
            ))
        else:
            raise ValueError("Only support binary segmentation for now.")
    
    if seg_pred is not None:
        assert seg_pred.shape[0] == 1, "Only support one prediction at a time."
        num_classes = torch.unique(seg_pred).shape[0]
        if num_classes == 2:
            mesh = matrix_to_marching_cubes(seg_pred.squeeze().cpu().numpy())
            x, y, z = mesh.vertices.T
            I, J, K = mesh.faces.T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color="blue",
                opacity=0.25,
                name="seg_pred"
            ))
        else:
            raise ValueError("Only support binary segmentation for now.")

    if mesh_pred is not None:
        assert mesh_pred._N == 1, "Only support one mesh at a time."
        # rescale the pred mesh to fit the seg_true
        mesh_pred.scale_verts_(seg_true.shape[-1])
        mesh_pred.offset_verts_(torch.tensor([seg_true.shape[-1] / 2] * 3))
        for mesh in mesh_pred:
            x, y, z = mesh.verts_packed().T
            I, J, K = mesh.faces_packed().T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color="blue",
                opacity=0.1,
                name="meshes_pred"
            ))

    return fig

def draw_train_loss(train_loss: dict, super_params: Namespace, task_code: str, phase: str):
    sns.set_theme(style="whitegrid")
    _, ax = plt.subplots(figsize=(10, 8))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    df = pd.DataFrame(train_loss)
    df.set_index(df.index + 1, inplace=True)
    if phase == "subdiv":
        for i, coeff in enumerate(super_params.lambda_, start=1):
            df.iloc[:, i] = df.iloc[:, i - 1] - coeff * df.iloc[:, i]

    if len(df) > 0:
        colors = sns.color_palette("hls", len(df.columns.values))
        for i in range(len(df.columns.values) - 1):
            ax = sns.lineplot(
                x=df.index.values, y=df.iloc[:, i], 
                ax=ax, color=colors[i], label=df.columns[i+1]
            )
            curve = ax.lines[i]
            x_i = curve.get_xydata()[:, 0]
            y_i = curve.get_xydata()[:, 1]
            ax.fill_between(x_i, y_i, color=colors[i], alpha=0.6)
        plt.legend()

    plt.savefig(f"{super_params.ckpt_dir}/{task_code}/{super_params.run_id}/{phase}_loss.png")


def draw_eval_score(eval_score: dict, super_params: Namespace, task_code: str, module: str):
    df = pd.DataFrame(eval_score)
    df["Epoch"] = super_params.delay_epochs + (df.index + 1) * super_params.val_interval
    df_melted = df.melt(id_vars="Epoch", var_name="Label", value_name="Score")
    mean_scores = df.drop("Epoch", axis=1).mean(axis=1)
    mean_scores.name = 'Average Score'
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(x=df_melted["Epoch"], y=df_melted["Score"], ax=ax, color="skyblue", showfliers=False, width=0.2)
    sns.lineplot(x=mean_scores.index.values, y=mean_scores, ax=ax, color="green", label="Average")
    LOW = df.drop("Epoch", axis=1).idxmin(axis=1)
    HIGH = df.drop("Epoch", axis=1).idxmax(axis=1)
    for epoch, (l, h) in enumerate(zip(LOW, HIGH)):
        ax.text(epoch, df.loc[epoch, l], f'{l}', horizontalalignment="center", color="black", weight="semibold")
        ax.text(epoch, df.loc[epoch, h], f'{h}', horizontalalignment="center", color="black", weight="semibold")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.savefig(f"{super_params.ckpt_dir}/{task_code}/{super_params.run_id}/eval_{module}_score.png")