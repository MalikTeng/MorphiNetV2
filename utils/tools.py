from typing import Union
from matplotlib import axis
from monai.utils.misc import str2list
import torch
import numpy as np
import pandas as pd
from argparse import Namespace
from torch import Tensor
import torch.nn.functional as F
from torch.nn.functional import one_hot
from pytorch3d.structures import Meshes, Pointclouds
from trimesh.voxel.ops import matrix_to_marching_cubes
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go


__all__ = ["draw_plotly", "draw_train_loss", "draw_eval_score"]

def draw_plotly(
    seg_true: Union[Tensor, None] = None, seg_pred: Union[Tensor, None] = None, 
    df_true: Union[Tensor, None] = None, df_pred: Union[Tensor, None] = None,
    mesh_pred: Union[Meshes, None] = None 
    ):
    fig = make_subplots(rows=1, cols=1)

    if seg_true is not None:
        num_classes = len(torch.unique(seg_true))
        if num_classes == 2:
            mesh = matrix_to_marching_cubes(seg_true[0].cpu().numpy())
            y, x, z = mesh.vertices.T
            I, J, K = mesh.faces.T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color="pink",
                opacity=0.25,
                name="seg_true"
            ))
        else:
            print("ERROR: Only support binary segmentation for now.")
    
    if seg_pred is not None:
        num_classes = len(torch.unique(seg_pred))
        if num_classes == 2:
            mesh = matrix_to_marching_cubes(seg_pred[0].cpu().numpy())
            y, x, z = mesh.vertices.T
            I, J, K = mesh.faces.T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color="blue",
                opacity=0.25,
                name="seg_pred"
            ))
        else:
            print("ERROR: Only support binary segmentation for now.")

    if mesh_pred is not None:
        assert mesh_pred._N == 1, "Only support one mesh at a time."
        # transform from NDC space to world space
        mesh_pred.offset_verts_(torch.tensor([1.0] * 3))
        if df_pred is None:
            mesh_pred.scale_verts_(seg_true.shape[-1] / 2)
        else:
            mesh_pred.scale_verts_(df_pred.shape[-1] / 2)
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

    if df_pred is not None:
        # compute the gradient of the signed distance field
        grad_pred = torch.gradient(-df_pred, dim=(1, 2, 3), edge_order=1)
        grad_pred = torch.stack(grad_pred, dim=1)
        grad_pred /= torch.norm(grad_pred, dim=1, keepdim=True)
        # mute any nan/inf values in label_grad
        grad_pred[torch.isnan(grad_pred)] = 0
        grad_pred[torch.isinf(grad_pred)] = 0

        # draw the zero-level set from the df_pred
        mesh = matrix_to_marching_cubes((df_pred[0].cpu().numpy() < 1.2).astype(int))
        verts, faces = mesh.vertices, mesh.faces
        if seg_true is not None:
            verts *= seg_true.shape[-1] / df_pred.shape[-1]
        y, x, z = verts.T
        I, J, K = faces.T
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=I, j=J, k=K,
            color="gray",
            opacity=0.25,
            name="df_pred"
        ))

        if mesh_pred is not None:
            # compute the grad respective to the mesh vertices
            grad_pred *= df_pred.unsqueeze(1)
            verts = mesh_pred.verts_padded()
            verts = 2 * (verts / df_pred.shape[-1] - 0.5)   # pixel space to NDC space
            mesh_grad = F.grid_sample(
                grad_pred.permute(0, 1, 4, 2, 3).float(),           # (N, C: yxz, D, H, W)
                verts.unsqueeze(1).unsqueeze(1),                    # (N, 1, 1, V, 3: xyz)
                align_corners=False
            ).view(1, 3, -1).transpose(-1, -2)[..., [1, 0, 2]]      # (N, C: yxz, 1, 1, V) -> (N, V, C: xyz)
            verts = (verts / 2 + 0.5) * df_pred.shape[-1]   # NDC space to pixel space
            x, y, z = verts[0].numpy().T
            u, v, w = mesh_grad[0].numpy().T
            fig.add_trace(go.Cone(
                x=x, y=y, z=z,
                u=u, v=v, w=w,
                colorscale="Viridis", sizemode="scaled", sizeref=1, showscale=True
            ))

    if mesh_pred is not None or seg_pred is not None:
        fig.write_html(f"{'seg_true vs mesh_pred' if mesh_pred is not None else 'seg_true vs seg_pred'}.html")

    return fig

def draw_train_loss(train_loss: dict, super_params: Namespace, task_code: str, phase: str):
    sns.set_theme(style="whitegrid")
    _, ax = plt.subplots(figsize=(10, 8))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    df = pd.DataFrame(train_loss)
    df.set_index(df.index + 1, inplace=True)
    if phase == "gsn":
        lambda_ = super_params.lambda_
    else:
        lambda_ = [1]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    if len(df) > 1:
        for i, coeff in enumerate(lambda_, start=1):
            df.iloc[:, i] = df.iloc[:, i - 1] - coeff * df.iloc[:, i]
        colors = sns.color_palette("hls", len(df.columns.values))
        for i in range(len(df.columns.values) - 1):
            ax = sns.lineplot(
                x=df.index.values, y=df.iloc[:, i].values, 
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
    df["Epoch"] = super_params.train_epochs + (df.index + 1) * super_params.val_interval
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