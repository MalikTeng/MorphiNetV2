from typing import Union
from argparse import Namespace
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go

__all__ = ["draw_plotly", "draw_train_loss", "draw_eval_score"]




def extract_surface_vertices_pytorch3d(volume, isolevel=0.5):
    """Extract surface vertices using PyTorch3D marching cubes.
    
    Args:
        volume: 3D volume tensor or numpy array
        isolevel: Threshold for surface extraction
        
    Returns:
        tuple: (vertices, faces) as numpy arrays, or (empty_array, empty_array) if extraction fails
    """
    try:
        from pytorch3d.ops.marching_cubes import marching_cubes
    except ImportError:
        raise ImportError("PyTorch3D not available. Install with: pip install pytorch3d")
    
    # Convert to tensor if needed
    if isinstance(volume, np.ndarray):
        volume_tensor = torch.from_numpy(volume).float()
    elif hasattr(volume, 'float'):
        volume_tensor = volume.float()
    else:
        try:
            volume_array = np.array(volume)
            volume_tensor = torch.from_numpy(volume_array).float()
        except Exception as e:
            print(f"Failed to convert volume to tensor: {e}")
            return np.array([]), np.array([])
    
    # Ensure 4D tensor (batch dimension)
    if volume_tensor.dim() == 3:
        volume_tensor = volume_tensor.unsqueeze(0)
    
    try:
        verts, faces = marching_cubes(volume_tensor, isolevel=isolevel, return_local_coords=False)
        
        # Handle case where marching cubes returns empty results
        if isinstance(verts, list):
            if len(verts) == 0 or (len(verts) > 0 and isinstance(verts[0], list)):
                return np.array([]), np.array([])
            else:
                verts_np = verts[0].numpy()
                faces_np = faces[0].numpy()
        else:
            verts_np = verts[0].numpy()
            faces_np = faces[0].numpy()
            
        return verts_np, faces_np
        
    except Exception as e:
        print(f"PyTorch3D marching cubes failed: {e}")
        return np.array([]), np.array([])


def draw_plotly(
    seg_true: Union[Tensor, None] = None, seg_pred: Union[Tensor, None] = None, 
    df_true: Union[Tensor, None] = None, df_pred: Union[Tensor, None] = None,
    mesh_pred: Union[Meshes, None] = None, save_html: bool = False, 
    save_dir: str = None, filename: str = None, export_static: bool = True, 
    export_png_filename: str = None, **kwargs
    ):
    """Draw the plotly figure for visualization using PyTorch3D marching cubes.
    
    Args:
        seg_true: ground truth segmentation, shape (C, H, W, D)
        seg_pred: predicted segmentation, shape (C, H, W, D)
        df_true: ground truth distance field, shape (C, H, W, D)
        df_pred: predicted distance field, shape (C, H, W, D)
        mesh_pred: predicted mesh
        save_html: whether to save the figure as HTML file
        save_dir: directory to save the HTML file
        filename: name of the HTML file to save
        export_static: whether to export a static image for wandb compatibility
        export_png_filename: custom filename for the exported PNG (if different from HTML name)
    """
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])
    
    # Set up the layout with proper 3D scene configuration
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',  # This ensures proper scaling
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=20, r=20, t=30, b=20)  # Tighter margins for better rendering
    )

    if seg_true is not None:
        num_classes = len(torch.unique(seg_true))
        if num_classes == 2:
            vertices, faces = extract_surface_vertices_pytorch3d(seg_true[0].cpu().numpy(), isolevel=0.1)
        else:
            # Extract myocardium (class 2) surface
            myocardium_mask = (seg_true[0] == 2).cpu().numpy().astype(np.float32)
            vertices, faces = extract_surface_vertices_pytorch3d(myocardium_mask, isolevel=0.1)
        
        if len(vertices) > 0 and len(faces) > 0:
            x, y, z = vertices.T
            I, J, K = faces.T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color="pink",
                opacity=0.25,
                name="seg_true"
            ))
    
    if seg_pred is not None:
        num_classes = len(torch.unique(seg_pred))
        if num_classes == 2:
            vertices, faces = extract_surface_vertices_pytorch3d(seg_pred[0].cpu().numpy(), isolevel=0.1)
        else:
            # Extract myocardium (class 2) surface
            myocardium_mask = (seg_pred[0] == 2).cpu().numpy().astype(np.float32)
            vertices, faces = extract_surface_vertices_pytorch3d(myocardium_mask, isolevel=0.1)
        
        if len(vertices) > 0 and len(faces) > 0:
            x, y, z = vertices.T
            I, J, K = faces.T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color="blue",
                opacity=0.25,
                name="seg_pred"
            ))

    if mesh_pred is not None:
        assert mesh_pred._N == 1, "Only support one mesh at a time."
        # transform from NDC space to world space
        mesh_pred.offset_verts_(torch.tensor([1.0] * 3))
        # Use consistent reference size - prefer df_pred, fallback to seg_true, then seg_pred
        if df_pred is not None:
            reference_size = df_pred.shape[-1]
        elif seg_true is not None:
            reference_size = seg_true.shape[-1]
        elif seg_pred is not None:
            reference_size = seg_pred.shape[-1]
        else:
            reference_size = 32  # Default fallback size
        mesh_pred.scale_verts_(reference_size / 2)
        
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
        
        mesh_c = kwargs.get("mesh_c")
        if mesh_c is not None:
            mesh_c = mesh_c / 2 + 0.5
            mesh_c = mesh_c * reference_size
            for i, center in enumerate(mesh_c):
                fig.add_trace(go.Scatter3d(
                    x=[center[0].item()], y=[center[1].item()], z=[center[2].item()],
                    mode="markers", marker=dict(size=5, color="blue"),
                    name=f"mesh_c{i}"
                ))

    if df_pred is not None:
        # draw the zero-level set from the df_pred
        df_mask = (df_pred[-1].cpu().numpy() <= 1).astype(np.float32)
        vertices, faces = extract_surface_vertices_pytorch3d(df_mask, isolevel=0.1)
        
        if len(vertices) > 0 and len(faces) > 0:
            x, y, z = vertices.T
            I, J, K = faces.T
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color="gray",
                opacity=0.25,
                name="df_pred"
            ))


        if seg_true is not None:
            # plot the center of lv and rv from the distance field
            for i, name in zip([1, 2], ["lv", "rv"]):
                center = torch.nonzero(df_pred[i] <= 1).float().mean(0)
                fig.add_trace(go.Scatter3d(
                    x=[center[2].item()], y=[center[1].item()], z=[center[0].item()],
                    mode="markers", marker=dict(size=5, color="blue"),
                    name=f"center_{name}"
                ))
        
    # Save the figure as HTML if requested
    if save_html and save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        file_name = filename or "plotly_figure.html"
        html_path = os.path.join(save_dir, file_name)
        fig.write_html(html_path)
        
        # Also export as static image for wandb compatibility
        if export_static:
            # Use custom PNG filename if provided, otherwise derive from HTML filename
            png_filename = export_png_filename or file_name.replace('.html', '.png')
            img_path = os.path.join(save_dir, png_filename)
            try:
                # Try to export as static image using plotly's built-in functionality
                fig.write_image(img_path, scale=2)
            except Exception as e:
                print(f"Failed to save static image: {e}")
                # Fallback to matplotlib if plotly export fails
                try:
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D
                    
                    # Create a simple 3D plot with matplotlib to serve as a fallback
                    plt.figure(figsize=(10, 10))
                    ax = plt.subplot(111, projection='3d')
                    
                    # Add a title to identify the contents
                    plot_title = "3D Visualization"
                    if "seg_true" in locals() and seg_true is not None:
                        plot_title += " - Segmentation"
                    if "mesh_pred" in locals() and mesh_pred is not None:
                        plot_title += " - Mesh"
                    if "df_pred" in locals() and df_pred is not None:
                        plot_title += " - Distance Field"
                    
                    ax.set_title(plot_title)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    
                    # Add a text note about the interactive version
                    plt.figtext(0.5, 0.01, "Interactive 3D visualization available in HTML file", 
                                ha='center', fontsize=10)
                    
                    plt.savefig(img_path, dpi=200, bbox_inches='tight')
                    plt.close()
                except Exception as e2:
                    print(f"Failed to save fallback image: {e2}")
        
    return fig


def draw_train_loss(train_loss: dict, super_params: Namespace, task_code: str, phase: str, ckpt_dir=None):
    sns.set_theme(style="whitegrid")
    _, ax = plt.subplots(figsize=(10, 8))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    df = pd.DataFrame(train_loss)
    df.set_index(df.index + 1, inplace=True)
    if phase == "gsn":
        lambda_ = [super_params.lambda_0, super_params.lambda_1]
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

    # Use either provided ckpt_dir or construct it from super_params
    save_path = f"{ckpt_dir}/{phase}_loss.png" if ckpt_dir else f"{super_params.ckpt_dir}/{task_code}/{super_params.run_id}/{phase}_loss.png"
    plt.savefig(save_path)


def draw_eval_score(eval_score: dict, super_params: Namespace, task_code: str, module: str, ckpt_dir=None):
    df = pd.DataFrame(eval_score)
    df["Epoch"] = super_params.train_epochs + (df.index + 1) * super_params.val_interval
    df_melted = df.melt(id_vars="Epoch", var_name="Label", value_name="Score")
    mean_scores = df.drop("Epoch", axis=1).mean(axis=1)
    mean_scores.name = 'Average Score'
    sns.set_theme(style="whitegrid")
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
    
    # Use either provided ckpt_dir or construct it from super_params
    save_path = f"{ckpt_dir}/eval_{module}_score.png" if ckpt_dir else f"{super_params.ckpt_dir}/{task_code}/{super_params.run_id}/eval_{module}_score.png"
    plt.savefig(save_path)
