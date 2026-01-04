"""
    a cnn-mlp decoder module mapping medical images to signed distance fields in end-to-end manner, segmentation from cnn-based encoder serves as the common pre-text task for the decoder. 

    required preprocessing:
    1. down-sample the CMR images to have isotropic voxel size, including
        1.1. resample to have isotropic voxel size,
        1.2. crop the foreground of the resampled data,
        1.3. resize the cropped data to have the same size 16 x 16 x 16.
    2. down-sample the CTA images & labels to have isotropic voxel size, including
        2.0. mask CTA images near the basal and apex plane
        2.1. resample to have the same resolution as CMR,
        2.2. crop the foreground,
        2.3. resize the cropped data to have the same size 16 x 16 x 16.
    3. compute the signed distance fields from the ground truth segmentation using edt package, https://github.com/seung-lab/euclidean-distance-transform-3d.
"""
from typing import List
import torch
import torch.nn as nn
from torch.nn import Module, ModuleList, Linear, LayerNorm, LeakyReLU, Sequential
from monai.networks.layers.factories import Conv, Norm
from pytorch3d.ops import taubin_smoothing
from pytorch3d.structures import Meshes
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
)
from torch_geometric.nn import MessagePassing, DeepGCNLayer, GCNConv
from torch_geometric.nn.dense.linear import Linear as DenseLinear
from torch_geometric.utils import add_self_loops,degree

from .parts import ResNetBlock, ResNetBottleneck

__all__ = ["GSN", "Subdivision", "LocalMeshWarper", "UpscalingResNet"]


# function for pre-computed faces index
@ torch.no_grad()
class Subdivision():
    def __init__(self, 
                 mesh: Meshes, num_layers: int,
                 mesh_label: torch.LongTensor=None,
                 allow_subdiv_faces: List[torch.LongTensor]=[None, None]
                 ) -> list:
        
        self.faces_levels = []
        self.labels_levels = []
        for l in range(num_layers):
            new_faces = self.subdivide_faces_fn(mesh, allow_subdiv_faces[l])
            self.faces_levels.append(new_faces)
            verts = mesh.verts_packed()
            edges = mesh.edges_packed()
            new_verts = verts[edges].mean(dim=1)
            new_verts = torch.cat([verts, new_verts], dim=0)
            mesh = Meshes(verts=[new_verts], faces=[new_faces])

            # mesh_label = mesh_label.tile([4]) if mesh_label is not None else None
            # self.labels_levels.append(mesh_label)
            mesh_label = torch.cat([mesh_label, mesh_label[edges].max(dim=1).values], dim=0)
            self.labels_levels.append(mesh_label)

    def subdivide_faces_fn(self, mesh: Meshes, allow_subdiv_faces: torch.LongTensor=None):
        verts_packed = mesh.verts_packed()
        faces_packed = mesh.faces_packed()
        faces_packed_to_edges_packed = (
            verts_packed.shape[0] + mesh.faces_packed_to_edges_packed()
        )
        if allow_subdiv_faces is not None:
            faces_packed = faces_packed[allow_subdiv_faces]
            faces_packed_to_edges_packed = faces_packed_to_edges_packed[allow_subdiv_faces]

        f0 = torch.stack([
            faces_packed[:, 0],                     # 0
            faces_packed_to_edges_packed[:, 2],     # 3
            faces_packed_to_edges_packed[:, 1],     # 4
        ], dim=1)
        f1 = torch.stack([
            faces_packed[:, 1],                     # 1
            faces_packed_to_edges_packed[:, 0],     # 5
            faces_packed_to_edges_packed[:, 2],     # 3
        ], dim=1)
        f2 = torch.stack([
            faces_packed[:, 2],                     # 2
            faces_packed_to_edges_packed[:, 1],     # 4
            faces_packed_to_edges_packed[:, 0],     # 5
        ], dim=1)
        f3 = faces_packed_to_edges_packed           # 5, 4, 3

        subdivided_faces_packed = torch.cat([f0, f1, f2, f3], dim=0)

        if allow_subdiv_faces is not None:
            subdivided_faces_packed = torch.cat(
                [mesh.faces_packed()[~allow_subdiv_faces], subdivided_faces_packed], dim=0
            )

        return subdivided_faces_packed
    

class GSNLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.hidden_features = kwargs.get("hidden_features", 16)

        self.lin = Sequential(
            DenseLinear(in_channels, self.hidden_features, bias=False,
                   weight_initializer='glorot'),
            LayerNorm(self.hidden_features),
            LeakyReLU(inplace=True),
            DenseLinear(self.hidden_features, self.hidden_features, bias=False,
                   weight_initializer='glorot'),
            LayerNorm(self.hidden_features),
            LeakyReLU(inplace=True),
            DenseLinear(self.hidden_features, out_channels, bias=False,
                   weight_initializer='glorot')
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.lin:
            if isinstance(m, DenseLinear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        # Step 1: add self-loops to the edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: normalisation
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 3: propagating messages
        x = self.propagate(edge_index, x=x, norm=norm)

        return x

    def message(self, x_i, x_j, norm):
        # Step 1: h_{\theta}(x_j - x_i)
        f = self.lin(x_j - x_i)
        return norm.view(-1, 1) * f if norm is not None else f


class LocalMeshWarper(nn.Module):
    """
    Network component for applying local offset warping to mesh vertices based on distance fields.
    This implements the local offset stage from the mesh warping pipeline.
    """
    def __init__(self, num_iterations: int = 30):
        super().__init__()
        self.num_iterations = num_iterations
    
    def sample_gradient_field(self, direction, mesh_vertices):
        """
        Manual trilinear interpolation for gradient field sampling with torch.gradient compatibility.
        
        Replaces F.grid_sample to maintain differentiability through torch.gradient operations.
        Implements padding_mode="zeros" behavior for out-of-bounds coordinates.
        
        Args:
            direction: [B, C, H, W, D] - gradient field with channels [dH, dW, dD]
            mesh_vertices: [B, N, 3] - vertex positions in NDC coordinates [X, Y, Z]
            
        Returns:
            sampled_directions: [B, N, 3] - interpolated vectors in [dX, dY, dZ] order
        """
        B, C, H, W, D = direction.shape
        N = mesh_vertices.shape[1]
        
        # Step 1: Convert NDC [-1,1] to continuous volume indices [0,size-1]
        indices = (mesh_vertices + 1) * torch.tensor([D-1, W-1, H-1], device=mesh_vertices.device) / 2
        
        # Check for out-of-bounds vertices (padding_mode="zeros")
        out_of_bounds = (mesh_vertices < -1).any(dim=-1) | (mesh_vertices > 1).any(dim=-1)  # [B, N]
        
        # Step 2: Extract floor indices and interpolation weights
        indices_floor = torch.floor(indices).long()
        indices_floor[..., 0] = torch.clamp(indices_floor[..., 0], 0, D-1)
        indices_floor[..., 1] = torch.clamp(indices_floor[..., 1], 0, W-1)
        indices_floor[..., 2] = torch.clamp(indices_floor[..., 2], 0, H-1)
        
        indices_ceil = indices_floor + 1
        indices_ceil[..., 0] = torch.clamp(indices_ceil[..., 0], 0, D-1)
        indices_ceil[..., 1] = torch.clamp(indices_ceil[..., 1], 0, W-1)
        indices_ceil[..., 2] = torch.clamp(indices_ceil[..., 2], 0, H-1)
        
        weights = indices - indices_floor.float()
        
        # Step 3: Extract indices and weights
        d0, w0, h0 = indices_floor[..., 0], indices_floor[..., 1], indices_floor[..., 2]
        d1, w1, h1 = indices_ceil[..., 0], indices_ceil[..., 1], indices_ceil[..., 2]
        wd, ww, wh = weights[..., 0], weights[..., 1], weights[..., 2]
        
        # Step 4: Efficient trilinear interpolation per batch
        result = torch.zeros(B, N, C, device=direction.device, dtype=direction.dtype)
        
        for b in range(B):
            # Sample 8 corners for batch b [N, C]
            c000 = direction[b, :, h0[b], w0[b], d0[b]].T
            c001 = direction[b, :, h0[b], w0[b], d1[b]].T
            c010 = direction[b, :, h0[b], w1[b], d0[b]].T
            c011 = direction[b, :, h0[b], w1[b], d1[b]].T
            c100 = direction[b, :, h1[b], w0[b], d0[b]].T
            c101 = direction[b, :, h1[b], w0[b], d1[b]].T
            c110 = direction[b, :, h1[b], w1[b], d0[b]].T
            c111 = direction[b, :, h1[b], w1[b], d1[b]].T
            
            # Trilinear interpolation: lerp(lerp(lerp(edges, D), W), H)
            wd_b, ww_b, wh_b = wd[b][:, None], ww[b][:, None], wh[b][:, None]
            
            # Interpolate along D dimension
            c00 = c000 * (1 - wd_b) + c001 * wd_b
            c01 = c010 * (1 - wd_b) + c011 * wd_b
            c10 = c100 * (1 - wd_b) + c101 * wd_b
            c11 = c110 * (1 - wd_b) + c111 * wd_b
            
            # Interpolate along W dimension
            c0 = c00 * (1 - ww_b) + c01 * ww_b
            c1 = c10 * (1 - ww_b) + c11 * ww_b
            
            # Interpolate along H dimension
            result[b] = c0 * (1 - wh_b) + c1 * wh_b
            
            # Apply padding_mode="zeros" for out-of-bounds vertices
            result[b][out_of_bounds[b]] = 0.0
        
        # Step 5: Reorder channels from [dH, dW, dD] to [dX, dY, dZ]
        sampled_directions = result[..., [2, 1, 0]]
        
        return sampled_directions
    
    # def mesh_quality(self, meshes):
    #     import trimesh
    #     import pyvista as pv
    #     import numpy as np
    #     mesh = trimesh.Trimesh(
    #         vertices=meshes.verts_packed().cpu().numpy(),
    #         faces=meshes.faces_packed().cpu().numpy(),
    #     )
    #     mesh_pv = pv.wrap(mesh)
        
    #     aspect = np.median(np.array(mesh_pv.compute_cell_quality('aspect_ratio').active_scalars))
    #     jacobian_values = np.array(mesh_pv.compute_cell_quality('scaled_jacobian').active_scalars)
    #     jacobian = np.median(jacobian_values)

    #     print(f"DEBUG: mesh_quality: aspect={aspect}, jacobian={jacobian}")

    def forward(self, meshes, df_preds, vert_labels):
        """
        Apply local offset warping to mesh vertices based on distance fields.
        
        Args:
            meshes: PyTorch3D Meshes object containing the meshes to warp
            df_preds: Distance field predictions of shape (B, C, D, H, W)
            vert_labels: Vertex labels tensor for the current mesh subdivision level
            
        Returns:
            Warped meshes with updated vertices
        """
        b, *_, d = df_preds.shape
        verts = meshes.verts_padded()
        
        # Use the same precision as the mesh vertices (AMP compatible)
        verts_dtype = verts.dtype
        device = verts.device

        # Process both LV and RV related vertices for LV+RV template mesh
        for i, l in zip([1, 0, 2, 0], [[0], [2], [1], [3]]):  # lv-endo, lv-epi, rv-endo, rv-epi
            df_pred = df_preds[:, i].to(dtype=verts_dtype, device=device)
            verts_idx = torch.any(torch.stack([vert_labels == j for j in l]), dim=0)

            # Skip if no vertices match the current label
            if not verts_idx.any():
                continue

            # Calculate the gradient of the distance field
            direction = torch.gradient(-df_pred, dim=(1, 2, 3), edge_order=1)
            direction = torch.stack(direction, dim=1)
            
            # Calculate the norm of each direction vector
            direction_norm = torch.norm(direction, dim=1, keepdim=True)
            
            # Only normalize vectors with norm > 1, keep vectors with norm <= 1 unchanged
            mask = (direction_norm > 1.0)
            direction = torch.where(mask, direction / (direction_norm + 1e-16), direction)
            
            # Handle any invalid values
            direction[torch.isnan(direction)] = 0
            direction[torch.isinf(direction)] = 0
            
            # Apply iterative offset
            for _ in range(self.num_iterations):
                # Use manual trilinear interpolation instead of F.grid_sample for torch.gradient compatibility
                mesh_vertices = verts[:, verts_idx].to(dtype=verts_dtype)  # [B, N, 3] in NDC space
                offset = self.sample_gradient_field(direction.to(dtype=verts_dtype), mesh_vertices)  # [B, N, 3]

                # Transform from NDC space to pixel space
                verts = d * (verts / 2 + 0.5)
                verts[:, verts_idx] += offset
                # Transform verts back to NDC space
                verts = 2 * (verts / d - 0.5)

        meshes = meshes.update_padded(verts)

        return meshes


class GSN(nn.Module):
    def __init__(self, hidden_features: int, num_layers: int = 2, num_iterations: int = 30):
        super().__init__()

        self.gcn_layers = ModuleList([
            GSNLayer(3, 3, bias=False, hidden_features=hidden_features)
            for _ in range(num_layers)
        ])
        
        # Initialize mesh warper
        self.mesh_warper = LocalMeshWarper(num_iterations)

    def forward(self, meshes: Meshes, subdivided_faces: list[torch.LongTensor], df_preds: torch.Tensor = None, labels_levels: list[torch.LongTensor] = None):
        # Ensure all vertices are in the correct data type for AMP compatibility
        verts_precision = next(self.parameters()).dtype
        if meshes.verts_padded().dtype != verts_precision:
            meshes = meshes.update_padded(meshes.verts_padded().to(verts_precision))
        
        level_outs = []
        for l, gcn_layer in enumerate(self.gcn_layers):
            if len(subdivided_faces) > 0:
                # 1. create new vertices at the middle of the edges.
                new_faces = subdivided_faces[l].expand(meshes._N, -1, -1).to(meshes.device)
                verts = meshes.verts_padded()
                edges = meshes[0].edges_packed()
                edge_verts = verts[:, edges].mean(dim=2)
                new_verts = torch.cat([verts, edge_verts], dim=1)
            
            else:
                new_verts = meshes.verts_padded()
                new_faces = meshes.faces_padded()

            # 2. create new meshes with the same topology as the original mesh.
            meshes = Meshes(verts=new_verts, faces=new_faces)
            
            # 3. update the vertices with learnt offsets.
            offsets = gcn_layer(
                meshes.verts_packed(), meshes.edges_packed().t().contiguous()
                )
            meshes = meshes.offset_verts(offsets)

            # 4. output the new mesh
            level_outs.append(meshes)

        # 5. Apply mesh warping post-processing during evaluation
        if not self.training and df_preds is not None and labels_levels is not None:
            warped_level_outs = []
            for l, meshes in enumerate(level_outs):
                warped_meshes = self.mesh_warper(meshes, df_preds, labels_levels[l])
                warped_meshes = taubin_smoothing(warped_meshes, 0.5, -0.53, 10)
                warped_level_outs.append(warped_meshes)
            level_outs = warped_level_outs

        return level_outs


class UpscalingResNet(nn.Module):
    """
    Custom ResNet decoder that can upscale segmentation predictions by a specified ratio.
    Uses transpose convolutions for upscaling while maintaining segmentation quality.
    
    Args:
        spatial_dims: number of spatial dimensions (2 or 3)
        in_channels: number of input channels
        out_channels: number of output channels
        upscale_ratio: upscaling factor for output compared to input
        layers: list/tuple of number of residual blocks per layer (length determines number of layers)
        act: activation type and arguments
        norm: normalization type and arguments
    """
    
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 4,
        out_channels: int = 4,
        upscale_ratio: int = 2,
        layers: tuple = (1, 2, 2, 4),
        act: tuple = ("leakyrelu", {"inplace": True, "negative_slope": 0.1}),
        norm: tuple = ("INSTANCE", {"affine": True}),
    ):
        super().__init__()
        
        self.spatial_dims = spatial_dims
        self.upscale_ratio = upscale_ratio
        self.num_encoder_layers = len(layers)
        
        # Get appropriate layer types based on spatial dimensions
        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_type = Norm[norm[0], spatial_dims]
        
        # Handle activation type properly
        if isinstance(act[1], dict):
            if act[0].lower() == "leakyrelu":
                act_type = nn.LeakyReLU(**act[1])
            else:
                act_type = getattr(nn, act[0])(**act[1])
        else:
            if act[0].lower() == "leakyrelu":
                act_type = nn.LeakyReLU()
            else:
                act_type = getattr(nn, act[0])()
        
        # Create a fresh activation for each use
        def get_activation():
            if isinstance(act[1], dict):
                if act[0].lower() == "leakyrelu":
                    return nn.LeakyReLU(**act[1])
                else:
                    return getattr(nn, act[0])(**act[1])
            else:
                if act[0].lower() == "leakyrelu":
                    return nn.LeakyReLU()
                else:
                    return getattr(nn, act[0])()
        
        # Encoder layers (downsampling) - use user-defined layers parameter
        self.encoder_layers = nn.ModuleList()
        current_channels = in_channels
        
        # Initial conv
        initial_out_channels = 32  # Start with smaller channels, will be scaled up
        self.initial_conv = nn.Sequential(
            conv_type(current_channels, initial_out_channels, kernel_size=3, padding=1, bias=False),
            norm_type(initial_out_channels),
            act_type
        )
        current_channels = initial_out_channels
        
        # Dynamic channel calculation based on number of layers
        # Start with initial_out_channels and double for each layer
        layer_channels = [initial_out_channels * (2 ** i) for i in range(self.num_encoder_layers)]
        
        # Ensure we don't go above reasonable channel limits
        max_channels = 512
        layer_channels = [min(ch, max_channels) for ch in layer_channels]
        
        for i, (num_blocks, out_ch) in enumerate(zip(layers, layer_channels)):
            layer = []
            stride = 2 if i > 0 else 1  # First layer doesn't downsample
            
            # First block with potential downsampling
            layer.append(self._make_residual_block(
                current_channels, out_ch, stride, norm_type, get_activation, spatial_dims
            ))
            current_channels = out_ch
            
            # Additional blocks based on layers parameter
            for _ in range(num_blocks - 1):
                layer.append(self._make_residual_block(
                    current_channels, out_ch, 1, norm_type, get_activation, spatial_dims
                ))
            
            self.encoder_layers.append(nn.Sequential(*layer))
        
        # Decoder layers (upsampling)
        self.decoder_layers = nn.ModuleList()
        
        # Calculate total upsampling needed
        # encoder downsamples by 2^(num_encoder_layers-1), decoder needs to upsample by that * upscale_ratio
        encoder_downsample_factor = 2 ** (self.num_encoder_layers - 1)
        total_upsample_factor = encoder_downsample_factor * upscale_ratio
        
        # Number of upsampling layers needed
        num_upsample_layers = int(torch.log2(torch.tensor(total_upsample_factor, dtype=torch.float32)).item())
        
        # Decoder upsampling layers - reverse the encoder channel progression
        decoder_channels = layer_channels[::-1][1:] + [initial_out_channels // 2]  # Reverse and add final smaller channel
        
        # Ensure we have enough decoder channels
        while len(decoder_channels) < num_upsample_layers:
            decoder_channels.append(decoder_channels[-1] // 2 if decoder_channels[-1] > 16 else 16)
        
        # Take only the number we need
        decoder_channels = decoder_channels[:num_upsample_layers]
        
        for i in range(num_upsample_layers):
            in_ch = current_channels if i == 0 else decoder_channels[i-1]
            out_ch = decoder_channels[i] if i < len(decoder_channels) else 16
            
            layer = []
            # Transpose convolution for upsampling
            if spatial_dims == 3:
                layer.append(nn.ConvTranspose3d(
                    in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
                ))
            else:
                layer.append(nn.ConvTranspose2d(
                    in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
                ))
            
            layer.extend([
                norm_type(out_ch),
                get_activation(),
                # Additional residual block for better quality
                self._make_residual_block(out_ch, out_ch, 1, norm_type, get_activation, spatial_dims)
            ])
            
            self.decoder_layers.append(nn.Sequential(*layer))
            current_channels = out_ch
        
        # Final output layer
        self.final_conv = conv_type(current_channels, out_channels, kernel_size=3, padding=1)
    
    def _make_residual_block(self, in_channels, out_channels, stride, norm_type, get_activation, spatial_dims):
        """Create a residual block"""
        conv_type = Conv[Conv.CONV, spatial_dims]
        
        layers = [
            conv_type(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            norm_type(out_channels),
            get_activation(),
            conv_type(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_type(out_channels),
        ]
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                conv_type(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_type(out_channels),
            )
        else:
            shortcut = nn.Identity()
        
        return ResidualBlock(nn.Sequential(*layers), shortcut, get_activation())
    
    def forward(self, x):
        # Store input shape for potential skip connections
        input_shape = x.shape
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Encoder (downsampling)
        skip_connections = []
        for layer in self.encoder_layers:
            skip_connections.append(x)
            x = layer(x)
        
        # Decoder (upsampling)
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            # Optional: add skip connections from encoder
            # if i < len(skip_connections):
            #     skip = skip_connections[-(i+1)]
            #     # Resize skip connection to match current x
            #     if skip.shape[2:] != x.shape[2:]:
            #         skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear' if self.spatial_dims==3 else 'bilinear')
            #     x = x + skip
        
        # Final output
        x = self.final_conv(x)
        
        return x


class ResidualBlock(nn.Module):
    """Simple residual block helper"""
    def __init__(self, main_path, shortcut, activation):
        super().__init__()
        self.main_path = main_path
        self.shortcut = shortcut
        self.activation = activation
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.main_path(x)
        return self.activation(out + residual)
