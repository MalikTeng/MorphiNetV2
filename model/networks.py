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
import torch
import torch.nn as nn

from collections.abc import Callable
from functools import partial
from typing import Any, Dict, Union

from monai.networks.layers.factories import Conv, Norm, Pool
from monai.networks.layers.utils import get_pool_layer
from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option

from .parts import ResNetBlock, ResNetBottleneck

__all__ = ["ResNet", "AutoEncoder", "GSN", "Subdivision"]


class AutoEncoder(nn.Module):
    """
        concate the encoder and decoder to predict the distance field. down-sample should be applied to scale the output segmentation into sizes matching the target distance field.
    """
    def __init__(self, encoder, decoder, downsample_scale, spatial_dims=3) -> None:
        super().__init__()

        conv_type: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        norm_type: Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d] = Norm[Norm.BATCH, spatial_dims]

        self.encoder = encoder
        self.decoder = decoder

        num_layers = torch.log2(torch.tensor(downsample_scale)).int().item()
        ds_layer = []
        for _ in range(num_layers):
            ds_layer.extend([
                conv_type(
                    encoder.out_channels, encoder.out_channels, 
                    kernel_size=3, stride=2, padding=1, bias=True
                    ),
                norm_type(encoder.out_channels),
                ])
        self.ds_block = nn.Sequential(*ds_layer)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_seg = self.encoder(x)
        x_df = self.ds_block(x_seg)
        x_df = self.decoder(x_df)

        return x_df, x_seg


class ResNet(nn.Module):
    """
    ResNet based on: `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? <https://arxiv.org/pdf/1711.09577.pdf>`_.
    Adapted from `<https://github.com/kenshohara/3D-ResNets-PyTorch/tree/master/models>`_.

    Args:
        block: which ResNet block to use, either Basic or Bottleneck.
            ResNet block class or str.
            for Basic: ResNetBlock or 'basic'
            for Bottleneck: ResNetBottleneck or 'bottleneck'
        layers: how many layers to use.
        block_inplanes: determine the size of planes at each step. Also tunable with widen_factor.
        spatial_dims: number of spatial dimensions of the input image.
        n_input_channels: number of input channels for first convolutional layer.
        conv1_t_size: size of first convolution layer, determines kernel and padding.
        conv1_t_stride: stride of first convolution layer.
        no_max_pool: bool argument to determine if to use maxpool layer.
        shortcut_type: which downsample block to use. Options are 'A', 'B', default to 'B'.
            - 'A': using `self._downsample_basic_block`.
            - 'B': kernel_size 1 conv + norm.
        widen_factor: widen output for each layer.
        num_classes: number of output (classifications).
        feed_forward: whether to add the FC layer for the output, default to `True`.
        bias_downsample: whether to use bias term in the downsampling block when `shortcut_type` is 'B', default to `True`.

    """

    def __init__(
        self,
        block: Union[ResNetBlock, ResNetBottleneck, str],
        layers: list[int],
        block_inplanes: list[int],
        spatial_dims: int = 3,
        n_input_channels: int = 3,
        conv1_t_size: Union[tuple[int], int] = 7,
        conv1_t_stride: Union[tuple[int], int] = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        num_classes: int = 400,
        feed_forward: bool = True,
        bias_downsample: bool = True,  # for backwards compatibility (also see PR #5477)
    ) -> None:
        super().__init__()

        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlock
            elif block == "bottleneck":
                block = ResNetBottleneck
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)

        conv_type: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        norm_type: Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d] = Norm[Norm.BATCH, spatial_dims]

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.bias_downsample = bias_downsample

        conv1_kernel_size = ensure_tuple_rep(conv1_t_size, spatial_dims)
        conv1_stride = ensure_tuple_rep(conv1_t_stride, spatial_dims)

        self.conv1 = conv_type(
            n_input_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,  # type: ignore
            stride=conv1_stride,  # type: ignore
            padding=tuple(k // 2 for k in conv1_kernel_size),  # type: ignore
            bias=False,
        )
        self.bn1 = norm_type(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], spatial_dims, shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], spatial_dims, shortcut_type)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], spatial_dims, shortcut_type)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], spatial_dims, shortcut_type)
        self.out = conv_type(block_inplanes[3] * block.expansion, num_classes, kernel_size=1, stride=1, bias=True)

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int, spatial_dims: int = 3) -> torch.Tensor:
        out: torch.Tensor = get_pool_layer(("avg", {"kernel_size": 1, "stride": stride}), spatial_dims=spatial_dims)(x)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(
        self,
        block: Union[ResNetBlock, ResNetBottleneck],
        planes: int,
        blocks: int,
        spatial_dims: int,
        shortcut_type: str,
        stride: int = 1,
    ) -> nn.Sequential:
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        downsample: Union[nn.Module, partial, None] = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if look_up_option(shortcut_type, {"A", "B"}) == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    spatial_dims=spatial_dims,
                )
            else:
                downsample = nn.Sequential(
                    conv_type(
                        self.in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=self.bias_downsample,
                    ),
                    norm_type(planes * block.expansion),
                )

        layers = [
            block(
                in_planes=self.in_planes, planes=planes, spatial_dims=spatial_dims, stride=stride, downsample=downsample
            )
        ]

        self.in_planes = planes * block.expansion
        for _i in range(1, blocks):
            layers.append(block(self.in_planes, planes, spatial_dims=spatial_dims))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.out(x)
        x = self.relu(x)

        return x


"""
    implementation of forming Loop subdivision method as message passing neural network. this takes Pytorch3d.Mesh object as input and output. 
    
    subdvided faces will have the same orientation as the original fases, i.e., if the original faces are counter-clockwise then the subdivided faces are alse counter-clockewise. presume that the input mesh is homogeneous, i.e., all faces are triangles.
    
    a walkthrough of the method follows,
    1. create a faces indices for the subdivided mesh that is pre-computed and can be used for multiple meshes that has the same topology.
    2. create message passing using torch-geometric.MessagePassing base class with 'mean' aggregate method.
    3. create new vertices using the passage method and concatenate them to the original vertices.
    4. output the new mesh with the same topology as the original mesh.
"""
from typing import List
import torch
from torch import Tensor
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
)
import torch.nn as nn
from torch.nn import Sequential, ModuleList, LeakyReLU
from pytorch3d.structures import Meshes
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import scatter, to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes


# function for pre-computed faces index
@ torch.no_grad()
class Subdivision():
    def __init__(self, 
                 mesh: Meshes, num_layers: int,
                 allow_subdiv_faces: List[torch.LongTensor]=[None, None]
                 ) -> list:
        
        self.faces_levels = []
        for l in range(num_layers):
            new_faces = self.subdivide_faces_fn(mesh, allow_subdiv_faces[l])
            self.faces_levels.append(new_faces)
            verts = mesh.verts_packed()
            edges = mesh.edges_packed()
            new_verts = verts[edges].mean(dim=1)
            new_verts = torch.cat([verts, new_verts], dim=0)
            mesh = Meshes(verts=[new_verts], faces=[new_faces])

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
    
# # function for pre-computed faces index
# @ torch.no_grad()
# class Subdivision():
#     def __init__(self, 
#                  mesh: Meshes, num_layers: int,
#                  allow_subdiv_faces: List[torch.LongTensor]=[None, None]
#                  ) -> list:
        
#         self.faces_levels = []
#         for l in range(num_layers):
#             new_faces = self.subdivide_faces_fn(mesh, allow_subdiv_faces[l])
#             self.faces_levels.append(new_faces)

#             verts = mesh.verts_packed()
#             edges = mesh.edges_packed()
#             new_verts = verts[edges].mean(dim=1)
#             face_points = verts[mesh.faces_packed()].mean(dim=1)
#             new_verts = torch.cat([verts, new_verts, face_points], dim=0)
#             mesh = Meshes(verts=[new_verts], faces=[new_faces])

#     def subdivide_faces_fn(self, mesh: Meshes, allow_subdiv_faces: torch.LongTensor=None):
#         verts_packed = mesh.verts_packed()
#         faces_packed = mesh.faces_packed()
#         faces_packed_to_edges_packed = (
#             verts_packed.shape[0] + mesh.faces_packed_to_edges_packed()
#         )
#         faces_idx = torch.arange(faces_packed_to_edges_packed.max() + 1, faces_packed_to_edges_packed.max() + mesh._F + 1,device=faces_packed.device)
#         if allow_subdiv_faces is not None:
#             faces_packed = faces_packed[allow_subdiv_faces]
#             faces_packed_to_edges_packed = faces_packed_to_edges_packed[allow_subdiv_faces]
#             faces_idx = faces_idx[allow_subdiv_faces]

#         f0 = torch.stack([
#             faces_packed[:, 0],                     # 0
#             faces_packed_to_edges_packed[:, 2],     # 3
#             faces_idx,                              # 6
#         ], dim=1)
#         f1 = torch.stack([
#             faces_packed[:, 1],                     # 1
#             faces_idx,                              # 6
#             faces_packed_to_edges_packed[:, 2],     # 3
#         ], dim=1)
#         f2 = torch.stack([
#             faces_packed[:, 1],                     # 1
#             faces_packed_to_edges_packed[:, 0],     # 5
#             faces_idx,                              # 6
#         ], dim=1)
#         f3 = torch.stack([
#             faces_packed[:, 2],                     # 2
#             faces_idx,                              # 6
#             faces_packed_to_edges_packed[:, 0],     # 5
#         ], dim=1)
#         f4 = torch.stack([
#             faces_packed[:, 2],                     # 2
#             faces_packed_to_edges_packed[:, 1],     # 4
#             faces_idx,                              # 6
#         ], dim=1)
#         f5 = torch.stack([
#             faces_packed[:, 0],                     # 0
#             faces_idx,                              # 6
#             faces_packed_to_edges_packed[:, 1],     # 4
#         ], dim=1)

#         subdivided_faces_packed = torch.cat([f0, f1, f2, f3, f4, f5], dim=0)

#         if allow_subdiv_faces is not None:
#             subdivided_faces_packed = torch.cat(
#                 [mesh.faces_packed()[~allow_subdiv_faces], subdivided_faces_packed], dim=0
#             )

#         return subdivided_faces_packed
    
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, flow="source_to_target", dtype=None):

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


class GSNLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_features = kwargs.get('hidden_features', 16)

        self.lin = Sequential(
            Linear(in_channels, self.hidden_features, bias=False,
                   weight_initializer='glorot'),
                   LeakyReLU(inplace=True),
            Linear(self.hidden_features, self.hidden_features, bias=False,
                     weight_initializer='glorot'),
                     LeakyReLU(inplace=True),
            Linear(self.hidden_features, out_channels, bias=False,
                        weight_initializer='glorot')
        )

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for m in self.lin:
            if isinstance(m, Linear):
                m.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight, x.size(self.node_dim),
            self.flow, x.dtype)

        x = self.lin(x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class UpdateLayer(MessagePassing):
    def __init__(self, **kwargs):

        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor=None):
        
        # edge_index, edge_weight = gcn_norm(  # yapf: disable
        #     edge_index, edge_weight, x.size(self.node_dim),
        #     self.flow, x.dtype)
        
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                           size=None)
        
        return out
    
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class GSN(nn.Module):
    def __init__(self, hidden_features: int, num_layers: int = 2):
        super().__init__()

        self.gcn_layers = ModuleList([
            GSNLayer(3, 3, bias=False, hidden_features=hidden_features)
            for _ in range(num_layers)
        ])

        self.update_layer = UpdateLayer()

    def face_to_edge(self, faces: torch.LongTensor, num_nodes: int):
        edge_index = torch.cat([faces[:2], faces[1:], faces[::2]], dim=1)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        return edge_index

    def forward(self, meshes: Meshes, subdivided_faces: list[torch.LongTensor]):
        
        for l, gcn_layer in enumerate(self.gcn_layers):
            # 1. update the vertices with learnt offsets.
            offsets = gcn_layer(
                meshes.verts_packed(), meshes.edges_packed().t().contiguous()
                )
            meshes = meshes.offset_verts(offsets)

            # 2. create new vertices at the middle of the edges.
            new_faces = subdivided_faces[l].expand(meshes._N, -1, -1).to(meshes.device)
            centre_faces = new_faces[:, 3*meshes._F:, :] - meshes._V
            centre_edges = self.face_to_edge(centre_faces.view(-1, 3).t().contiguous(), meshes._V)
            verts = meshes.verts_padded()
            edges = meshes[0].edges_packed()
            new_verts = verts[:, edges].mean(dim=2)
            new_verts = self.update_layer(new_verts, centre_edges)
            new_verts = torch.cat([verts, new_verts], dim=1)

            # 3. create new meshes with the same topology as the original mesh.
            meshes = Meshes(verts=new_verts, faces=new_faces)
        
        return meshes
