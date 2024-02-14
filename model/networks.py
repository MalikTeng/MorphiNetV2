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
import numpy as np

from einops import rearrange

from .parts import ConvLayer, ResBlock

__all__ = ["ResNet", "SDFNet", "GSN"]


class SDFNet(nn.Module):
    def __init__(self, 
                 encoder: nn.Module, decoder: nn.Module,
                 ) -> None:
        super().__init__()

        self.encoder_ = encoder
        self.decoder_ = decoder

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, 1e-2, mode="fan_out", nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        _, seg = self.encoder_(x)
        _, sdf = self.decoder_(seg)

        return sdf, seg


class ResNet(nn.Module):
    def __init__(
        self,
        init_filters: int,
        in_channels: int,
        out_channels: int,
        act: str = 'relu',
        norm: str = 'batch',
        num_init_blocks: tuple = (1, 2, 2, 4),
    ):
        super().__init__()
        self.init_filters = init_filters
        self.in_resnet = in_channels
        self.out_resnet = out_channels
        self.act = act
        self.norm = norm

        self.blocks = self._make_blocks(num_init_blocks)
        
    def _make_blocks(self, num_init_blocks: tuple) -> nn.ModuleList:
        down_blocks  = nn.ModuleList()
        in_channels = self.in_resnet
        out_channels = None
        for i in range(len(num_init_blocks)):
            if i == 0:
                out_channels = self.init_filters
            else:
                out_channels = in_channels * 2
            down_blocks.append(nn.Sequential(
                ConvLayer(
                    in_channels, out_channels, 1, 1, 0,
                    self.norm, self.act,
                    ), 
                ResBlock(
                    out_channels, out_channels,
                    self.norm, self.act,
                    num_layers=num_init_blocks[i]
                    )
                ))
            in_channels = out_channels
        # output layer
        down_blocks.append(ConvLayer(
            out_channels, self.out_resnet, 1, 1, 0,
            None, None
        ))
            
        return down_blocks

    def forward(self, x):
        for block in self.blocks[:-1]:
            x, _ = block(x)
        x_seg = self.blocks[-1](x)

        return x, x_seg



"""
    implementation of forming Loop subdivision method as message passing neural network. this takes Pytorch3d.Mesh object as input and output. 
    
    subdvided faces will have the same orientation as the original fases, i.e., if the original faces are counter-clockwise then the subdivided faces are alse counter-clockewise. presume that the input mesh is homogeneous, i.e., all faces are triangles.
    
    a walkthrough of the method follows,
    1. create a faces indices for the subdivided mesh that is pre-computed and can be used for multiple meshes that has the same topology.
    2. create message passing using torch-geometric.MessagePassing base class with 'mean' aggregate method.
    3. create new vertices using the passage method and concatenate them to the original vertices.
    4. output the new mesh with the same topology as the original mesh.
"""
from typing import Any
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Sequential, ModuleList, ReLU, LeakyReLU, Dropout, Linear, BatchNorm1d, LayerNorm
from pytorch3d.structures import Meshes
from torch_geometric.nn import MessagePassing, GCNConv, ChebConv
from torch_geometric.nn import Sequential as SeqGCN
from torch_geometric.utils import add_self_loops, degree


def subdivide_faces_fn(mesh):
    verts_packed = mesh.verts_packed()
    faces_packed = mesh.faces_packed()
    faces_packed_to_edges_packed = (
        mesh.faces_packed_to_edges_packed() + verts_packed.shape[0]
    )
    f0 = torch.stack(
        [
            faces_packed[:, 0],
            faces_packed_to_edges_packed[:, 2],
            faces_packed_to_edges_packed[:, 1],
        ],
        dim=1,
    )
    f1 = torch.stack(
        [
            faces_packed[:, 1],
            faces_packed_to_edges_packed[:, 0],
            faces_packed_to_edges_packed[:, 2],
        ],
        dim=1,
    )
    f2 = torch.stack(
        [
            faces_packed[:, 2],
            faces_packed_to_edges_packed[:, 1],
            faces_packed_to_edges_packed[:, 0],
        ],
        dim=1,
    )
    f3 = faces_packed_to_edges_packed
    subdivided_faces_packed = torch.cat([f0, f1, f2, f3], dim=0)

    return subdivided_faces_packed.requires_grad_(False)
    

class PositionNet(MessagePassing):
    def __init__(self, hidden_features: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.fc_ = Sequential(
            Linear(3, hidden_features, bias=False),
            LayerNorm(hidden_features),
            LeakyReLU(inplace=True),
            Linear(hidden_features, hidden_features // 2, bias=False),
            LayerNorm(hidden_features // 2),
            LeakyReLU(inplace=True),
        )

        self.out_ = Linear(hidden_features // 2, 3, bias=True)

        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        # add self loops to the graph.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # apply linear transformation to the input features.
        x = self.fc_(x)

        # calculate the norm
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # propagate the messages.
        x = self.propagate(edge_index, x=x, norm=norm)

        # apply linear transformation to the output features.
        x = self.out_(x)

        return x
    
    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j


class SubdivideMeshes(nn.Module):
    def __init__(self, subdivided_faces=None, hidden_features=16) -> None:
        super().__init__()

        self._N = -1
        if subdivided_faces is not None:
            self.register_buffer("_subdivided_faces", subdivided_faces)
        else:
            raise ValueError("either meshes or subdivided_faces must be provided")

        # TODO: develop own GCN layer with learable weights.
        # self.gcn_layer_ = PositionNet(hidden_features=hidden_features)
        self.gcn_layer_ = SeqGCN(
            "x, edge_index",[
                (GCNConv(3, hidden_features), "x, edge_index -> x"),
                LeakyReLU(inplace=True),
                (GCNConv(hidden_features, hidden_features), "x, edge_index -> x"),
                LeakyReLU(inplace=True),
                Dropout(0.5),
                (GCNConv(hidden_features, 3), "x, edge_index -> x")
                ])

        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, 1e-2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, meshes):
        # 1. update the vertices with learnt offsets.
        offsets = self.gcn_layer_(meshes.verts_packed(), 
                                  meshes.edges_packed().t().contiguous())
        meshes = meshes.offset_verts(offsets)

        # 2. create new vertices at the middle of the edges.
        verts = meshes.verts_padded()
        edges = meshes[0].edges_packed()
        new_verts = verts[:, edges].mean(dim=2)
        new_verts = torch.cat([verts, new_verts], dim=1)

        # 3. create new meshes with the same topology as the original mesh.
        new_faces = self._subdivided_faces.expand(meshes._N, -1, -1)
        new_meshes = Meshes(verts=new_verts, faces=new_faces)

        return new_meshes

        # verts = meshes.verts_padded()

        # # create new vertices at the middle of the edges.
        # edges = meshes[0].edges_packed()
        # new_verts = verts[:, edges].mean(dim=2)
        # new_verts = torch.cat([verts, new_verts], dim=1)

        # # create new meshes with the same topology as the original mesh.
        # new_faces = self._subdivided_faces.expand(meshes._N, -1, -1)
        # new_meshes = Meshes(verts=new_verts, faces=new_faces)

        # # update the all vertices with the offsets.
        # new_edges = new_meshes.edges_packed()
        # new_verts = new_meshes.verts_packed()
        # offsets = self.gcn_layer_(new_verts, new_edges.t().contiguous())

        # new_meshes = new_meshes.offset_verts(offsets)

        # return new_meshes
    

class GSN(nn.Module):
    def __init__(self, meshes: Meshes, hidden_features: int, num_layers: int = 2):
        super().__init__()
        assert len(meshes) == 1, "requires only one initial mesh"

        # pre-computed faces index
        self.faces_layers_ = []
        for _ in range(num_layers):
            verts = meshes.verts_packed()
            edges = meshes.edges_packed()
            new_faces = subdivide_faces_fn(meshes)
            self.faces_layers_.append(new_faces)
            new_verts = verts[edges].mean(dim=1)
            new_verts = torch.cat([verts, new_verts], dim=0)
            meshes = Meshes(verts=[new_verts], faces=[new_faces])

        self.gcn_layers_ = ModuleList([
            SubdivideMeshes(subdivided_faces=self.faces_layers_[i], 
                            hidden_features=hidden_features) 
            for i in range(num_layers)
            ])

    def forward(self, meshes: Meshes):
        for gsn_ in self.gcn_layers_:
            meshes = gsn_(meshes)

        return meshes