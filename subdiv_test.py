from typing import List
import torch
from torch_geometric.utils import to_undirected
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
import plotly.io as pio
import plotly.graph_objects as go


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def face_to_edge(faces: torch.LongTensor, num_nodes: int):
    edge_index = torch.cat([faces[:2], faces[1:], faces[::2]], dim=1)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    return edge_index

# example usage
mesh = ico_sphere(2).to(DEVICE)

# create pre-computed subdivision matrix
faces_levels = Subdivision(mesh, 1).faces_levels

# create new vertices at the middle of the edges.
new_faces = faces_levels[0].expand(mesh._N, -1, -1).to(mesh.device)
verts = mesh.verts_padded()
edges = mesh.edges_packed()
edge_verts = verts[:, edges].mean(dim=2)
new_verts = torch.cat([verts, edge_verts], dim=1)

# centre_faces = new_faces[:, 3*mesh._F:, :] - mesh._V
# centre_edges = face_to_edge(centre_faces.view(-1, 3).t().contiguous(), mesh._V)

mesh_ = Meshes(verts=new_verts, faces=new_faces)

# visualise the new_verts in a contrasting color than verts using plotly
verts_ = mesh_.verts_packed().cpu().numpy()
faces_ = mesh_.faces_packed().cpu().numpy()
fig = go.Figure()
# add mesh
fig.add_trace(
    go.Mesh3d(
        x=verts_[:, 0],
        y=verts_[:, 1],
        z=verts_[:, 2],
        i=faces_[:, 0],
        j=faces_[:, 1],
        k=faces_[:, 2],
        opacity=0.8,
        color='lightpink',
    )
)

# add the wireframe of original edges
verts = mesh.verts_packed().cpu().numpy()
edges = mesh.edges_packed().cpu().numpy()
tris = verts[edges]
xe, ye, ze = [], [], []
for t in tris:
    xe.extend([t[k%2][0] for k in range(3)] + [None])
    ye.extend([t[k%2][1] for k in range(3)] + [None])
    ze.extend([t[k%2][2] for k in range(3)] + [None])
fig.add_trace(
    go.Scatter3d(
        x=xe,
        y=ye,
        z=ze,
        mode='lines',
        line=dict(color='white', width=1.5),
        hoverinfo='none',
        showlegend=False
    )
)

# add the wireframe of new faces
centre_faces = new_faces[:, 3*mesh._F:, :].cpu().numpy()
tris = verts_[centre_faces[0]]
xe, ye, ze = [], [], []
for t in tris:
    xe.extend([t[k%3][0] for k in range(4)] + [None])
    ye.extend([t[k%3][1] for k in range(4)] + [None])
    ze.extend([t[k%3][2] for k in range(4)] + [None])
fig.add_trace(
    go.Scatter3d(
        x=xe,
        y=ye,
        z=ze,
        mode='lines',
        line=dict(color='black', width=1.0),
        hoverinfo='none',
        showlegend=False
    )
)

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    )
)
fig.write_html("subdiv_test.html", auto_open=True)

