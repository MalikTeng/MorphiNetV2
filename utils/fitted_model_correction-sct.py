"""
Extraction and registration of meshes from GPfiles.

Extraction part:
    Source files: control_meshes are txt files include coordinates of surface meshes' points and class label for each point written in lines.
    Target files: OBJ files are mesh models converted form control_meshes files.
    working directory:
        source files: '/mnt/data/Experiment/Data/original_data/ScotHeart/surface_point/*control_mesh.txt'
        target files: '/mnt/data/Experiment/Data/original_data/ScotHeart/subdivided_mesh/patient_id.obj'
    Working flow:
        1. load control_mesh by lines and get coordinates of points and class label for each node.
        2. apply surface subdivision and convert atlas into mesh (trimesh).
        3. save mesh into OBJ files.

Registration part:
    Source files: OBJ files are mesh models converted from control_meshes files.
    Target files: OBJ file is a 3D atlas model of myocardium.
    working directory:
        source files: '/mnt/data/Experiment/Data/original_data/ScotHeart/subdivided_mesh/patient_id.obj'
        reference file: '/home/yd21/Documents/Nasreddin/template/cap/init_mesh_v2-cap_myo.obj'
        target files: '/mnt/data/Experiment/Data/original_data/ScotHeart/registered_mesh/patient_id.obj'
    Working flow:
        1. load source and reference files.
        2. performing registration (using procrustes & icp in trimesh).
        3. visualize the result and examing the registration quality.
        4. save the registered mesh into OBJ files.

Possible problems:
    the registered mesh may not align with its paired segmentation (in 3D Sliceer), because of the different coordinate system.
"""
import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import trimesh
from skimage.measure import marching_cubes 
from sklearn.cluster import KMeans
from scipy.ndimage import zoom
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import plotly.io as pio
import plotly.graph_objects as go

# subdivision matrix
subdivision_matrix = np.asarray(
    pd.read_table(
        "./template/subdivision_matrix.txt",
        header=None, delim_whitespace=True, engine='c'
    ),
    dtype=np.float32
)
points_label = np.array(
    [[0, 1499], [1500, 2164], [2165, 3223], [3224, 5581],
    [5582, 5630], [5631, 5655], [5656, 5696], [5697, 5729],
    [5730, 5809]]
)
"""
    LV_ENDOCARDIAL = 0 
    RV_SEPTUM = 1
    RV_FREEWALL = 2
    EPICARDIAL =3
    MITRAL_VALVE = 4
    AORTA_VALVE = 5
    TRICUSPID_VALVE = 6
    PULMONARY_VALVE = 7
    RV_INSERT = 8
"""
colors = [tuple(int(255 * i) for i in mcl.to_rgb(v)) for v in mcl.TABLEAU_COLORS.values()]
points_label = np.concatenate(
    [
        np.array(
            [color], dtype=np.float32
        ).repeat(end - start + 1, axis=0)
        for (start, end), color in zip(points_label, colors[:9])
    ],
    axis=0,
    dtype=object
)

# face index
faces_index = np.asarray(
    pd.read_table(
        "./template/ETIndicesSorted.txt",
        header=None, delim_whitespace=True, engine='c'
    ),
    dtype=np.int32
)
faces_index = faces_index - 1
# exclude faces on valves
exclude_points = np.zeros(5810, dtype=np.int32)
exclude_points[5582:5730] = 1
exclude_index = exclude_points[faces_index].sum(axis=1) == 3

# load GPfiles by lines and get coordinates of points and class label for each node.
def load_GPfile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [[float(i) for i in line.strip('\n').split(' ')[:3]] for line in lines]
    points = np.asarray(lines, dtype=np.float32)
    return points

# load reference/source files.
def load_mesh(file_path, flip=False):
    mesh = trimesh.load(file_path)
    if flip:
        mesh.vertices[:, 0] = -mesh.vertices[:, 0]
    return mesh

# apply surface subdivision and convert atlas into mesh (trimesh).
def node2mesh(points):
    # apply surface subdivision
    points = np.matmul(subdivision_matrix, points)
    # convert to trimesh
    mesh = trimesh.Trimesh(vertices=points, faces=faces_index, vertex_colors=points_label)
    # exclude faces on valves
    mesh = mesh.submesh([~exclude_index])[0]
    return mesh

# save mesh into OBJ files.
def save_mesh(mesh, file_path):
    mesh.export(file_path)

# find the anchor points
def get_anchor_points(mesh):
    # find all the surface nodes of the mesh
    lv = mesh.vertices[:1500]
    rv = mesh.vertices[1500:3224]
    mitral_valve = mesh.vertices[5582:5656]
    # pulmonary_valve = mesh.vertices[5697:5730]
    # find the centroid of mitral/aorta and left ventricle apex (the furthest node on lv to the centroid of mitral/aorta)
    mitral_valve_c = np.mean(mitral_valve, axis=0)
    lv_apex = lv[np.argmax(np.linalg.norm(lv - mitral_valve_c, axis=1))]
    # find the centroid of the left and right ventricle
    lv_c = np.mean(lv, axis=0)
    rv_c = np.mean(rv, axis=0)
    # return the anchor points
    return np.array([mitral_valve_c, lv_apex, lv_c, rv_c])

# performing registration and save the result into target files.
def registration(source, reference, target):
    # find the anchor points
    anchors_source = get_anchor_points(source)
    anchors_reference = get_anchor_points(reference)

    # first registration, find the registration matrix using procrustes method
    matrix = trimesh.registration.procrustes(
        anchors_source, anchors_reference, 
        weights=None,
        reflection=False, translation=True, scale=True
        )
    # second registration, find the registration matrix using icp method
    matrix = trimesh.registration.icp(
        source.vertices, reference.vertices,
        initial=matrix[0],
        weights=None,
        scale=True,
        )
    # apply the registration matrix to source files
    source.apply_transform(matrix[0])
    
    # apply the registration matrix to landmarks
    anchors_source = np.matmul(matrix[0], 
                            np.concatenate([anchors_source, np.ones((4, 1))], axis=1).T).T[:, :3]

    # # visualize the result if the loss is too large
    # vis_seg_mesh(source, reference, anchors_source, anchors_reference)
    
    # save the result into target files
    save_mesh(source, target)


# visualize source mesh and referecence segmentation for a check
def vis_seg_mesh(source, reference, anchors_source, anchors_reference):
    # plot the mesh derived from segmentation
    fig = go.Figure(
        data=go.Mesh3d(
            x=reference.vertices[:,0], y=reference.vertices[:,1], z=reference.vertices[:,2],
            i=reference.faces[:,0], j=reference.faces[:,1], k=reference.faces[:,2],
            opacity=1.0,
            color='lightpink'
        )
    )
    # plot the landmarks from reference
    for landmark, name in zip(anchors_reference, ['mitral_valve, lv_apex, lv_c, rv_c']):
        fig.add_trace(
            go.Scatter3d(
                x=[landmark[0]], y=[landmark[1]], z=[landmark[2]],
                mode='markers',
                marker=dict(
                    size=5,
                    color='red',
                    opacity=0.8
                ),
                name=f"seg-{name}"
            )
        )

    # plot the mesh
    fig.add_trace(go.Mesh3d(
        x=source.vertices[:, 0], y=source.vertices[:, 1], z=source.vertices[:, 2],
        i=source.faces[:, 0], j=source.faces[:, 1], k=source.faces[:, 2],
        color='lightblue',
        opacity=0.5,
        name='mesh'
    ))
    # plot the landmarks
    for landmark, name in zip(anchors_source, ['mitral_valve, lv_apex, lv_c, rv_c']):
        fig.add_trace(
            go.Scatter3d(
                x=[landmark[0]], y=[landmark[1]], z=[landmark[2]],
                mode='markers',
                marker=dict(
                    size=5,
                    color='blue',
                    opacity=0.8
                ),
                name=f"mesh-{name}"
            )
        )
    # set the layout to show the image in the center of the plot
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='x',
                backgroundcolor='rgb(200, 200, 230)',
                gridcolor='rgb(255, 255, 255)',
                showbackground=True,
                zerolinecolor='rgb(255, 255, 255)',
            ),
            yaxis=dict(
                title='y',
                backgroundcolor='rgb(230, 200,230)',
                gridcolor='rgb(255, 255, 255)',
                showbackground=True,
                zerolinecolor='rgb(255, 255, 255)',
            ),
            zaxis=dict(
                title='z',
                backgroundcolor='rgb(230, 230,200)',
                gridcolor='rgb(255, 255, 255)',
                showbackground=True,
                zerolinecolor='rgb(255, 255, 255)',
            ),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=0, y=0, z=2)
            ),
            dragmode='turntable',
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
        ),
        margin=dict(r=10, l=10, b=10, t=10)
    )
    # identify the mesh and segmentation
    fig.update_traces(
        selector=dict(type='mesh3d'),
        showlegend=True,
        legendgroup='mesh'
    )
    fig.update_traces(
        selector=dict(type='scatter3d'),
        showlegend=True,
        legendgroup='segmentation'
    )
    # save the figure
    pio.write_html(fig, f"./{os.path.basename(patient_id)}_registered.html")


if __name__ == '__main__':
    # # extraction part
    # # source directory
    # source_dir = '/mnt/data/Experiment/Data/original_data/ScotHeart/surface_point'
    # # target directory
    # target_dir = '/mnt/data/Experiment/Data/original_data/ScotHeart/subdivided_mesh'
    # # get patient ids
    # surface_point_files = os.listdir(source_dir)
    # # extract meshes from surface points and predefined face index
    # print('extract meshes from surface points and predefined face index')
    # for surface_point_file in tqdm(surface_point_files):
    #     patient_id = "_".join(surface_point_file.split('_')[:2])
    #     # load GPfiles
    #     points = load_GPfile(
    #         os.path.join(source_dir, surface_point_file)
    #         )
    #     # convert vertices and labels into mesh (trimesh)
    #     mesh = node2mesh(points)
    #     # save mesh into OBJ files
    #     save_mesh(mesh, os.path.join(target_dir, patient_id + '.obj'))
    
    # registration part
    # source directory
    source_dir = '/mnt/data/Experiment/Data/original_data/ScotHeart/subdivided_mesh'
    # reference
    reference_dir = '/home/yd21/Documents/Nasreddin/template/cap/init_mesh_v2-cap_myo.obj'
    # target directory
    target_dir = '/mnt/data/Experiment/Data/original_data/ScotHeart/registered_mesh'
    # get patient ids
    obj_files = os.listdir(source_dir)
    # register meshes to reference
    print('register meshes to reference')
    for obj_file in tqdm(obj_files):
        patient_id = obj_file.strip(".obj")
        # load reference/source files
        reference = load_mesh(reference_dir)
        source = load_mesh(os.path.join(source_dir, obj_file), flip=True)
        # performing registration (using mesh_other or icp in trimesh) and save the result into target files
        source = registration(source, reference, os.path.join(target_dir, obj_file))
