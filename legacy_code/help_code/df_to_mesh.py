"""
    generate surface mesh for NDF data preparation
"""
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.join(os.path.dirname(__file__), "../data")))
import tqdm
import json
import numpy as np
np.random.seed(42)
from trimesh.voxel.ops import matrix_to_marching_cubes
from trimesh.smoothing import filter_laplacian, filter_taubin
from monai.transforms import (
    Compose,
    Spacing,
    CropForeground,
    Resize,
    SpatialPad,
    EnsureType
)

from data import pre_transform


# root directory for the NDF data
part = 'lv'
export_dir = f"/mnt/data/Experiment/Data/NDF_data-{part}/"
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
split = {"train": "training", "test": "test"}

for phase in ["train", "test"]:
    # data augmentation for resizing the segmentation prediction into crop window size
    post_transform = [
        Spacing([2.0, 2.0, 2.0],  mode="nearest"),
        CropForeground(),
        Resize(128, size_mode="longest", mode="nearest-exact"),
        SpatialPad(128, method="symmetric", mode="constant"),
        EnsureType(),
    ]
    post_transform = Compose(post_transform)

    # dataset_name = "SCOTHEART"
    # keys = ["ct_image", "ct_label"]
    # transform = pre_transform(keys, "ct", "valid", False, [128, 128, 128], [4, 4, 4], 2)
    # dataset_root = "/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset020_SCOTHEART"
    # label_repo = json.load(open("dataset/dataset_task20_f0.json", 'r'))[split[phase]]

    # dataset_name = "MMWHS"
    # keys = ["ct_image", "ct_label"]
    # transform = pre_transform(keys, "ct", "valid", False, [128, 128, 128], [4, 4, 4], 2)
    # dataset_root = "/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset022_MMWHS_CT"
    # label_repo = json.load(open("dataset/dataset_task22_f0.json", 'r'))[split[phase]]

    # dataset_name = "CAP"
    # keys = ["mr_image", "mr_label"]
    # transform = pre_transform(keys, "mr", "valid", False, [128, 128, 128], [4, 4, 4], 2)
    # dataset_root = "/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX"
    # label_repo = json.load(open("dataset/dataset_task11_f0.json", 'r'))[split[phase]]

    dataset_name = "ACDC"
    keys = ["mr_image", "mr_label"]
    transform = pre_transform(keys, "mr", "valid", False, [128, 128, 128], [4, 4, 4], 2, target="acdc")
    dataset_root = "/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset021_ACDC"
    label_repo = json.load(open("dataset/dataset_task21_f0.json", 'r'))[split[phase]]

    # Create the dictionary for the JSON file
    if dataset_name == "CAP":
        file_lists = [os.path.basename(i["label"]).split('-')[0] + f"-{j}" 
                      for i in label_repo for j in ["ED", "ES"]]
    elif dataset_name == "SCOTHEART":
        file_lists = [os.path.basename(i["label"]).split('-')[0] + "-ED"
                      for i in label_repo]
    else:
        file_lists = [os.path.basename(i["label"]).split('.')[0] + "-ED" for i in label_repo]

    json_dict = {
        dataset_name: {
            part: file_lists
             },
        }

    # Save the JSON file with the phase as the filename
    json_filename = f"{dataset_name}_{phase}.json"
    json_filepath = os.path.join(export_dir, json_filename)
    with open(json_filepath, 'w') as json_file:
        json.dump(json_dict, json_file)

    for label in tqdm.tqdm(label_repo):
        voxel = transform({
            keys[0]: os.path.join(dataset_root, label["image"]),
            keys[1]: os.path.join(dataset_root, label["label"])
        })[keys[1]]
        for i in range(voxel.shape[0]):
            voxel_ = post_transform(voxel[i].unsqueeze(0))
            voxel_ = voxel_[0] == 2   # select the myocardium label

            # Convert the label volume to a trimesh object
            mesh = matrix_to_marching_cubes(voxel_.cpu().numpy())
            mesh.vertices = mesh.vertices[:, [1, 0, 2]] # i, j, k -> x, y, z

            # Smooth the label mesh using the Taubin smoothing algorithm
            mesh = filter_taubin(mesh, 0.77, -0.34, 30)

            # Save the mesh to the destination directory
            if dataset_name == "CAP":
                mesh_filename = os.path.basename(label["label"]).split('-')[0] + "-{}.obj".format(["ED", "ES"][i])
            elif dataset_name == "SCOTHEART":
                mesh_filename = os.path.basename(label["label"]).split('.')[0] + ".obj"
            else:
                mesh_filename = os.path.basename(label["label"]).split('.')[0] + "-ED.obj"
            destination_dir = os.path.join(export_dir, dataset_name, part)
            os.makedirs(destination_dir, exist_ok=True)
            mesh_filepath = os.path.join(destination_dir, mesh_filename)
            mesh.export(mesh_filepath)
