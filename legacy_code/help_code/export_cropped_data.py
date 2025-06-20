"""
    generate cropped image/label data for the DeformNet training
"""
import os, sys
pwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pwd)  # Add parent directory to Python path
import tqdm
import json
import numpy as np
np.random.seed(42)
from trimesh.voxel.ops import matrix_to_marching_cubes
from trimesh.smoothing import filter_laplacian, filter_taubin
from monai.transforms import (
    Compose,
    Spacingd,
    CropForegroundd,
    Resized,
    SpatialPadd,
    EnsureTyped
)
from monai.data.utils import decollate_batch
import nibabel as nib

from data.transform import pre_transform


# root directory for the NDF data
export_dir = "/mnt/data/Experiment/Data/DeformNet_Data/"
split = {"train": "train_fold0", "val": "validation_fold0", "test": "test"}

for phase in ["train", "val", "test"]:

    # dataset_name = "CAP"
    # keys = ["mr_image", "mr_label"]
    # transform = pre_transform(keys, "mr", "valid", False, [128, 128, 128], [4, 4, 4], 2)
    # dataset_root = "/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset011_CAP_SAX"
    # label_repo = json.load(open("dataset/dataset_task11_f0.json", 'r'))[split[phase]]

    # dataset_name = "ACDC"
    # keys = ["mr_image", "mr_label"]
    # transform = pre_transform(keys, "mr", "valid", False, [128, 128, 128], [4, 4, 4], 2, target="acdc")
    # dataset_root = "/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset021_ACDC/"
    # label_repo = json.load(open("dataset/dataset_task21_f0.json", 'r'))[split[phase]]

    # dataset_name = "SCOTHEART"
    # keys = ["ct_image", "ct_label"]
    # transform = pre_transform(keys, "ct", "valid", False, [128, 128, 128], [4, 4, 4], 2)
    # dataset_root = "/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset020_SCOTHEART"
    # label_repo = json.load(open("dataset/dataset_task20_f0.json", 'r'))[split[phase]]

    # dataset_name = "MMWHS"
    # keys = ["ct_image", "ct_label"]
    # transform = pre_transform(keys, "ct", "valid", False, [128, 128, 128], [4, 4, 4], 2)
    # dataset_root = "/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset022_MMWHS_CT/"
    # label_repo = json.load(open("dataset/dataset_task22_f0.json", 'r'))[split[phase]]

    # data augmentation for resizing the segmentation prediction into crop window size
    post_transform = Compose([
        Spacingd(keys, [2.0, 2.0, 2.0], mode=("bilinear", "nearest")),
        CropForegroundd(keys, source_key=keys[1]),
        Resized(keys, 128, size_mode="longest", mode=("bilinear", "nearest-exact")),
        SpatialPadd(keys, 128, method="symmetric", mode="minimum"),
        EnsureTyped(keys),
    ])

    for label in tqdm.tqdm(label_repo):
        voxel = transform({
            keys[0]: os.path.join(dataset_root, label["image"]),
            keys[1]: os.path.join(dataset_root, label["label"])
        })
        
        voxel = post_transform(voxel)
        
        for key in keys:
            print(f"Exporting {key} data...")
            
            for i in range(voxel[key].shape[0]):
                voxel_ = voxel[key][i] == 2 if "label" in key else voxel[key][i]    # select the endocardium label

                # save the voxel_ as nii.gz file
                modality, category = key.split("_")
                export_to = os.path.join(export_dir, dataset_name, "nii", 
                                         modality + '_' + phase + ('_seg' if "label" in key else ''))
                os.makedirs(export_to, exist_ok=True)
                nib.save(
                    nib.Nifti1Image(voxel_.numpy().astype(np.uint8 if "label" in key else np.float32), voxel[key].affine), 
                    os.path.join(export_to, os.path.basename(label[category]).replace(
                        ".nii.gz" if "label" in key else "_0000.nii.gz", f"-{i:02d}.nii.gz").replace(
                            ".seg.nrrd" if "label" in key else "_0000.seq.nrrd", f"-{i:02d}.nii.gz"
                        ))
                    )
