# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from sklearn.model_selection import KFold


def create_dataset_json(
    dataset_input_dir: str, 
    task_name: str, 
    task_description: str, 
    labels: str, 
    modality: str,
    image_file_extension: str,
    label_file_extension: str
    ) -> list:
    data_json_new = dict()
    
    data_json_new["name"] = task_name
    data_json_new["description"] = task_description
    data_json_new["tensorImageSize"] = "3D"
    data_json_new["reference"] = "https://cardiacatlas.org/challenges/segmentation/"
    data_json_new["licence"] = "CC-BY-SA 4.0"
    data_json_new["release"] = "n/a"
    data_json_new["modality"] = {
        "0": modality
    }
    data_json_new["labels"] = labels

    train_list = set([case.replace(label_file_extension, '') for case in os.listdir(dataset_input_dir + "/labelsTr")])   # seperate train and valid set on a patient-bases
    train_list = np.array(list(train_list))
    data_json_new["training"] = [
        {
            "image": f"./imagesTr/{case}{image_file_extension}",
            "label": f"./labelsTr/{case}{label_file_extension}"
        }
        for case in train_list
    ]
    data_json_new["numTraining"] = len(data_json_new["training"])

    # if os.path.exists(dataset_input_dir + "/imagesTs"):
    #     test_list = set([case.replace(image_file_extension, '') for case in os.listdir(dataset_input_dir + "/imagesTs")])
    #     test_list = np.array(list(test_list))
    #     data_json_new["test"] = [
    #         {
    #             "image": f"./imagesTs/{case}{image_file_extension}",
    #         }
    #         for case in test_list
    #     ]
    #     data_json_new["numTest"] = len(data_json_new["test"])
    # else:
    #     data_json_new["test"] = []
    #     data_json_new["numTest"] = 0
    
    if 'cap' in task_name.lower():
        file_dir = "/mnt/data/Experiment/MICCAI_24/cap/MorphiNet/lv/f0"
    elif 'scot' in task_name.lower():
        file_dir = "/mnt/data/Experiment/MICCAI_24/sct/MorphiNet/lv/f0"
    elif 'mmwhs' in task_name.lower():
        file_dir = "/mnt/data/Experiment/MICCAI_24/mmwhs/MorphiNet/lv/f0"
    else:
        raise ValueError("Task name not recognized!")
    
    # copy files from image and label Tr folders to image and label Ts folders
    if not os.path.exists(os.path.join(dataset_input_dir, "labelsTs")):
        os.makedirs(os.path.join(dataset_input_dir, "imagesTs"))
        os.makedirs(os.path.join(dataset_input_dir, "labelsTs"))
    for case in os.listdir(file_dir):
        os.system(f"cp {dataset_input_dir}/imagesTr/{case.split('-')[0]}* {dataset_input_dir}/imagesTs/")
        os.system(f"cp {dataset_input_dir}/labelsTr/{case.split('-')[0]}* {dataset_input_dir}/labelsTs/")

    data_json_new["test"] = [{
        "image": f"./imagesTs/{case.replace(label_file_extension, '')}{image_file_extension}",
        "label": f"./labelsTs/{case.replace(label_file_extension, '')}{label_file_extension}"
    } for case in os.listdir(f"{dataset_input_dir}/labelsTs/")]
    data_json_new["numTest"] = len(data_json_new["test"])
    
    return data_json_new


def create_datalist(args):

    if args.file_extension == ".nrrd":
        image_file_extension = "_0000.seq.nrrd"
        label_file_extension = ".seg.nrrd"
    elif args.file_extension == ".nii.gz":
        image_file_extension = "_0000.nii.gz"
        label_file_extension = ".nii.gz"

    dataset_new = create_dataset_json(
        args.input_dir + args.task_name, 
        args.task_name, args.description, args.labels,
        args.modality, image_file_extension, label_file_extension
        )

    dataset_with_folds = dataset_new.copy()
    keys = [os.path.basename(line["label"]).strip(label_file_extension) for line in dataset_with_folds["training"]]
    dataset_train_dict = dict(zip(keys, dataset_with_folds["training"]))
    all_keys_sorted = np.sort(keys)
    kfold = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    for j, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
        val_data = []
        train_data = []
        train_keys = np.array(all_keys_sorted)[train_idx]
        test_keys = np.array(all_keys_sorted)[test_idx]
        for key in test_keys:
            val_data.append(dataset_train_dict[key])
        for key in train_keys:
            train_data.append(dataset_train_dict[key])

        dataset_with_folds[f"validation_fold{j}"] = val_data
        dataset_with_folds[f"train_fold{j}"] = train_data

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, f"dataset_task{args.task_name.split('_')[0][-2:]}_f0.json"), "w") as f:
        json.dump(dataset_with_folds, f)
        print(f"data list fold_0 for {args.task_name} has been created!")
        f.close()


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    # change these with info for the task you want to update
    parser.add_argument("-input_dir", "--input_dir", type=str, 
                        default="/mnt/data/Experiment/Data/MorphiNet-MR_CT/")
    parser.add_argument("-file_extension", "--file_extension", type=str, 
                        default=".nrrd", help="the file extension of the data (.nii.gz / .nrrd)")
    parser.add_argument("-task_name", "--task_name", type=str, 
                        default="Dataset011_CAP_SAX", help="the task name")
    parser.add_argument("-d", "--description", help="the task description",
                        default="CAP SAX MR image data w/ cross-validation")
    parser.add_argument("-l", "--labels", type=json.loads, help="the label name",
                        default='{"0": "background", "1": "lv", "2": "lv-myo", "3": "rv", "4": "rv-myo"}')
    parser.add_argument("-m", "--modality", type=str, 
                        default="MR", help="the modality name")
    
    parser.add_argument("-output_dir", "--output_dir", type=str, 
                        default="dataset/")
    parser.add_argument("-num_folds", "--num_folds", type=int, 
                        default=5, help="number of folds")
    parser.add_argument("-seed", "--seed", type=int, 
                        default=42, help="seed number")

    args = parser.parse_args()

    create_datalist(args)
