"""
    help code to decide the cropping size of differentiable slicer for training with CAP dataset.
"""
import os, json
from glob import glob
import numpy as np

slice_info_path = "/mnt/data/Experiment/Data/CAP/Dataset017_CAP_combined/slice_info/"
slice_info_list = glob(os.path.join(slice_info_path, "*.json"))

data_shape = {
    "sax": [], "2ch": [], "3ch": [], "4ch": []
}
for slice_info in slice_info_list:
    with open(slice_info, "r") as f:
        info_dict = json.load(f)
        for key in data_shape.keys():
            data_shape[key].append(np.asarray(
                info_dict[f"{key.upper()}"]["data_shape"]))
            
# get the median size of the data shape
for key in data_shape.keys():
    data_shape[key] = np.median(np.asarray(data_shape[key]), axis=0)

# print the median size of the data shape
print(data_shape)
    