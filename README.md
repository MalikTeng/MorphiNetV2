# Introduction
This repository contains scripts and tools that run code for the paper titled ‘[Adaptive Bi-ventricle Surface Reconstruction from Cardiovascular Imaging]()’. The sole purpose of this repository is to provide a reference for the paper. Please note that the code is not optimized for efficiency or guaranteed to be bug-free on arbitrary machine.
 
![Alt text](figure/Fig-1.png)
<!-- 
To gain a better understanding of the idea behind the network design and parameter settings, we invite you to watch the following brief presentation using your MICCAI 2024 access.

[MICCAI Virtual Presentation](https://miccai2024.conflux.events/app/schedule/session/3294/2693) -->

## Connect and Contact with the Author
[LinkedIn](https://www.linkedin.com/in/malik-teng-86085149/)

# Installation
The code is tested on Ubuntu 20.04.6 LTS, Python 3.9.17, and PyTorch 1.12.1. To install the code, first clone the repository:

    $ git clone https://github.com/MalikTeng/MorphiNet
    
    Then install the conda environment:
    $ conda env create -f environment.yml
    
    Then activate the environment:
    $ conda activate morphinet

# Usage
## Data
The data used in the paper is not included in this repository. The data is available upon request, and details provided in the paper. The data is organized in the following structure:
```
data
├── imagesTr
│   ├── 1.nii.gz
│   ├── 2.nii.gz
│   └── ...
├── labelsTr
│   ├── 1.nii.gz
│   ├── 2.nii.gz
│   └── ...
```
Template meshes were needed for runing the code and are organized in the following structure. 

_The `template_mesh` is a triangular surface mesh of the left and right ventricles derived from the shape atlas method elaborated in [Charlène et al., JCMR 2019](https://www.sciencedirect.com/science/article/pii/S1097664723002144), and the three `control_mesh` are epi- or endo-part from the `template_mesh` and with all holes of valve replaced with new triangle faces._ 
```
template
├── template_mesh-myo.obj   
├── control_mesh-lv.obj
├── control_mesh-myo.obj
└── control_mesh-rv.obj
```

## Preprocessing
_Data preprocessing is a must so that images, segmentations, and template meshes are in the same space._ 

- For inference on new CMR data, we provide `data_preprocessing.py` to prepare the data for the network. 
_Before running the code, search for and follow instructions for producing `.nrrd` file from your `.dcm` data and do check if the predefined `source_nrrd_dir` and `output_nrrd_dir` are correct in the provided script._

- Locate `utils/create_datalist.py` to create a json file that contains a list of data for training/validation/test with five-fold crossvalidation. For simplicity, the code in this repository is based on the first fold. 
    ```
    $ python utils/create_datalist.py \
        --input_dir /path/to/your/preprocessed/data \
        --file_extension .nrrd \    # or .nii.gz based on your data format
        --task_name name_of_your_data \
        --description \             # a description of your data for reference
        --labels {'0': 'background', '1': 'lv', '2': 'lv-myo', '3'; 'rv', '4': 'rv-myo'} \                 
                                    # default segmentation labels for left and right ventricular cavity and myocardium
        --modality CT \             # or 'MR' based on your data modality
    ```

## Training
Detail structure of MorphiNet can be found in the paper.

### Training Process
The training process contains three stages:

1. **UNet**: optimising both segmentation UNet with either CT or CMR ground-truth.
2. **ResNet**: learning a continuous distance field from CT latent feature extracted from the last UNet layer. 
3. **GSN**: deforming and refining the template mesh warped based on the learnt distance field.

_We recommand using [Weights and Bias](https://wandb.ai) for monitoring the training process._ 

### Command line Running
Running the whole training process is straightforward, just use commandline tool. See details of adjustable parameters in the `control.sh` script.

A typical combination of parameters is as follows:

_Parameters of the network is customizable but not recommended._
```
    $ python main.py \

    --save_on sct \             # option to run the network on either CT ('sct') or CMR ('mr') data
    --mr_json_dir ./dataset/dataset_taskXX_f0.json # dataset file for CMR data \
    --mr_data_dir /path/to/preprocessed/CMR/data \

    --control_mesh_dir ./template/template_mesh-myo.obj \
    --subdiv_levels 2 \         # the number of GSN layers equates the subdivision level in a Loop surface subdivision method
    
    --use_ckpt /path/to/your/network/check_point/ \
    --pretrain_epochs 100 \     # 100 epochs to train segmentation UNets
    --train_epochs 150 \        # 50 epochs to train ResNet
    --max_epochs 200 \          # 50 epochs to train GSN
    --val_interval 10 \         # evaluate the network after every 10 epochs
    
    --hidden_features_gsn 64 \  # size of hidden features for GSN layers
    --pixdim 4 4 4 \            # volume spacing of downsized latent feature from the last UNet layer
    --lambda_0 2.07 \           # coefficient for Chamfer distance term in loss
    --lambda_1 0.89 \           # for point-to-surface term
    --lambda_2 2.79 \           # for Laplacian smoothing term
    --temperature 2.42 \        # temperature for warping template mesh in the learnt distance field
    --lr 0.001 \                # learning rate
    --batch_size 1 \            # batch size, unfortunately, only support batch size of 1
    --mode online \             # online mode for wandb, can be 'online', 'offline', or 'disabled'
```