# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MorphiNet is a medical AI system for **Adaptive Bi-ventricle Surface Reconstruction from Cardiovascular Imaging**. It reconstructs 3D cardiac surface meshes from CT and CMR (Cardiac Magnetic Resonance) images using a three-stage neural network pipeline: UNet segmentation → ResNet distance field prediction → Graph Subdivision Networks (GSN) for mesh deformation.

## Environment Setup

**Recommended installation:**
```bash
chmod +x install_morphinet.sh
./install_morphinet.sh
```

**Basic installation:**
```bash
conda env create -f environment.yml
conda activate morphinet
```

**Critical requirements:**
- Python 3.10
- PyTorch 2.1.0 with CUDA 11.8
- PyTorch3D (medical mesh operations)
- MONAI 1.3.0 (medical imaging framework)
- Custom CUDA extensions (automatically built via Ninja)

## Common Commands

### Training
```bash
# Automated training with preset parameters
chmod +x control.sh
./control.sh

# Manual training
python main.py --save_on sct --mr_json_dir ./dataset/dataset_task11_f0.json --template_mesh_dir ./template/template_mesh-myo.obj --max_epochs 100 --pretrain_epochs 50 --train_epochs 75
```

### Inference/Testing
```bash
python test.py --save_on cap --mr_json_dir ./dataset/dataset_task11_f0.json --output_dir /path/to/output --ckpt_dir /path/to/checkpoint --template_mesh_dir ./template/template_mesh-myo.obj
```

### Data Preprocessing
```bash
# Preprocess CMR data from DICOM
python data_preprocessing.py

# Create dataset JSON file
python utils/create_datalist.py --input_dir /path/to/data --file_extension .nrrd --task_name your_task --modality CT --labels '{"0": "background", "1": "lv", "2": "lv-myo", "3": "rv", "4": "rv-myo"}'
```

## Architecture

### Three-Stage Pipeline
1. **UNet Stage** (`--pretrain_epochs`): Cardiac structure segmentation
2. **ResNet Stage** (`--train_epochs`): Signed distance field prediction  
3. **GSN Stage** (`--max_epochs` - `--train_epochs`): Template mesh deformation using Graph Subdivision Networks

### Key Directories
- **`/data/`** - Data loading, MONAI transforms, preprocessing pipeline
- **`/model/`** - Neural architectures (GSN, ResNet, UNet variants)
- **`/utils/`** - Visualization, metrics, custom CUDA rasterization operations
- **`/template/`** - Template mesh files (.obj format) for cardiac structures
- **`/dataset/`** - JSON configuration files for cross-validation splits
- **`/pretrained/`** - Pre-trained model weights

### Multi-Modal Training
The system supports simultaneous CT + MR training. Use `--ct_ratio` to control the proportion of CT data in mixed training.

## Important Parameters

### Network Architecture
- `--hidden_features_gsn 64` - GSN layer feature dimensions
- `--subdiv_levels 2` - Number of graph subdivision layers
- `--pixdim 4 4 4` - Spatial resolution of UNet feature maps

### Loss Function Coefficients
- `--lambda_0 0.18` - Chamfer distance weight
- `--lambda_1 0.64` - Laplacian smoothing weight
- `--iteration 10` - Distance field warping iterations

### Data Handling
- `--_4d` - Enable for 4D CMR data
- `--_mr` - Train exclusively on MR data
- `--crop_window_size 128 128 128` - Input patch size

## Data Organization Requirements

**Dataset structure:**
```
DATASET_NAME/
├── imagesTr/          # Training images (.nii.gz or .nrrd)
├── labelsTr/          # Training segmentations  
├── imagesTs/          # Test images
└── labelsTs/          # Test segmentations
```

**Template meshes:**
```
template/
├── template_mesh-myo.obj    # Base myocardium template
├── control_mesh-lv.obj      # Left ventricle control mesh
├── control_mesh-myo.obj     # Myocardium control mesh
└── control_mesh-rv.obj      # Right ventricle control mesh
```

## Modular Architecture (NEW)

The codebase has been refactored from a monolithic structure into clean, modular components:

### New Structure
- **`data/`** - `DataLoaderManager`, `DataPreprocessor` for unified data handling
- **`model/`** - `MeshOperations`, `ModelInference` for model-specific operations  
- **`training/`** - `MorphiNetTrainer`, `MorphiNetValidator`, `LossManager` for training logic
- **`evaluation/`** - `MorphiNetMetrics` for evaluation and metrics computation
- **`pipeline/`** - `MorphiNetOrchestrator` for coordinating all components
- **`utils/`** - `CheckpointManager` for model persistence

### Usage Patterns
```bash
# New modular interface
python main_modular.py --mode online --save_on sct

# Component demonstration
python demo_modular.py

# Backward compatible (original interface still works)
python main.py --mode online --save_on sct
```

### Programming Interface
```python
# New modular approach
from run_modular import create_training_pipeline
pipeline = create_training_pipeline(super_params)
pipeline.train_full_pipeline()

# Or use orchestrator directly
from pipeline.orchestrator import MorphiNetOrchestrator
orchestrator = MorphiNetOrchestrator(super_params)

# Backward compatible
from run import TrainPipeline  # Still works identically
```

## Development Notes

- **Batch size limitation**: Only supports `--batch_size 1` due to mesh processing constraints
- **CUDA dependency**: Custom mesh rasterization operations require proper CUDA/Ninja setup
- **Memory management**: Uses MONAI caching (`--cache_rate 1.0`) for medical image data
- **Experiment tracking**: Weights & Biases integration (`--mode online/offline/disabled`)
- **Interactive visualization**: Plotly-based 3D mesh visualization in `/iframe_figures/`
- **Modular design**: Components can be tested and used independently
- **Backward compatibility**: All existing code continues to work unchanged

## Testing Framework

- **Component testing**: `python demo_modular.py` tests all modular components
- **Integration testing**: Individual components can be imported and tested separately
- **Legacy testing**: `test.py` for inference validation (original approach)
- **Visual validation**: Mesh reconstruction inspection and Plotly visualizations
- **Quantitative metrics**: Computed via `evaluation/metrics.py` or `utils/tools.py`