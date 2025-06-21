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
python main.py --validation_modality ct --mr_json_dir ./dataset/dataset_task11_f0.json --template_mesh_dir ./template/template_mesh-myo.obj --max_epochs 100 --pretrain_epochs 50 --train_epochs 75
```

### Inference/Testing
```bash
python test.py --validation_modality mr --target acdc --mr_json_dir ./dataset/dataset_task11_f0.json --output_dir /path/to/output --ckpt_dir /path/to/checkpoint --template_mesh_dir ./template/template_mesh-myo.obj
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
- `--validation_modality ct/mr` - Validation modality ('ct' for CT data, 'mr' for MR data)
- `--target dataset_name` - Target dataset identifier for testing (e.g., 'acdc', 'cap', 'scotheart')
- `--crop_window_size 128 128 128` - Input patch size
- Note: Deprecated parameters `--_4d` and `--_mr` have been removed

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

## Modular Architecture

The codebase has been fully refactored from a monolithic structure (preserved in `@legacy_code/`) into clean, modular components:

### Current Modular Structure
- **`data/`** - `DataLoaderManager`, `DataPreprocessor` for unified data handling
- **`model/`** - `MeshOperations`, `ModelInference` for model-specific operations  
- **`training/`** - `MorphiNetTrainer`, `MorphiNetValidator`, `LossManager` for training logic
- **`evaluation/`** - `MorphiNetMetrics` for evaluation and metrics computation
- **`pipeline/`** - `MorphiNetOrchestrator` for coordinating all components
- **`utils/`** - `CheckpointManager` for model persistence

### Training Commands (Updated)
```bash
# Primary training interface - automated 3-phase pipeline
chmod +x control.sh
./control.sh

# Direct modular training
python main.py --validation_modality ct --mr_json_dir ./dataset/dataset_task11_f0.json --template_mesh_dir ./template/template_mesh-myo.obj --max_epochs 3 --pretrain_epochs 1 --train_epochs 2
```

### Programming Interface
```python
# Recommended modular approach
from pipeline.orchestrator import MorphiNetOrchestrator
orchestrator = MorphiNetOrchestrator(super_params)
orchestrator.train_full_pipeline()  # Automated 3-phase training

# Alternative simplified interface
from run_modular import create_training_pipeline
pipeline = create_training_pipeline(super_params)
pipeline.train_full_pipeline()

# Backward compatibility (legacy interface still works)
from run import TrainPipeline
pipeline = TrainPipeline(super_params, seed=42, num_workers=4)
```

### Individual Component Usage
```python
# Use data loading independently
from data.loaders import DataLoaderManager
loader_manager = DataLoaderManager(config, num_workers=4)

# Use mesh operations independently  
from model.mesh_operations import MeshOperations
mesh_ops = MeshOperations(config)

# Use training components independently
from training.trainer import MorphiNetTrainer
trainer = MorphiNetTrainer(config, models, optimizers, ...)
```

## Development Notes

- **Batch size limitation**: Only supports `--batch_size 1` due to mesh processing constraints
- **CUDA dependency**: Custom mesh rasterization operations require proper CUDA/Ninja setup
- **Memory management**: Uses MONAI caching (`--cache_rate 1.0`) for medical image data
- **Experiment tracking**: Weights & Biases integration (`--mode online/offline/disabled`)
- **Interactive visualization**: Plotly-based 3D mesh visualization in `/iframe_figures/`
- **Modular design**: Components can be tested and used independently
- **Clean architecture**: Only modular components supported
- **Sample limiting**: Use `--max_samples N` to limit dataset size for testing (0 = full dataset)
- **Half precision**: PyTorch3D mesh operations require float32, automatically handled

## Testing Framework

- **Component testing**: `python demo_modular.py` tests all modular components
- **Integration testing**: Individual components can be imported and tested separately
- **Inference testing**: Use modular pipeline for inference validation
- **Visual validation**: Mesh reconstruction inspection and Plotly visualizations
- **Quantitative metrics**: Computed via `evaluation/metrics.py` or `utils/tools.py`

## Operational Flow Details

- MorphiNet training consists of three stages: UNet phase, ResNet phase, and GSN phase
- **UNet Phase**:
  - Loads both MR and CT data
  - Trains encoder_ct and encoder_mr networks
  - Validates both MR and CT data encoders
- **ResNet Phase**:
  - Uses only CT data
  - Passes frozen encoder_ct output to ResNet
  - Combines ResNet output with frozen encoder_ct for final prediction
  - Validates encoder_ct + ResNet combination
- **GSN Phase**:
  - Uses only CT data
  - Employs frozen encoder_ct and frozen ResNet
  - Generates prediction converted to distance field
  - Processes distance field with warped template mesh in GSN network
  - Validates full encoder_ct + ResNet + GSN pipeline
- **Validation Modes**:
  - When `validation_modality == 'ct'`: CT data used for validation/testing
  - When `validation_modality == 'mr'`: MR data used for validation/testing
- **Parameter Changes**:
  - `save_on` renamed to `validation_modality` for clarity (now uses 'ct'/'mr' values)
  - Added `target` parameter for testing dataset identification
  - Added `max_samples` parameter for development/testing (0 = full dataset)
  - Deprecated parameters `_mr` and `_4d` have been removed
  - Legacy training modes removed - only modular architecture supported

## Legacy Code Management

- **Repository Policy**:
  - `@legacy_code/` directory contains the original monolithic implementation
  - Legacy code should NOT be edited but preserved for reference
  - Contains older versions of code useful for understanding system evolution
  - Original 2500+ line monolithic `run.py` preserved as historical reference

## Recent Updates (Previous Session)

### Bug Fixes Applied
- **Training Loss Calculation**: Fixed zero loss issue by implementing dynamic dataloader access
- **WandB Media Upload**: Resolved through proper dataloader integration (6+ media files per epoch)  
- **Half Precision Compatibility**: Fixed PyTorch3D mesh operations with float32 conversion
- **Sample Limiting**: Added configurable sample limiting for faster development cycles

### Current Status
- ✅ **Full Pipeline**: All three training phases (UNet → ResNet → GSN) working correctly
- ✅ **Loss Calculation**: Training losses properly computed (UNet: ~5.8, ResNet: ~2.5)
- ✅ **Validation**: Best model tracking and visualization generation operational
- ✅ **Logging**: WandB integration fully functional with media uploads
- ✅ **Memory Management**: Improved through modular architecture and garbage collection