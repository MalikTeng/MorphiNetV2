# MorphiNet - Cardiac Surface Reconstruction Neural Pipeline

**🫀 Cardiac AI System**: MorphiNet reconstructs 3D cardiac surface meshes from CT/CMR images using a three-stage neural pipeline (UNet → ResNet → GSN).

This repository provides a **modular, production-ready** implementation of MorphiNet with comprehensive training, testing, and data processing capabilities for multi-modal cardiac imaging datasets.

## 🎯 Current Status (2025)

### ✅ Production-Ready Features
- **Modular Architecture**: Clean separation of concerns across specialized modules
- **Phase-Specific Checkpoint Saving**: UNet, ResNet, and GSN models saved independently
- **Advanced Testing Pipeline**: Automatic checkpoint detection with comprehensive validation
- **Cross-Dataset Training**: Histogram matching for robust multi-dataset training
- **Interactive Data Orientation**: Widget-based workflow for data validation
- **Comprehensive Documentation**: Updated guides for all major features

### 🚀 Recent Major Updates
- **Enhanced Checkpoint System**: Automatic detection, phase-specific saving, best model tracking
- **Improved Testing Infrastructure**: Real model inference with WandB integration
- **Histogram Matching Fixes**: Production-ready cross-dataset intensity normalization
- **Sequential Transformations**: Generic flip/swap sequences with affine compensation
- **Memory-Efficient Processing**: Optimized data loading and GPU memory management

## 🏗️ Architecture Overview

### Three-Stage Neural Pipeline
```
Input: CT/MR Images → UNet → ResNet → GSN → Output: 3D Surface Mesh
                      ↓       ↓        ↓
                 Segmentation Distance  Mesh
                              Fields   Deformation
```

1. **UNet Stage**: Semantic segmentation (image → labels)
2. **ResNet Stage**: Distance field generation (labels → distance fields) 
3. **GSN Stage**: Template mesh deformation (distance fields → surface mesh)

### Modular Components

```
MorphiNet/
├── data/                    # Data processing pipeline
│   ├── loaders.py          # DataLoaderManager - unified data loading
│   ├── preprocessors.py    # DataPreprocessor - transforms & processing
│   ├── components.py       # UniversalCanonicalResampled, SequentialTransformd
│   └── utils/geometry.py   # Geometric transformation utilities
├── model/                   # Neural network architecture
│   ├── networks.py         # UNet, ResNet, GSN model definitions
│   ├── mesh_operations.py  # Template mesh processing
│   └── inference.py        # Sliding window inference
├── training/                # Training pipeline
│   ├── trainer.py          # MorphiNetTrainer - 3-phase training
│   ├── validators.py       # MorphiNetValidator - validation & checkpoints
│   └── losses.py           # LossManager - optimizers & loss functions
├── pipeline/                # Pipeline orchestration
│   ├── orchestrator.py     # MorphiNetOrchestrator - main coordinator
│   └── testing.py          # MorphiNetTester - inference testing
├── utils/                   # Utilities and tools
│   ├── checkpoint_manager.py # Model persistence
│   └── histogram_preprocessor.py # Cross-dataset normalization
├── data_check/              # Interactive data validation
│   └── orientation.py      # 5-section orientation workflow
└── docs/                    # Comprehensive documentation
```

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/MalikTeng/MorphiNet
cd MorphiNet

# Install using provided script (recommended)
chmod +x install_morphinet.sh
./install_morphinet.sh

# Or use conda environment
conda env create -f environment.yml
conda activate morphinet
```

### Training
```bash
# Launch training with default parameters
./control.sh

# Custom training configuration
python main.py --validation_modality ct --max_epochs 10 --batch_size 2

# Quick development test
python main.py --max_samples 2 --max_epochs 5 --mode disabled
```

### Testing (Model Inference)
```bash
# Automatic checkpoint detection (recommended)
python main.py --inference_only --test_phase unet --test_dataset scotheart --max_samples 2

# Manual checkpoint specification
python main.py --inference_only --test_phase resnet --test_dataset mmwhs \
  --use_ckpt "/path/to/checkpoint/dir" --max_samples 2

# Test multiple phases
python main.py --inference_only --test_phase both --test_dataset cap --max_samples 5
```

### Hyperparameter Sweeps
```bash
# Test sweep configuration
python test_sweep.py --simulate

# Create and run hyperparameter sweep
./run_sweep.sh create

# Join existing sweep
./run_sweep.sh join <sweep_id>

# Manual sweep control
python sweep_agent.py --create_sweep --project "MorphiNet-Sweep"
python sweep_agent.py --sweep_id <sweep_id> --count 5
```

### Data Orientation Validation
```bash
# Launch interactive orientation workflow
jupyter lab data_check/orientation.py

# Or use VS Code with Jupyter extension
code data_check/orientation.py
```

## 🗂️ Checkpoint Management

### New Checkpoint Architecture (2025)
```
/path/to/checkpoints/dynamic/
└── {run_id}/                    # e.g., ct--myo--f0--2025-07-11-0650
    └── trained_weights/
        ├── best_UNet_CT.pth     # Best UNet CT model
        ├── best_UNet_MR.pth     # Best UNet MR model  
        ├── best_ResNet.pth      # Best ResNet model
        ├── best_GSN.pth         # Best GSN model
        ├── final_UNet_CT.pth    # Final models after training
        ├── final_UNet_MR.pth
        ├── final_ResNet.pth
        ├── final_GSN.pth
        ├── {epoch}_UNet_CT.pth  # Per-epoch checkpoints
        ├── {epoch}_UNet_MR.pth
        ├── {epoch}_ResNet.pth
        └── {epoch}_GSN.pth
```

### Features
- **Automatic Detection**: System finds most recent checkpoint by modification time
- **Phase-Specific Saving**: Each training phase saves its own checkpoints
- **Best Model Tracking**: Best models saved when validation scores improve
- **Final Checkpoint**: Models saved after training completion
- **Backward Compatibility**: Supports legacy checkpoint formats

## 🔄 Cross-Dataset Training

### Histogram Matching System
MorphiNet includes production-ready histogram matching for cross-dataset intensity normalization:

```bash
# Generate histogram cache files (run once)
python -m utils.histogram_preprocessor /path/to/data

# Validate histogram matching effectiveness
python -m utils.histogram_preprocessor /path/to/data --analyze-histograms
```

### Supported Dataset Mappings
- **MR Datasets**: ACDC → CAP (normalize ACDC intensities to CAP distribution)
- **CT Datasets**: SCOTHEART → MMWHS (normalize SCOTHEART intensities to MMWHS distribution)

### Cache Files Generated
```
cdf_cache/
├── cap_reference_cdf.npy           # CAP MR target reference
├── mmwhs_reference_cdf.npy         # MMWHS CT target reference
├── acdc_to_cap_lut.npy             # ACDC→CAP mapping
└── scotheart_to_mmwhs_lut.npy      # SCOTHEART→MMWHS mapping
```

## 🧪 Testing & Validation

### Supported Test Phases
- **UNet Phase**: Tests segmentation encoders (CT: encoder_ct, MR: encoder_mr)
- **ResNet Phase**: Tests UNet→ResNet pipeline with distance field refinement
- **Both Phases**: Comprehensive testing of full pipeline components

### Supported Datasets
- **ACDC** (MR): Automated Cardiac Diagnosis Challenge
- **CAP** (MR): Cardiac Atlas Project  
- **SCOTHEART** (CT): Scottish Heart Imaging
- **MMWHS** (CT): Multi-Modality Whole Heart Segmentation

### Output Metrics
- **UNet Phase**: Dice score, IoU score per dataset
- **ResNet Phase**: MSE score for distance field prediction
- **WandB Integration**: Image visualizations and performance tracking

## 📋 Data Orientation Workflow

### Interactive 5-Section Process
1. **Preparation**: Environment setup and dependency loading
2. **Data Processing**: Process cardiac dataset with baseline orientation
3. **Initial Visualization**: Visualize raw data to identify orientation issues
4. **Custom Transformation**: Configure rotation/flip transformations via widgets
5. **Validation**: Apply transformations and validate corrected orientation

### Features
- **Widget-Based Interface**: Interactive controls for transformation parameters
- **3D Visualization**: Real-time Plotly visualizations with coordinate axes
- **Matrix Generation**: Automatic transformation matrix creation and validation
- **Reusable Transformations**: Save matrices for training/inference workflows

## 🔧 Advanced Features

### Sequential Transformations
Generic flip/swap transformation system supporting arbitrary sequences:

```python
# Custom transformation sequences
transform = SequentialTransformd(['ct_image', 'ct_label'], sequence="s:xy f:x f:z")

# ACDC backward compatibility
acdc_transform = ACDCSequentialTransform(['mr_label'])  # Unchanged
```

### Memory-Efficient Processing
- **Sample Limiting**: Configurable sample limits for development/testing
- **GPU Memory Management**: Optimized memory usage with proper cleanup
- **Batch Processing**: Efficient handling of large datasets

### Development Tools
```bash
# Quick development cycles
python main.py --max_samples 2 --max_epochs 1 --mode disabled

# Geometry transformation testing
python -m pytest tests/test_geometry.py -v --cov=data.utils.geometry

# Import validation
python -c "from data.components import UniversalCanonicalResampled; print('✅ Imports OK')"
```

## 📚 Documentation

### Key Guides
- **Training**: Complete 3-phase training pipeline documentation
- **Testing**: Comprehensive model inference and validation guide
- **Data Processing**: Multi-modal data handling and transformation
- **Troubleshooting**: Common issues and solutions
- **Architecture**: Detailed modular component documentation

### Recent Updates (2025)
- Enhanced checkpoint system with automatic detection
- Production-ready histogram matching implementation
- Comprehensive testing pipeline documentation
- Interactive data orientation workflow guide
- Cross-dataset training best practices

## 🛠️ Environment Requirements

### Core Dependencies
- **Python**: 3.10
- **PyTorch**: 2.1.0 with CUDA 11.8
- **PyTorch3D**: Latest from conda pytorch3d channel
- **MONAI**: Medical imaging transformations
- **Plotly**: Interactive 3D visualizations
- **Weights & Biases**: Experiment tracking

### Hardware Requirements
- **GPU**: CUDA-capable GPU recommended for training
- **Memory**: 16GB+ RAM for full dataset processing
- **Storage**: ~100GB for datasets and checkpoints

## 🎯 Current Development Status

### ✅ Completed Features
- **Modular Architecture Refactoring**: Complete separation of concerns
- **Enhanced Checkpoint System**: Phase-specific saving with automatic detection
- **Testing Pipeline Rewrite**: Real model inference with comprehensive validation
- **Histogram Matching Fixes**: Production-ready cross-dataset normalization
- **Interactive Data Validation**: 5-section orientation workflow
- **Sequential Transformation System**: Generic flip/swap sequences
- **WandB Hyperparameter Sweeps**: Bayesian optimization with 19 searchable parameters
- **Documentation Updates**: Comprehensive guides for all features

### 🔄 Ongoing Improvements
- Performance optimizations for large-scale training
- Additional dataset integration and validation
- Enhanced visualization capabilities
- Extended testing coverage

## 📞 Support & Contact

**Primary Contact**: [Malik Teng on LinkedIn](https://www.linkedin.com/in/malik-teng-86085149/)

**Repository**: MorphiNet - Adaptive Bi-ventricle Surface Reconstruction from Cardiovascular Imaging

---

*MorphiNet provides a complete, production-ready solution for cardiac surface reconstruction with state-of-the-art deep learning techniques and comprehensive data processing capabilities.*