# MorphiNet Data Orientation Interactive Workflow

**🫀 Cardiac AI System**: MorphiNet reconstructs 3D cardiac surface meshes from CT/CMR images using a three-stage neural pipeline (UNet → ResNet → GSN).

This repository provides an **interactive data orientation workflow** via `data_check/orientation.py` to ensure your cardiac imaging data is correctly oriented before training or inference.

## 🎯 Quick Start - Data Orientation Workflow

The `orientation.py` script provides a **5-section interactive notebook** to:
1. **Process** your cardiac dataset (baseline)
2. **Visualize** the raw orientation 
3. **Configure** custom transformations (rotation/flip)
4. **Apply** transformations and re-process data
5. **Validate** the corrected orientation

### Prerequisites

1. **Install MorphiNet Environment**:
   ```bash
   git clone https://github.com/MalikTeng/MorphiNet
   cd MorphiNet
   conda env create -f environment.yml
   conda activate morphinet
   ```

2. **Required Dependencies**:
   - PyTorch 2.1.0 with CUDA 11.8
   - `ipywidgets` (for interactive interface)
   - `plotly` (for 3D visualization)
   - `trimesh` (for mesh operations)

3. **Verify Project Structure**:
   ```
   MorphiNet/
   ├── data/                    # Data processing modules
   ├── model/                   # Neural network models  
   ├── dataset/                 # Dataset JSON configurations
   ├── template/                # Template mesh files
   └── data_check/              # Interactive workflow tools
       └── orientation.py       # Main orientation script
   ```

### How to Launch

**Option 1 - Jupyter Notebook (Recommended)**:
```bash
# Open in JupyterLab
jupyter lab data_check/orientation.py

# Or VS Code with Jupyter extension
code data_check/orientation.py
```

**Option 2 - Command Line**:
```bash
# Convert and execute as notebook
jupyter nbconvert --to notebook --execute data_check/orientation.py
```

## 📋 Section-by-Section Workflow

### Section 1: 📋 Preparation
**Purpose**: Environment setup and dependency loading

**What happens**:
- Automatically detects MorphiNet root directory
- Loads heavy dependencies (torch, plotly, trimesh, widgets)
- Runs environment checks for required folders
- Configures optimized processing parameters

**Expected output**:
```
✅ All dependencies loaded successfully
✅ MorphiNet project structure verified
✅ Interactive widgets available
✅ Preparation completed in X.XX seconds
```

**If you see errors**: Install missing dependencies or verify folder structure.

---

### Section 2: 📂 Data Processing (Baseline)
**Purpose**: Process cardiac dataset with default orientation

**Interactive interface**:
- **Dataset dropdown**: Choose SCOTHEART (CT), ACDC (MR), CAP (MR), or MMWHS (CT)
- **Max samples slider**: Start with 1 sample for testing
- **Process button**: Begins data processing

**What to do**:
1. Select a dataset (SCT recommended for demos)
2. Click "🔄 Process Data"
3. Wait for processing completion

**Expected output**:
```
✅ Successfully processed 1 sample(s)!
📊 Dataset: SCT
📊 Modality: CT
📋 Image shape: [1, 200, 32, 32, 32]
📋 Label shape: [200, 32, 32, 32]
```

**Result**: Baseline (untransformed) data ready for visualization.

---

### Section 3: 🎨 Initial Visualization (Baseline)
**Purpose**: Visualize raw data orientation to identify issues

**Interactive interface**:
- **Check data button**: Loads processed data from Section 2
- **Sample dropdown**: Select which sample to visualize
- **Structure dropdown**: Choose Left Ventricle (1), Myocardium (2), or Right Ventricle (3)
- **Max points slider**: Control point cloud density (5K-15K)

**What to do**:
1. Click "🔍 Check Processed Data"
2. Select sample and structure (Myocardium recommended)
3. Click "🎨 Create Visualization"

**Expected output**:
- Interactive 3D Plotly visualization
- Surface point cloud + template mesh overlay
- Colored coordinate axes (X=red, Y=green, Z=blue)

**What to look for**: Note the orientation of heart structures relative to coordinate axes. This is your **baseline reference**.

---

### Section 4: 🔧 Custom Transformation Matrix Generation
**Purpose**: Create rotation/flip transformations to correct orientation

**Interactive controls**:
- **Rotation Axis**: X, Y, or Z axis
- **Direction**: Clockwise (CW) or Counter-clockwise (CCW)
- **Count**: 0-3 quarter turns (×90°)
- **Flip Plane**: XY, XZ, or YZ plane
- **Sequence**: Order of operations (Flip → Rotation or Rotation → Flip)

**Three-step workflow**:
1. **Configure**: Set rotation and flip parameters
2. **Generate**: Click "🔄 Generate Matrix" to preview transformation
3. **Confirm**: Click "✅ Confirm Transform" to store matrix

**Expected output**:
```
📐 PHYSICAL-SPACE MATRIX (For User Inspection):
[[ 0.  0.  1.  0.]
 [ 0.  1.  0.  0.]
 [-1.  0.  0.  0.]
 [ 0.  0.  0.  1.]]

📐 Basis Vector Transformations:
   +X direction [1, 0, 0] → [0, 0, -1]
   +Y direction [0, 1, 0] → [0, 1, 0] 
   +Z direction [0, 0, 1] → [1, 0, 0]

✅ Transformation confirmed and stored!
```

**Result**: Custom transformation matrix ready for data processing.

---

### Section 4.5: 🚀 Apply Custom Transformation to Data
**Purpose**: Re-process data with confirmed transformation matrix

**What happens**:
- Automatically detects transformation matrix from Section 4
- Uses same dataset selected in Section 2
- Applies custom transformation during processing

**What to do**:
1. Verify transformation matrix is confirmed
2. Select same dataset as Section 2
3. Click "🔄 Process Data"

**Expected output**:
```
🔄 Applying custom transformation matrix to data processing
✅ Successfully processed 1 sample(s) with transformation!
🔍 Verifying transformation was applied...
```

**Result**: Transformed data ready for final validation.

---

### Section 5: 🎭 Visualize Transformed Data  
**Purpose**: Validate transformation results and compare with baseline

**What to do**:
1. Click "🔍 Check Processed Data" (automatically uses transformed data)
2. Select same sample and structure as Section 3
3. Click "🎨 Create Visualization"

**Expected output**:
- New 3D visualization with corrected orientation
- Same coordinate axes for easy comparison with Section 3

**Validation checklist**:
- ✅ Heart structures aligned with expected coordinate directions
- ✅ Template mesh properly overlays point cloud
- ✅ Coordinate axes show intuitive anatomical orientation

## 🔄 Common Transformation Examples

### Y-axis 90° CCW Rotation (Most Common)
```
Configuration:
- Rotation Axis: Y-axis
- Direction: Counter-clockwise  
- Count: 1
- Flip: None

Effect: Rotates X→Z, Z→-X (frontal view correction)
```

### XY Plane Flip
```
Configuration:
- Rotation: None
- Flip Plane: XY

Effect: Flips Z-axis direction (superior/inferior correction)
```

### Combined Transform
```
Configuration:
- Rotation Axis: Z-axis, CW, Count: 1
- Flip Plane: YZ  
- Sequence: Flip → Rotation

Effect: First flips X-axis, then rotates around Z-axis
```

## 🛠️ Troubleshooting

### Common Issues

**"No processed data found"**
- Solution: Complete Section 2 first

**"Interactive widgets not available"**
- Solution: `pip install ipywidgets`, restart kernel

**"No surface mesh extracted"**
- Solution: Try different label value (1, 2, or 3) or different dataset

**Visualization shows no transformation**
- Solution: Check Section 4 confirmation message, ensure matrix was stored

**"Files not found" during processing**
- Solution: Verify dataset paths in `dataset/` folder and data directories

### Performance Tips

- Start with 1 sample for testing
- SCT dataset is most reliable for demos
- Use 8,000 points for good quality/speed balance
- Processing typically takes 1-2 minutes per sample

## 🎯 Expected Final Result

After completing all sections, you will have:

1. ✅ **Baseline data** processed and visualized (Sections 2 & 3)
2. ✅ **Custom transformation matrix** defined and confirmed (Section 4)  
3. ✅ **Transformed data** processed with new orientation (Section 4.5)
4. ✅ **Validation visualization** showing corrected alignment (Section 5)
5. ✅ **Reusable transformation matrix** for training/inference workflows

The confirmed transformation matrix can be applied to your entire dataset during MorphiNet training or inference using the `custom_affine_matrix` parameter.

## 📚 Next Steps

Once data orientation is validated:

1. **Training**: Use the transformation matrix in main MorphiNet training pipeline
2. **Inference**: Apply same transformation to new data for consistent results  
3. **Batch Processing**: Scale the confirmed transformation to process entire datasets

For full MorphiNet training and inference instructions, see the complete documentation in the repository.

## 🏗️ Architecture Overview

MorphiNet features a modular architecture:

```
MorphiNet/
├── data/                    # Data processing and loading
├── model/                   # Neural network models (UNet, ResNet, GSN)
├── training/                # Training components and losses
├── evaluation/              # Evaluation and metrics
├── pipeline/                # Pipeline orchestration
├── utils/                   # Utilities and checkpoint management
├── data_check/              # Interactive orientation workflow
│   └── orientation.py       # Main orientation validation script
├── main.py                  # Main training script (modular)
├── run.py                   # Backward compatibility layer
└── legacy_code/             # Original monolithic implementation
```

## Installation

### Method 1: Using environment.yml (Basic installation)
```bash
conda env create -f environment.yml
conda activate morphinet
```

### Method 2: Using the installation script (Recommended)
```bash
chmod +x install_morphinet.sh
./install_morphinet.sh
```

### Environment Requirements
- **Python**: Version 3.10
- **PyTorch**: Version 2.1.0 with CUDA 11.8 support
- **PyTorch3D**: Installed from conda using the pytorch3d channel
- **CUDA**: Version 11.8 (must match PyTorch CUDA version)

---

**Contact**: [Malik Teng on LinkedIn](https://www.linkedin.com/in/malik-teng-86085149/)

**Repository**: MorphiNet - Adaptive Bi-ventricle Surface Reconstruction from Cardiovascular Imaging