#!/bin/bash
set -e

# Create conda environment from configuration
echo "Creating morphinet environment..."
conda env create -f environment.yml

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate morphinet

# Ensure PATH includes conda bin for ninja
export PATH=$CONDA_PREFIX/bin:$PATH

# Create ninja symlink if needed
if [ -f "$CONDA_PREFIX/bin/ninja" ]; then
  ln -sf "$CONDA_PREFIX/bin/ninja" "$CONDA_PREFIX/bin/ninja-build" 2>/dev/null || true
fi

# Install PyTorch Geometric optional dependencies
echo "Installing PyTorch Geometric optional dependencies..."
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Verify core installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch {torch.__version__} (CUDA {torch.version.cuda})')"
python -c "import pytorch3d; print(f'PyTorch3D {pytorch3d.__version__}')"
python -c "import torch_geometric; print(f'PyTorch Geometric {torch_geometric.__version__}')"
python -c "import monai; print(f'MONAI {monai.__version__}')"
ninja --version

# Create PATH activation script
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/ninja_path.sh" << 'EOF'
#!/bin/bash
export PATH=$CONDA_PREFIX/bin:$PATH
EOF
chmod +x "$CONDA_PREFIX/etc/conda/activate.d/ninja_path.sh"

echo "✅ Installation complete! Activate with: conda activate morphinet"
