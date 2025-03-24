#!/bin/bash

# Create the conda environment from the basic configuration
echo "Creating conda environment with Python 3.10..."
conda env create -f environment.yml

# Activate the environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate morphinet

# Fix for Ninja installation issues
echo "Ensuring Ninja is properly installed and available..."
conda install -y -c conda-forge ninja
pip install -U ninja
which ninja || echo "WARNING: Ninja is not in PATH"
export PATH=$CONDA_PREFIX/bin:$PATH
echo "Updated PATH: $PATH"

# Install NRRD file format support
echo "Installing SimpleITK and pynrrd for NRRD file format support..."
pip install SimpleITK pynrrd

# Check if IPython is installed, if not install it
python -c "import IPython" 2>/dev/null || {
  echo "IPython not found, installing via conda..."
  conda install -y ipython
}

# Check if PyTorch3D was installed, if not install it via conda
python -c "import pytorch3d" 2>/dev/null || {
  echo "PyTorch3D not found, installing via conda..."
  conda install -y pytorch3d -c pytorch3d
}

# Create a symbolic link for the ninja executable
echo "Creating symbolic link for ninja..."
if [ -f "$CONDA_PREFIX/bin/ninja" ]; then
  ln -sf "$CONDA_PREFIX/bin/ninja" "$CONDA_PREFIX/bin/ninja-build" 2>/dev/null || true
fi

# Install PyTorch Geometric dependencies with correct CUDA version
echo "Installing PyTorch Geometric dependencies..."
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-geometric==2.4.0

# Install torchdiffeq
echo "Installing torchdiffeq..."
pip install torchdiffeq

# Install seaborn
echo "Installing seaborn..."
pip install seaborn

# Verify installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA version: {torch.version.cuda}')"
python -c "import pytorch3d; print(f'PyTorch3D version: {pytorch3d.__version__}')" || {
  echo "WARNING: PyTorch3D installation failed. Attempting to install from source..."
  pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
  python -c "import pytorch3d; print(f'PyTorch3D version: {pytorch3d.__version__}')" || echo "ERROR: PyTorch3D installation failed!"
}
python -c "import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')"
python -c "import torch_scatter, torch_sparse; print('PyG extensions imported successfully')"
python -c "import seaborn; print(f'Seaborn version: {seaborn.__version__}')"
python -c "import IPython; print(f'IPython version: {IPython.__version__}')"
python -c "import SimpleITK; print(f'SimpleITK version: {SimpleITK.__version__}')" || echo "WARNING: SimpleITK not installed correctly"
python -c "import nrrd; print(f'pynrrd version: {nrrd.__version__}')" || echo "WARNING: pynrrd not installed correctly"

# Verify Ninja in a way that PyTorch's cpp_extension will find it
echo "Verifying Ninja installation..."
ninja --version 
python -c "import subprocess; print('Ninja path:', subprocess.check_output(['which', 'ninja']).decode().strip())"
echo "Create a startup script to ensure correct environment setup"

# Create activation script to ensure correct environment on startup
cat > "$CONDA_PREFIX/etc/conda/activate.d/ninja_path.sh" << 'EOF'
#!/bin/bash
export PATH=$CONDA_PREFIX/bin:$PATH
EOF
chmod +x "$CONDA_PREFIX/etc/conda/activate.d/ninja_path.sh"

echo "Installation complete! Activate the environment with: conda activate morphinet"
echo "Ninja is installed at: $(which ninja)"
