#!/usr/bin/env python3
"""
Convenient build script for CUDA voxelization extension.

This script provides just-in-time compilation of the CUDA extension
with automatic fallback and error handling.
"""

import os
import sys
import torch
import warnings
from torch.utils.cpp_extension import load

def get_extension_sources():
    """Get source files for the extension."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(current_dir, "voxelize_cuda.cpp"),
        os.path.join(current_dir, "cuda_kernels.cu"),
        os.path.join(current_dir, "cuda_geometry.cuh"),  # Header dependency
    ]

def check_source_files():
    """Check if all required source files exist."""
    sources = get_extension_sources()
    missing_files = []
    
    for source in sources:
        if not os.path.exists(source):
            missing_files.append(source)
    
    if missing_files:
        print("ERROR: Missing source files:")
        for file in missing_files:
            print(f"  {file}")
        return False
    
    return True

def build_extension_jit(verbose=True):
    """Build extension using just-in-time compilation."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check requirements
    if not check_source_files():
        return None
        
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, extension may not work properly")
    
    # Get sources (only .cpp and .cu files for compilation)
    sources = get_extension_sources()[:2]  # Exclude .cuh files
    
    print("Building CUDA extension...")
    if verbose:
        print(f"  Sources: {[os.path.basename(s) for s in sources]}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {torch.version.cuda or 'Not available'}")
    
    try:
        # JIT compilation
        extension = load(
            name="voxelize_cuda_ext",
            sources=sources,
            extra_cflags=['-O3'],
            extra_cuda_cflags=[
                '-O3',
                '--use_fast_math', 
                '--expt-relaxed-constexpr',
                # Target common architectures
                '-gencode', 'arch=compute_70,code=sm_70',  # V100
                '-gencode', 'arch=compute_75,code=sm_75',  # RTX 20xx
                '-gencode', 'arch=compute_80,code=sm_80',  # A100
                '-gencode', 'arch=compute_86,code=sm_86',  # RTX 30xx
            ],
            extra_include_paths=[current_dir],
            verbose=verbose,
            build_directory=os.path.join(current_dir, 'build')
        )
        
        print("✅ CUDA extension built successfully!")
        
        # Test the extension
        if torch.cuda.is_available():
            try:
                info = extension.cuda_info()
                print("Extension info:")
                for line in info[:3]:
                    print(f"  {line}")
            except Exception as e:
                print(f"WARNING: Extension test failed: {e}")
        
        return extension
        
    except Exception as e:
        print(f"❌ Failed to build CUDA extension: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure CUDA toolkit is installed and compatible with PyTorch")
        print("2. Check that nvcc is in PATH")
        print("3. Verify C++ compiler supports C++14")
        print("4. Try: conda install cudatoolkit-dev")
        return None

def build_extension_setup():
    """Build extension using setup.py (more robust)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    setup_path = os.path.join(current_dir, 'setup.py')
    
    if not os.path.exists(setup_path):
        print(f"ERROR: {setup_path} not found")
        return False
    
    print("Building extension using setup.py...")
    import subprocess
    
    try:
        # Build in-place
        result = subprocess.run([
            sys.executable, setup_path, 
            'build_ext', '--inplace'
        ], cwd=current_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Extension built successfully with setup.py!")
            print(result.stdout)
            return True
        else:
            print("❌ Setup.py build failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Failed to run setup.py: {e}")
        return False

def main():
    """Main build function with multiple strategies."""
    print("CUDA Voxelization Extension Builder")
    print("=" * 40)
    
    # Strategy 1: JIT compilation (faster for development)
    print("\nTrying just-in-time compilation...")
    extension = build_extension_jit(verbose=True)
    
    if extension is not None:
        print("\n✅ JIT compilation successful!")
        return extension
    
    # Strategy 2: Setup.py compilation (more robust)  
    print("\nJIT failed, trying setup.py compilation...")
    if build_extension_setup():
        print("\n✅ Setup.py compilation successful!")
        try:
            import voxelize_cuda_ext
            return voxelize_cuda_ext
        except ImportError as e:
            print(f"❌ Failed to import built extension: {e}")
            return None
    
    print("\n❌ All build strategies failed!")
    print("\nFallback: The system will use CPU-only Trimesh implementation")
    return None

if __name__ == "__main__":
    extension = main()
    if extension:
        print(f"\nExtension loaded: {extension}")
        sys.exit(0)
    else:
        sys.exit(1)