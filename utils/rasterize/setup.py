#!/usr/bin/env python3
"""
Setup script for CUDA-accelerated mesh voxelization extension.

This script compiles the CUDA extension for fast triangle mesh voxelization.
The extension provides significant speedup (10-50x) compared to CPU fallback.

Usage:
    python setup.py build_ext --inplace    # Build in current directory
    python setup.py install                # Install to Python environment

Requirements:
    - PyTorch with CUDA support
    - CUDA toolkit (compatible with PyTorch version)
    - C++ compiler with CUDA support
"""

import os
import sys
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup

def get_cuda_architectures():
    """Get CUDA architectures to compile for."""
    # Common architectures across different GPU generations
    architectures = [
        'compute_70,code=sm_70',    # V100
        'compute_75,code=sm_75',    # RTX 20xx, T4
        'compute_80,code=sm_80',    # A100, A10
        'compute_86,code=sm_86',    # RTX 30xx
    ]
    
    # Add newer architectures if CUDA version supports them
    cuda_version = torch.version.cuda
    if cuda_version:
        major, minor = map(int, cuda_version.split('.')[:2])
        if major > 11 or (major == 11 and minor >= 1):
            architectures.append('compute_87,code=sm_87')  # A30
        if major > 11 or (major == 11 and minor >= 8):
            architectures.append('compute_89,code=sm_89')  # RTX 40xx series
            architectures.append('compute_90,code=sm_90')  # H100
    
    return architectures

def check_requirements():
    """Check if all requirements are met."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Extension will be built but may not work.")
        return False
        
    # Check CUDA version compatibility
    torch_cuda_version = torch.version.cuda
    if torch_cuda_version is None:
        print("ERROR: PyTorch not compiled with CUDA support")
        return False
        
    print(f"Building CUDA extension with:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch_cuda_version}")
    
    return True

def main():
    # Check requirements
    if not check_requirements():
        if '--force' not in sys.argv:
            print("\nUse --force to build anyway (not recommended)")
            sys.exit(1)
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Source files
    sources = [
        os.path.join(current_dir, 'voxelize_cuda.cpp'),
        os.path.join(current_dir, 'cuda_kernels.cu'),
    ]
    
    # Include directories
    include_dirs = [current_dir]
    
    # Compiler flags
    cxx_flags = ['-O3']
    
    # CUDA compiler flags
    nvcc_flags = [
        '-O3',
        '--use_fast_math',
        '--expt-relaxed-constexpr',
        '-Xptxas=-v',  # Verbose PTX compilation
    ]
    
    # Add architecture flags
    architectures = get_cuda_architectures()
    for arch in architectures:
        nvcc_flags.extend(['-gencode', f'arch={arch}'])
    
    # Add compute capability for current device if available
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        current_device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(current_device)
        current_arch = f'compute_{capability[0]}{capability[1]},code=sm_{capability[0]}{capability[1]}'
        if current_arch not in [arch.split('arch=')[1] for arch in architectures if 'arch=' in arch]:
            nvcc_flags.extend(['-gencode', f'arch={current_arch}'])
            print(f"  Added current GPU architecture: {current_arch}")
    
    # Create extension
    ext = CUDAExtension(
        name='voxelize_cuda_ext',
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            'cxx': cxx_flags,
            'nvcc': nvcc_flags
        },
        verbose=True
    )
    
    # Setup
    setup(
        name='voxelize_cuda_ext',
        ext_modules=[ext],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
        python_requires='>=3.7',
        install_requires=[
            'torch>=1.9.0',
        ],
        author='MorphiNet',
        description='CUDA-accelerated triangle mesh voxelization',
        long_description=__doc__,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: C++',
            'Programming Language :: CUDA',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )

if __name__ == '__main__':
    main()