#!/usr/bin/env python3
"""
Simplified CUDA extension builder with minimal configuration.
"""

import os
import torch
from torch.utils.cpp_extension import load

def build_cuda_extension():
    """Build CUDA extension with minimal, robust configuration."""
    print("Building CUDA extension with simplified configuration...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    sources = [
        os.path.join(current_dir, "voxelize_cuda.cpp"),
        os.path.join(current_dir, "cuda_kernels.cu"),
    ]
    
    try:
        # Minimal JIT compilation with RTX 3090 support
        extension = load(
            name="voxelize_cuda_ext",
            sources=sources,
            extra_cflags=['-O3'],
            extra_cuda_cflags=[
                '-O3',
                '--use_fast_math',
                '-gencode=arch=compute_86,code=sm_86',  # RTX 3090 architecture
            ],
            verbose=True,
            build_directory=os.path.join(current_dir, 'build_simple')
        )
        
        print("✅ CUDA extension built successfully!")
        
        # Test the extension
        info = extension.cuda_info()
        print("Extension info:")
        for line in info[:3]:
            print(f"  {line}")
        
        return extension
        
    except Exception as e:
        print(f"❌ Build failed: {e}")
        return None

if __name__ == "__main__":
    extension = build_cuda_extension()
    if extension:
        print("Success! Extension is ready to use.")
    else:
        print("Build failed - check error messages above")