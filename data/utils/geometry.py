#!/usr/bin/env python3
"""
Geometry transformation utilities for MorphiNet data processing.

This module provides comprehensive geometric transformation utilities for cardiac
image processing, including flip and swap operations with both matrix-based and
tensor-based implementations.

Key Components:
- Matrix builders for flip and swap transformations
- Sequential transformation parsing and composition
- Tensor-level transformation application
- Shape calculation utilities

Design Philosophy:
The module provides dual implementations:
1. Matrix-based: Using 4x4 homogeneous transformation matrices
2. Tensor-based: Using PyTorch native operations for efficiency

Both implementations produce identical results and are used in different
contexts throughout the MorphiNet pipeline.
"""

import numpy as np


def create_flip_matrix(mask_shape, axis='z'):
    """
    Create index-space matrix for flipping along specified axis.
    
    Algorithm:
    Creates a 4x4 homogeneous transformation matrix that flips coordinates
    along the specified axis. The flip operation is: new_coord = (size - 1) - old_coord
    
    For axis 'z' (depth): M[0,0] = -1, M[0,3] = D-1
    For axis 'y' (height): M[1,1] = -1, M[1,3] = H-1  
    For axis 'x' (width): M[2,2] = -1, M[2,3] = W-1
    
    Args:
        mask_shape: Shape tuple (H, W, D) or (..., H, W, D)
        axis: Axis to flip ('x', 'y', or 'z')
        
    Returns:
        4x4 numpy array representing the flip transformation matrix
        
    Example:
        >>> shape = (64, 128, 128)
        >>> matrix = create_flip_matrix(shape, 'z')
        >>> # matrix[0,0] = -1, matrix[0,3] = 63
    """
    H, W, D = mask_shape[-3:]
    M_index = np.eye(4, dtype=np.float32)
    
    if axis.lower() == 'x':
        M_index[2, 2] = -1
        M_index[2, 3] = D - 1
    elif axis.lower() == 'y':
        M_index[1, 1] = -1
        M_index[1, 3] = W - 1
    elif axis.lower() == 'z':
        M_index[0, 0] = -1
        M_index[0, 3] = H - 1
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")
    
    return M_index


def create_swap_matrix(mask_shape=None, pair='xy'):
    """
    Create index-space matrix for coordinate swaps between axis pairs.
    
    Algorithm:
    Creates a 4x4 homogeneous transformation matrix that swaps coordinates
    between two spatial axes. The swap operation permutes matrix rows/columns
    to reorder coordinate axes.
    
    Coordinate mappings:
    - xy: (h,w,d) → (h,d,w) - swap X ↔ Y (D ↔ W)
    - xz: (h,w,d) → (d,w,h) - swap X ↔ Z (D ↔ H)  
    - yz: (h,w,d) → (w,h,d) - swap Y ↔ Z (W ↔ H)
    
    Matrix operations must match manual tensor operations exactly:
    - xy: transpose(0,2,1) → (h,w,d) → (h,d,w)
    - xz: transpose(2,1,0) → (h,w,d) → (d,w,h)
    - yz: transpose(1,0,2) → (h,w,d) → (w,h,d)
    
    Args:
        mask_shape: Shape tuple (optional, kept for API compatibility)
        pair: Axis pair to swap ('xy', 'xz', or 'yz')
        
    Returns:
        4x4 numpy array representing the swap transformation matrix
        
    Example:
        >>> matrix = create_swap_matrix(pair='xy')
        >>> # matrix[1,2] = 1, matrix[2,1] = 1 (swap Y and X axes)
    """
    if pair == 'xy':   # (h,w,d) -> (h,d,w) - swap X <-> Y (D <-> W)
        M = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    elif pair == 'xz': # (h,w,d) -> (d,w,h) - swap X <-> Z (D <-> H)
        M = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    elif pair == 'yz': # (h,w,d) -> (w,h,d) - swap Y <-> Z (W <-> H)
        M = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    else:
        raise ValueError(f"Invalid pair: {pair}. Must be 'xy', 'xz', or 'yz'")
    
    return M


def get_swap_output_shape(mask_shape, pair='xy'):
    """
    Calculate the output shape for axis swaps with dynamic sizing.
    
    Algorithm:
    Determines the output shape after applying a coordinate swap operation.
    This is essential for matrix-based transformations where the output
    dimensions may differ from input dimensions.
    
    Shape Transformations:
    - xy: (H, W, D) → (H, D, W) - swap height and width
    - xz: (H, W, D) → (D, W, H) - swap depth and width
    - yz: (H, W, D) → (W, H, D) - swap depth and height
    
    Args:
        mask_shape: Input shape tuple (H, W, D) or (..., H, W, D)
        pair: Axis pair being swapped ('xy', 'xz', or 'yz')
        
    Returns:
        Output shape tuple after the swap operation
        
    Example:
        >>> get_swap_output_shape((64, 128, 96), 'xy')
        (64, 96, 128)  # swapped H and W
    """
    H, W, D = mask_shape[-3:]
    
    if pair == 'xy':   # swap X <-> Y (D <-> W) → (H, D, W)
        return (H, D, W)
    elif pair == 'xz': # swap X <-> Z (D <-> H) → (D, W, H)
        return (D, W, H)
    elif pair == 'yz': # swap Y <-> Z (W <-> H) → (W, H, D)
        return (W, H, D)
    else:
        raise ValueError(f"Invalid pair: {pair}. Must be 'xy', 'xz', or 'yz'")


def apply_matrix_transformation(mask, matrix, output_shape=None):
    """
    Apply 4x4 transformation matrix to 3D mask volume with optional output shape.
    
    Algorithm:
    Applies a 4x4 homogeneous transformation matrix to a 3D volume using
    scipy.ndimage.affine_transform. The transformation matrix is inverted
    because scipy requires the inverse mapping (output → input coordinates).
    
    Mathematical Framework:
    Given transformation matrix M and point X:
    Y = M @ X  (forward transformation)
    
    For resampling, we need: X = M^(-1) @ Y  (inverse transformation)
    scipy.ndimage.affine_transform expects the 3x3 linear part and 3x1 offset.
    
    Processing Steps:
    1. Invert the 4x4 transformation matrix
    2. Extract 3x3 linear part and 3x1 offset
    3. Apply affine transformation with specified output shape
    4. Return contiguous array to avoid stride issues
    
    Args:
        mask: 3D numpy array to transform
        matrix: 4x4 transformation matrix
        output_shape: Optional output shape tuple (uses input shape if None)
        
    Returns:
        Transformed 3D numpy array with specified output shape
        
    Example:
        >>> mask = np.random.rand(64, 128, 128)
        >>> flip_matrix = create_flip_matrix(mask.shape, 'z')
        >>> flipped = apply_matrix_transformation(mask, flip_matrix)
        >>> # Returns z-flipped volume
    """
    from scipy.ndimage import affine_transform
    
    M_inv = np.linalg.inv(matrix)
    M_scipy = M_inv[:3, :3]
    offset = M_inv[:3, 3]
    
    # Use provided output shape or default to input shape
    if output_shape is None:
        output_shape = mask.shape
    
    transformed = affine_transform(
        mask, 
        M_scipy, 
        offset=offset, 
        output_shape=output_shape,
        order=1, 
        cval=0.0
    )
    
    # Ensure contiguous array to avoid negative stride issues
    return np.ascontiguousarray(transformed)


def parse_transform_sequence(seq_str):
    """
    Parse transform sequence string into list of (operation, argument) tuples.
    
    Algorithm:
    Parses a string representation of transformation sequences into structured
    tuples for programmatic processing. Supports both comma and space separation.
    
    Syntax:
    - 'f:axis' for flip operations (axis: x, y, z)
    - 's:pair' for swap operations (pair: xy, xz, yz)
    - Multiple operations separated by spaces or commas
    
    Validation:
    - Ensures all tokens follow 'operation:argument' format
    - Validates operation types ('f' for flip, 's' for swap)
    - Validates arguments (axes for flip, pairs for swap)
    
    Args:
        seq_str: String like 'f:x s:xy f:y' or 's:yz s:xz f:z f:x s:xy'
        
    Returns:
        List of tuples like [('f','x'), ('s','xy'), ('f','y')]
        
    Raises:
        ValueError: If token format is invalid or arguments are unsupported
        
    Example:
        >>> parse_transform_sequence('s:yz s:xz f:z f:x s:xy')
        [('s', 'yz'), ('s', 'xz'), ('f', 'z'), ('f', 'x'), ('s', 'xy')]
    """
    if not seq_str:
        return []
    
    # Split on whitespace and commas
    tokens = [t.strip() for t in seq_str.replace(',', ' ').split() if t.strip()]
    steps = []
    
    for token in tokens:
        if ':' not in token:
            raise ValueError(f"Invalid token '{token}' - must be format 'f:x' or 's:xy'")
        
        kind, arg = token.split(':', 1)
        kind = kind.lower()
        arg = arg.lower()
        
        if kind not in ['f', 's']:
            raise ValueError(f"Invalid operation '{kind}' - must be 'f' (flip) or 's' (swap)")
        
        if kind == 'f':
            if arg not in ['x', 'y', 'z']:
                raise ValueError(f"Invalid flip axis '{arg}' - must be x, y, or z")
        elif kind == 's':
            if arg not in ['xy', 'xz', 'yz']:
                raise ValueError(f"Invalid swap pair '{arg}' - must be xy, xz, or yz")
        
        steps.append((kind, arg))
    
    return steps


def compose_sequence_matrix(steps, initial_shape):
    """
    Return composite 4×4 index-space matrix for a flip / swap sequence.
    
    Algorithm:
    Composes multiple flip and swap transformations into a single 4x4 matrix.
    Transformations are applied left-to-right in sequence order by pre-multiplication:
    
    M_final = M_n @ M_{n-1} @ ... @ M_2 @ M_1
    
    Where each M_i is either a flip or swap matrix. The shape is tracked and
    updated after each swap operation to ensure correct matrix construction.
    
    Mathematical Framework:
    For a sequence of transformations T_1, T_2, ..., T_n applied to point X:
    Y = T_n(T_{n-1}(...T_2(T_1(X))...))
    
    This is equivalent to: Y = (M_n @ M_{n-1} @ ... @ M_2 @ M_1) @ X
    
    Args:
        steps: List of (kind, arg) tuples from parse_transform_sequence
               kind: 'f' for flip, 's' for swap
               arg: axis ('x','y','z') for flip, pair ('xy','xz','yz') for swap
        initial_shape: Initial shape tuple (H, W, D)
        
    Returns:
        composite_matrix: 4x4 numpy array representing the full transformation
        
    Example:
        >>> steps = [('s', 'yz'), ('f', 'z')]
        >>> shape = (64, 128, 128)
        >>> matrix = compose_sequence_matrix(steps, shape)
        >>> # Applies swap Y↔Z, then flip Z (in that order)
    """
    comp = np.eye(4)
    shape = initial_shape
    
    for kind, arg in steps:
        if kind == 'f':
            m = create_flip_matrix(shape, arg)
        elif kind == 's':
            m = create_swap_matrix(shape, arg)
            shape = get_swap_output_shape(shape, arg)
        else:
            raise ValueError(f"Unknown operation: {kind}")
            
        # PRE-multiply to apply transformations in sequence order
        comp = m @ comp
    
    return comp


def apply_tensor_sequence(tensor, steps):
    """
    Apply sequential transformations to a PyTorch/numpy tensor using native operations.
    
    Algorithm:
    Applies a sequence of flip and swap operations directly to tensor data using
    PyTorch native operations. This provides an exact tensor-level implementation
    of the matrix-based transformations without requiring interpolation.
    
    Transformation Details:
    - Flip operations: torch.flip(tensor, dims=[axis_index])
    - Swap operations: tensor.transpose(dim1, dim2)
    
    Dimension Mapping:
    For 4D tensors (C, H, W, D):
    - 'x' flip/axis → dim 3 (D)
    - 'y' flip/axis → dim 2 (W)  
    - 'z' flip/axis → dim 1 (H)
    
    For 3D tensors (H, W, D):
    - 'x' flip/axis → dim 2 (D)
    - 'y' flip/axis → dim 1 (W)
    - 'z' flip/axis → dim 0 (H)
    
    This implementation mirrors the matrix-based transformation exactly but uses
    direct tensor manipulation for efficiency and numerical precision.
    
    Args:
        tensor: Input tensor/array with shape (C, H, W, D) or (H, W, D)
        steps: List of (kind, arg) tuples from parse_transform_sequence
               kind: 'f' for flip, 's' for swap
               arg: axis ('x','y','z') for flip, pair ('xy','xz','yz') for swap
        
    Returns:
        Transformed tensor with potentially different spatial dimensions
        
    Example:
        >>> tensor = torch.randn(1, 64, 128, 128)  # (C, H, W, D)
        >>> steps = [('s', 'yz'), ('f', 'z')]
        >>> result = apply_tensor_sequence(tensor, steps)
        >>> # Applies swap Y↔Z, then flip Z (same as matrix version)
    """
    import torch
    
    result = tensor
    
    for kind, arg in steps:
        if kind == 'f':  # Flip operation
            if arg == 'x':  # Flip along width (last spatial dim)
                if result.dim() == 4:  # (C, H, W, D)
                    result = torch.flip(result, dims=[3])
                else:  # (H, W, D)
                    result = torch.flip(result, dims=[2])
            elif arg == 'y':  # Flip along height
                if result.dim() == 4:  # (C, H, W, D)
                    result = torch.flip(result, dims=[2])
                else:  # (H, W, D)
                    result = torch.flip(result, dims=[1])
            elif arg == 'z':  # Flip along depth
                if result.dim() == 4:  # (C, H, W, D)
                    result = torch.flip(result, dims=[1])
                else:  # (H, W, D)
                    result = torch.flip(result, dims=[0])
                    
        elif kind == 's':  # Swap operation
            if arg == 'xy':  # Swap X ↔ Y (D ↔ W)
                if result.dim() == 4:  # (C, H, W, D) → (C, W, H, D)
                    result = result.transpose(2, 3)
                else:  # (H, W, D) → (W, H, D)
                    result = result.transpose(1, 2)
            elif arg == 'xz':  # Swap X ↔ Z (D ↔ H)
                if result.dim() == 4:  # (C, H, W, D) → (C, D, W, H)
                    result = result.transpose(1, 3)
                else:  # (H, W, D) → (D, W, H)
                    result = result.transpose(0, 2)
            elif arg == 'yz':  # Swap Y ↔ Z (W ↔ H)
                if result.dim() == 4:  # (C, H, W, D) → (C, W, H, D)
                    result = result.transpose(1, 2)
                else:  # (H, W, D) → (W, H, D)
                    result = result.transpose(0, 1)
    
    return result