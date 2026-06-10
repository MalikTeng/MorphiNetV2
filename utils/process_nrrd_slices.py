#!/usr/bin/env python3
"""
NRRD to NIFTI Medical Image Processing Script

This script processes cardiac MRI NRRD files by:
1. Stacking slice files per patient to create 4D volumes
2. Decomposing 4D volumes into separate 3D time frames
3. Saving each time frame as individual NIFTI files with proper affine matrices

Author: MorphiNet Processing Pipeline
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import logging
import argparse
from typing import List, Tuple, Optional, Dict, Any

_IMAGING_IMPORT_ERROR = None
try:
    import numpy as np
    import nrrd
    import nibabel as nib
except ImportError as exc:
    _IMAGING_IMPORT_ERROR = exc
    np = None
    nrrd = None
    nib = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_config import get_path_default


def _require_imaging_dependencies() -> None:
    if _IMAGING_IMPORT_ERROR is not None:
        raise RuntimeError(
            'process_nrrd_slices.py requires numpy, pynrrd, and nibabel. '
            'Activate the configured conda environment or install those packages before processing NRRD data.'
        ) from _IMAGING_IMPORT_ERROR


def setup_logging() -> logging.Logger:
    """Set up logging configuration for the processing pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('nrrd_processing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def get_nrrd_files_for_patient(patient_folder: str) -> List[Tuple[int, str]]:
    """
    Extract and sort NRRD files from a patient folder by sequence number.

    Args:
        patient_folder: Path to patient directory containing NRRD files

    Returns:
        List of (sequence_number, filename) tuples sorted by sequence number
    """
    nrrd_files = []

    if not os.path.exists(patient_folder):
        raise FileNotFoundError(f"Patient folder not found: {patient_folder}")

    for filename in os.listdir(patient_folder):
        if filename.endswith('.nrrd'):
            try:
                # Extract sequence number (first part before space)
                sequence_number = int(filename.split(' ')[0])
                nrrd_files.append((sequence_number, filename))
            except (ValueError, IndexError) as e:
                logging.warning(f"Could not parse sequence number from {filename}: {e}")
                continue

    # Sort by sequence number to maintain slice ordering
    nrrd_files.sort(key=lambda x: x[0])

    if not nrrd_files:
        raise ValueError(f"No valid NRRD files found in {patient_folder}")

    return nrrd_files


def read_and_stack_slices(patient_folder: str, nrrd_files: List[Tuple[int, str]]) -> Tuple[np.ndarray, Dict[str, Any], List[np.ndarray]]:
    """
    Read all NRRD slice files for a patient and stack them into a 4D volume.

    Args:
        patient_folder: Path to patient directory
        nrrd_files: List of (sequence_number, filename) tuples

    Returns:
        Tuple of (4D_volume_array, metadata_from_first_slice, list_of_slice_origins)
    """
    slices = []
    first_header = None
    slice_origins = []

    logging.info(f"Processing {len(nrrd_files)} slices for patient")

    for seq_num, filename in nrrd_files:
        file_path = os.path.join(patient_folder, filename)

        try:
            data, header = nrrd.read(file_path)

            # Store metadata from first slice
            if first_header is None:
                first_header = header
                logging.info(f"Reference slice shape: {data.shape}")

            # Store slice origin for affine matrix calculation
            slice_origins.append(header.get('space origin'))

            # Validate data shape consistency
            expected_shape = first_header.get('sizes', data.shape)
            if not np.array_equal(data.shape, expected_shape):
                logging.warning(f"Shape mismatch in slice {seq_num}: {data.shape} vs expected {expected_shape}")

            # Remove singleton dimension if present (from (20, 208, 256, 1) to (20, 208, 256))
            if data.ndim == 4 and data.shape[-1] == 1:
                data = data.squeeze(-1)

            slices.append(data)

        except Exception as e:
            logging.error(f"Failed to read slice {seq_num} ({filename}): {e}")
            raise

    # Stack slices along new axis: (time, height, width, slices)
    try:
        volume_4d = np.stack(slices, axis=-1)
        logging.info(f"Created 4D volume with shape: {volume_4d.shape}")
        return volume_4d, first_header, slice_origins

    except Exception as e:
        logging.error(f"Failed to stack slices: {e}")
        raise


def extract_affine_matrix(header: Dict[str, Any], slice_origins: List[np.ndarray] = None) -> np.ndarray:
    """
    Extract and construct a valid 4x4 affine matrix from NRRD header.

    Based on the NRRD space directions structure:
    - Row 0: Time direction (contains NaN, should be ignored)
    - Row 1: Spatial direction for height dimension
    - Row 2: Spatial direction for width dimension
    - Row 3: Spatial direction for slice dimension (unit vector)

    Args:
        header: NRRD header dictionary
        slice_origins: List of space origins from all slices (for calculating actual slice spacing)

    Returns:
        4x4 affine transformation matrix
    """
    # Initialize identity matrix
    affine = np.eye(4)

    # Extract space directions and space origin
    space_directions = header.get('space directions')
    space_origin = header.get('space origin')

    if space_directions is not None:
        # Convert to numpy array if needed
        if not isinstance(space_directions, np.ndarray):
            space_directions = np.array(space_directions)

        # NRRD space_directions format: (4, 3) where
        # Row 0: Time (NaN), Row 1-3: Spatial directions
        if space_directions.shape[0] >= 4 and space_directions.shape[1] >= 3:
            # Following the pattern from data/components.py:
            # m[:3, 0] = affine[1, :3]  # Row 1 -> Column 0
            # m[:3, 1] = affine[2, :3]  # Row 2 -> Column 1
            # m[:3, 2] = affine[3, :3]  # Row 3 -> Column 2

            # Extract spatial directions (skip row 0 which contains time/NaN)
            affine[:3, 0] = space_directions[1, :3]  # Height direction
            affine[:3, 1] = space_directions[2, :3]  # Width direction

            # For slice direction, calculate actual spacing from slice origins
            slice_direction_unit = space_directions[3, :3]  # Unit vector

            if slice_origins is not None and len(slice_origins) > 1:
                # Calculate actual slice spacing from origins
                slice_origins_array = np.array(slice_origins)
                actual_slice_spacing = slice_origins_array[1] - slice_origins_array[0]

                # Use actual spacing as slice direction (includes proper magnitude and direction)
                affine[:3, 2] = actual_slice_spacing

                logging.debug(f"Calculated actual slice spacing: {actual_slice_spacing}")
                logging.debug(f"Slice direction unit vector: {slice_direction_unit}")
                logging.debug(f"Actual spacing magnitude: {np.linalg.norm(actual_slice_spacing):.3f}")
            else:
                # Fallback to unit vector (not ideal, but better than crashing)
                affine[:3, 2] = slice_direction_unit
                logging.warning("No slice origins provided, using unit slice direction vector")

            logging.debug(f"Extracted spatial directions from NRRD:")
            logging.debug(f"  Height (row 1): {space_directions[1, :3]}")
            logging.debug(f"  Width (row 2): {space_directions[2, :3]}")
            logging.debug(f"  Slice (calculated): {affine[:3, 2]}")

        else:
            logging.warning(f"Unexpected space_directions shape: {space_directions.shape}")
            # Fallback to identity for spatial part
            pass

    if space_origin is not None:
        # Set translation part (space origin)
        space_origin = np.array(space_origin)
        if len(space_origin) >= 3:
            affine[:3, 3] = space_origin[:3]
            logging.debug(f"Set translation vector: {space_origin[:3]}")

    # Validate the affine matrix doesn't contain NaN in spatial components
    if np.any(np.isnan(affine[:3, :3])):
        logging.warning("Affine matrix contains NaN values, using identity matrix")
        affine = np.eye(4)
        if space_origin is not None and len(space_origin) >= 3:
            affine[:3, 3] = space_origin[:3]

    return affine


def validate_nifti_file(file_path: str) -> bool:
    """
    Validate that a NIFTI file was created correctly and can be read.

    Args:
        file_path: Path to NIFTI file

    Returns:
        True if file is valid, False otherwise
    """
    try:
        # Try to load the NIFTI file
        img = nib.load(file_path)

        # Check basic properties
        data = img.get_fdata()
        affine = img.affine

        # Validate data shape (should be 3D)
        if data.ndim != 3:
            logging.error(f"NIFTI file {file_path} has incorrect dimensions: {data.ndim}D instead of 3D")
            return False

        # Validate affine matrix (should be 4x4)
        if affine.shape != (4, 4):
            logging.error(f"NIFTI file {file_path} has invalid affine matrix shape: {affine.shape}")
            return False

        # Check for NaN values in spatial part of affine matrix
        if np.any(np.isnan(affine[:3, :3])):
            logging.error(f"NIFTI file {file_path} has NaN values in spatial affine matrix")
            return False

        logging.debug(f"NIFTI validation passed for {file_path}: shape={data.shape}, affine_shape={affine.shape}")
        return True

    except Exception as e:
        logging.error(f"Failed to validate NIFTI file {file_path}: {e}")
        return False


def decompose_and_save_timeframes(volume_4d: np.ndarray,
                                 header: Dict[str, Any],
                                 slice_origins: List[np.ndarray],
                                 patient_id: str,
                                 output_dir: str) -> None:
    """
    Decompose 4D volume into time frames and save as separate 3D NIFTI files.

    Args:
        volume_4d: 4D numpy array (time, height, width, slices)
        header: Metadata from original NRRD file
        slice_origins: List of space origins from all slices
        patient_id: Patient identifier for output filenames
        output_dir: Directory to save output files
    """
    num_timeframes = volume_4d.shape[0]
    logging.info(f"Decomposing {num_timeframes} time frames for patient {patient_id}")

    # Extract and construct proper affine matrix
    affine_matrix = extract_affine_matrix(header, slice_origins)
    logging.debug(f"Extracted affine matrix for {patient_id}: shape={affine_matrix.shape}")

    for frame_idx in range(num_timeframes):
        # Extract 3D volume for this time frame
        frame_data = volume_4d[frame_idx, :, :, :]  # Shape: (height, width, slices)

        # Ensure data is in correct orientation for NIFTI
        # frame_data is currently (height, width, slices)
        # For NIFTI, we typically want (slices, height, width) but this depends on the affine matrix
        # Keep original orientation for now and let the affine matrix handle the spatial mapping
        # frame_data shape: (height, width, slices) - this matches the spatial directions we extracted

        # Generate output filename with .nii.gz extension
        output_filename = f"{patient_id}_frame{frame_idx:02d}.nii.gz"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # Create NIFTI image with proper affine matrix
            nifti_img = nib.Nifti1Image(frame_data, affine_matrix)

            # Save NIFTI file
            nib.save(nifti_img, output_path)

            # Validate the saved file
            if validate_nifti_file(output_path):
                logging.debug(f"Saved and validated frame {frame_idx}: {output_path}")
            else:
                logging.error(f"Validation failed for frame {frame_idx}: {output_path}")
                raise ValueError(f"NIFTI validation failed for {output_path}")

        except Exception as e:
            logging.error(f"Failed to save frame {frame_idx} for patient {patient_id}: {e}")
            raise


def process_patient(patient_folder: str, patient_id: str, output_dir: str) -> None:
    """
    Process a single patient: stack slices and decompose into time frames.

    Args:
        patient_folder: Path to patient directory
        patient_id: Patient identifier
        output_dir: Output directory for processed files
    """
    logging.info(f"Processing patient: {patient_id}")

    try:
        # Step 1: Get and sort NRRD files
        nrrd_files = get_nrrd_files_for_patient(patient_folder)
        logging.info(f"Found {len(nrrd_files)} NRRD files for {patient_id}")

        # Step 2: Read and stack slices
        volume_4d, header, slice_origins = read_and_stack_slices(patient_folder, nrrd_files)

        # Step 3: Decompose and save time frames
        decompose_and_save_timeframes(volume_4d, header, slice_origins, patient_id, output_dir)

        logging.info(f"Successfully processed patient {patient_id}")

    except Exception as e:
        logging.error(f"Failed to process patient {patient_id}: {e}")
        raise


def main():
    """Main processing pipeline for NRRD medical image processing."""
    parser = argparse.ArgumentParser(description="Process CAP NRRD slices into MorphiNet NIFTI time frames")
    parser.add_argument(
        "--source_root",
        default=get_path_default("MORPHINET_CAP_NRRD_SOURCE_DIR"),
        help="Root directory containing source CAP NRRD patient folders",
    )
    parser.add_argument(
        "--output_root",
        default=get_path_default("MORPHINET_CAP_ALL_FRAME_IMAGES_TS_DIR"),
        help="Output imagesTs directory for decomposed NIFTI frames",
    )
    args = parser.parse_args()

    try:
        _require_imaging_dependencies()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Setup logging after parsing so --help remains side-effect free.
    logger = setup_logging()

    source_root = args.source_root
    output_root = args.output_root

    logger.info("Starting NRRD processing pipeline")
    logger.info(f"Source directory: {source_root}")
    logger.info(f"Output directory: {output_root}")

    # Validate paths
    if not os.path.exists(source_root):
        logger.error(f"Source directory does not exist: {source_root}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_root, exist_ok=True)

    # Get list of patient directories
    patient_folders = []
    for item in os.listdir(source_root):
        item_path = os.path.join(source_root, item)
        if os.path.isdir(item_path):
            patient_folders.append((item_path, item))

    logger.info(f"Found {len(patient_folders)} patient directories")

    # Process each patient
    success_count = 0
    error_count = 0

    for patient_folder, patient_id in patient_folders:
        try:
            process_patient(patient_folder, patient_id, output_root)
            success_count += 1

        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {e}")
            error_count += 1
            continue

    # Final report
    logger.info(f"Processing complete!")
    logger.info(f"Successfully processed: {success_count} patients")
    logger.info(f"Errors encountered: {error_count} patients")

    if error_count > 0:
        logger.warning("Some patients failed processing. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
