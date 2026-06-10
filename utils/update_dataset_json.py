#!/usr/bin/env python3
"""
Script to update dataset_task10_f0.json with all NIFTI files from Dataset010_CAP_SAX_ALL_FRAME.

This script:
1. Analyzes the folder structure and filenames in Dataset010_CAP_SAX_ALL_FRAME/imagesTs/
2. Updates the legacy JSON file to include all NIFTI files in test list only
3. Tests the _remap_abs_path function compatibility

Based on dataset_task21_f0.json.backup structure for test-only datasets.
"""

import os
import json
import glob
from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_config import get_path_default


def analyze_folder_structure(dataset_root):
    """
    Analyze the folder structure and filenames in the dataset directory.

    Args:
        dataset_root (str): Path to the dataset root directory

    Returns:
        dict: Analysis results containing file counts, naming patterns, and samples
    """
    images_dir = os.path.join(dataset_root, "imagesTs")
    labels_dir = os.path.join(dataset_root, "labelsTs")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    # Get all NIFTI files
    image_files = glob.glob(os.path.join(images_dir, "*.nii.gz"))
    label_files = glob.glob(os.path.join(labels_dir, "*.nii.gz")) if os.path.exists(labels_dir) else []

    image_files.sort()
    label_files.sort()

    print(f"\n=== FOLDER STRUCTURE ANALYSIS ===")
    print(f"Dataset root: {dataset_root}")
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    print(f"Total image files: {len(image_files)}")
    print(f"Total label files: {len(label_files)}")

    if image_files:
        # Analyze naming patterns
        sample_images = image_files[:5]
        sample_labels = label_files[:5] if label_files else []

        print(f"\nSample image filenames:")
        for i, f in enumerate(sample_images, 1):
            print(f"  {i:2d}. {os.path.basename(f)}")

        if sample_labels:
            print(f"\nSample label filenames:")
            for i, f in enumerate(sample_labels, 1):
                print(f"  {i:2d}. {os.path.basename(f)}")

        # Extract patient IDs and frame counts
        patients = set()
        frame_counts = {}
        matching_pairs = 0
        missing_labels = []

        for f in image_files:
            basename = os.path.basename(f)
            # Parse pattern: CHD{ID}-{SeqID}_frame{FrameNum}_0000.nii.gz (new format)
            try:
                if '_frame' in basename and '_0000.nii.gz' in basename:
                    # New format: CHD0001501-20070000_frame00_0000.nii.gz
                    parts = basename.split('_frame')
                    patient_seq = parts[0]  # e.g., CHD0001501-20070000
                    frame_part = parts[1].replace('_0000.nii.gz', '')  # e.g., 00

                    # Check for matching label file
                    expected_label = f"{patient_seq}_frame{frame_part}.nii.gz"
                    label_path = os.path.join(labels_dir, expected_label)

                    if os.path.exists(label_path):
                        matching_pairs += 1
                    else:
                        missing_labels.append(expected_label)

                elif '_frame' in basename and '.nii.gz' in basename and '_0000' not in basename:
                    # Old format: CHD0001501-20070000_frame00.nii.gz
                    parts = basename.split('_frame')
                    patient_seq = parts[0]
                    frame_part = parts[1].replace('.nii.gz', '')
                else:
                    print(f"Warning: Unrecognized filename pattern: {basename}")
                    continue

                # Extract patient ID (everything before last hyphen)
                patient_id = patient_seq.rsplit('-', 1)[0]  # e.g., CHD0001501
                patients.add(patient_id)

                if patient_seq not in frame_counts:
                    frame_counts[patient_seq] = []
                frame_counts[patient_seq].append(int(frame_part))

            except Exception as e:
                print(f"Warning: Could not parse filename: {basename} - {e}")

        print(f"\nUnique patients: {len(patients)}")
        print(f"Patient sequences: {len(frame_counts)}")
        print(f"Image-label matching pairs: {matching_pairs}/{len(image_files)}")

        if missing_labels:
            print(f"Missing labels: {len(missing_labels)} (first 3: {missing_labels[:3]})")

        # Show frame count distribution
        frame_count_dist = {}
        for seq, frames in frame_counts.items():
            count = len(frames)
            if count not in frame_count_dist:
                frame_count_dist[count] = 0
            frame_count_dist[count] += 1

        print(f"\nFrame count distribution:")
        for count, sequences in sorted(frame_count_dist.items()):
            print(f"  {count} frames: {sequences} sequences")

    return {
        'total_image_files': len(image_files),
        'total_label_files': len(label_files),
        'image_files': image_files,
        'label_files': label_files,
        'patients': sorted(patients) if image_files else [],
        'frame_counts': frame_counts,
        'matching_pairs': matching_pairs,
        'has_labels': len(label_files) > 0
    }


def create_updated_json(analysis_results, output_path, reference_json_path=None):
    """
    Create updated JSON file with all NIFTI files and matching labels.

    Args:
        analysis_results (dict): Results from analyze_folder_structure()
        output_path (str): Path to save the updated JSON file
        reference_json_path (str): Path to reference JSON for structure template

    Returns:
        dict: The created JSON structure
    """
    print(f"\n=== CREATING UPDATED JSON ===")

    # Load reference structure if provided
    base_structure = {
        "name": "Dataset010_CAP_SAX_ALL_FRAME",
        "description": "CAP cine MR SAX image data - full frame dataset with labels",
        "tensorImageSize": "3D",
        "reference": "https://cardiacatlas.org/challenges/segmentation/",
        "licence": "CC-BY-SA 4.0",
        "release": "n/a",
        "modality": {
            "0": "MR"
        },
        "labels": {
            "0": "background",
            "1": "lv",
            "2": "lv-myo",
            "3": "rv",
            "4": "rv-myo"
        },
        "training": [],
        "numTraining": 0,
        "test": [],
        "numTest": 0
    }

    if reference_json_path and os.path.exists(reference_json_path):
        print(f"Loading reference structure from: {reference_json_path}")
        with open(reference_json_path, 'r') as f:
            reference = json.load(f)

        # Update base structure with reference metadata
        base_structure.update({
            "description": "CAP cine MR SAX image data - full frame dataset with labels",
            "name": "Dataset010_CAP_SAX_ALL_FRAME",
            "tensorImageSize": reference.get("tensorImageSize", "3D"),
            "reference": reference.get("reference", ""),
            "licence": reference.get("licence", ""),
            "release": reference.get("release", "n/a"),
            "modality": reference.get("modality", {"0": "MR"}),
            "labels": reference.get("labels", {
                "0": "background", "1": "lv", "2": "lv-myo", "3": "rv", "4": "rv-myo"
            })
        })

    # Create test entries for all NIFTI files
    test_entries = []
    for image_file in analysis_results['image_files']:
        image_basename = os.path.basename(image_file)

        # Create image path relative to dataset root
        image_path = f"./imagesTs/{image_basename}"

        # Generate corresponding label path
        # Convert CHD0001501-20070000_frame00_0000.nii.gz -> CHD0001501-20070000_frame00.nii.gz
        if '_0000.nii.gz' in image_basename:
            label_basename = image_basename.replace('_0000.nii.gz', '.nii.gz')
        else:
            # Fallback for any different naming patterns
            label_basename = image_basename

        label_path = f"./labelsTs/{label_basename}"

        test_entries.append({
            "image": image_path,
            "label": label_path
        })

    # Update the JSON structure
    base_structure["test"] = test_entries
    base_structure["numTest"] = len(test_entries)

    print(f"Created {len(test_entries)} test entries")
    print(f"Image files: {analysis_results['total_image_files']}")
    print(f"Label files: {analysis_results['total_label_files']}")
    print(f"Matching pairs: {analysis_results['matching_pairs']}")

    # Save the updated JSON
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(base_structure, f, indent=2)

    print(f"Updated JSON saved to: {output_path}")

    return base_structure


def test_remap_abs_path_compatibility(json_data, mr_data_dir):
    """
    Test if _remap_abs_path method can identify the NIFTI files correctly.

    Args:
        json_data (dict): The JSON data structure
        mr_data_dir (str): The MR data directory path

    Returns:
        dict: Test results
    """
    print(f"\n=== TESTING _REMAP_ABS_PATH COMPATIBILITY ===")

    # Simulate what _remap_abs_path does for MR modal
    test_data_list = json_data.get('test', [])
    section = "Ts"  # For test data

    print(f"Testing with mr_data_dir: {mr_data_dir}")
    print(f"Number of test entries: {len(test_data_list)}")

    remapped = []
    missing_images = []
    missing_labels = []

    for i, d in enumerate(test_data_list[:5]):  # Test first 5 entries
        img_name = os.path.basename(d["image"])  # e.g., CHD0001501-20070000_frame00_0000.nii.gz
        label_name = os.path.basename(d["label"])  # e.g., CHD0001501-20070000_frame00.nii.gz

        # Generate case ID (strip _0000 suffix for new format)
        if '_0000.nii.gz' in img_name:
            stem = img_name.replace('_0000.nii.gz', '')  # CHD0001501-20070000_frame00
        else:
            stem = os.path.splitext(os.path.splitext(img_name)[0])[0]  # Fallback

        # This mimics what _remap_abs_path does
        mr_image_path = os.path.join(mr_data_dir, f"images{section}", img_name)
        mr_label_path = os.path.join(mr_data_dir, f"labels{section}", label_name)

        remapped_entry = {
            "mr_image": mr_image_path,
            "mr_label": mr_label_path,
            "mr_case_id": stem,
        }
        remapped.append(remapped_entry)

        # Check if files actually exist
        image_exists = os.path.exists(mr_image_path)
        label_exists = os.path.exists(mr_label_path)

        if not image_exists:
            missing_images.append(mr_image_path)
        if not label_exists:
            missing_labels.append(mr_label_path)

        status = "✓ BOTH EXIST" if (image_exists and label_exists) else \
                 "⚠ IMAGE ONLY" if image_exists else \
                 "⚠ LABEL ONLY" if label_exists else "✗ BOTH MISSING"

        print(f"  Entry {i+1}: {status}")
        print(f"    Image: {mr_image_path} {'✓' if image_exists else '✗'}")
        print(f"    Label: {mr_label_path} {'✓' if label_exists else '✗'}")
        print(f"    Case ID: {stem}")
        print()

    # Summary
    total_test_entries = len(test_data_list)
    print(f"=== COMPATIBILITY TEST SUMMARY ===")
    print(f"Total test entries: {total_test_entries}")
    print(f"Sample entries tested: {min(5, total_test_entries)}")
    print(f"Missing image files (in sample): {len(missing_images)}")
    print(f"Missing label files (in sample): {len(missing_labels)}")

    if missing_images:
        print(f"\nMissing image files:")
        for f in missing_images[:3]:
            print(f"  - {f}")
        if len(missing_images) > 3:
            print(f"  ... and {len(missing_images) - 3} more")

    if missing_labels:
        print(f"\nMissing label files:")
        for f in missing_labels[:3]:
            print(f"  - {f}")
        if len(missing_labels) > 3:
            print(f"  ... and {len(missing_labels) - 3} more")

    # Test the expected vs actual path structure
    expected_images_dir = os.path.join(mr_data_dir, f"images{section}")
    expected_labels_dir = os.path.join(mr_data_dir, f"labels{section}")
    actual_images_dir = os.path.join(mr_data_dir, "imagesTs")
    actual_labels_dir = os.path.join(mr_data_dir, "labelsTs")

    print(f"\nPath Structure Analysis:")
    print(f"Expected images dir: {expected_images_dir}")
    print(f"Actual images dir:   {actual_images_dir}")
    print(f"Expected labels dir: {expected_labels_dir}")
    print(f"Actual labels dir:   {actual_labels_dir}")
    print(f"Images path match: {expected_images_dir == actual_images_dir}")
    print(f"Labels path match: {expected_labels_dir == actual_labels_dir}")

    compatibility = {
        'total_entries': total_test_entries,
        'sample_tested': min(5, total_test_entries),
        'missing_images': missing_images,
        'missing_labels': missing_labels,
        'path_structure_match': (expected_images_dir == actual_images_dir and
                                expected_labels_dir == actual_labels_dir),
        'remapped_sample': remapped[:3]  # Store first 3 for inspection
    }

    return compatibility


def main():
    """Main function to orchestrate the JSON update process."""
    parser = argparse.ArgumentParser(description="Update MorphiNet dataset JSON for full-frame NIFTI data")
    parser.add_argument("--dataset_root", type=str,
                       default=get_path_default("MORPHINET_CAP_ALL_FRAME_DIR"),
                       help="Path to dataset root directory (containing imagesTs and labelsTs)")
    parser.add_argument("--output_json", type=str, default="dataset/dataset_task10_f0_updated.json",
                       help="Output path for updated JSON file")
    parser.add_argument("--reference_json", type=str, default="dataset/dataset_task21_f0.json.backup",
                       help="Reference JSON file for structure template")
    parser.add_argument("--mr_data_dir", type=str,
                       default=get_path_default("MORPHINET_CAP_ALL_FRAME_DIR"),
                       help="MR data directory for compatibility testing")

    args = parser.parse_args()

    print("="*80)
    print("MORPHINET DATASET JSON UPDATE SCRIPT")
    print("="*80)

    try:
        # Step 1: Analyze folder structure
        print(f"Step 1: Analyzing folder structure...")
        analysis_results = analyze_folder_structure(args.dataset_root)

        # Step 2: Create updated JSON
        print(f"\nStep 2: Creating updated JSON...")
        json_data = create_updated_json(
            analysis_results,
            args.output_json,
            args.reference_json if os.path.exists(args.reference_json) else None
        )

        # Step 3: Test _remap_abs_path compatibility
        print(f"\nStep 3: Testing _remap_abs_path compatibility...")
        compatibility = test_remap_abs_path_compatibility(json_data, args.mr_data_dir)

        # Final summary
        print(f"\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"✓ Analyzed {analysis_results['total_image_files']} image files and {analysis_results['total_label_files']} label files")
        print(f"✓ Created updated JSON with {json_data['numTest']} test entries")
        print(f"✓ Saved to: {args.output_json}")

        if compatibility['path_structure_match']:
            print(f"✓ Path structure is compatible with _remap_abs_path")
        else:
            print(f"⚠ Path structure requires attention for _remap_abs_path compatibility")

        print(f"\nNext steps:")
        print(f"1. Review the generated JSON file: {args.output_json}")
        print(f"2. Test with MorphiNet data loader")
        print(f"3. Update main dataset JSON if satisfied with results")

        if not compatibility['path_structure_match']:
            print(f"\nNote: The _remap_abs_path function expects 'imagesTs' but current")
            print(f"implementation might expect 'imagesTs'. Verify data loader compatibility.")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
