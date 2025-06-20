import os
import nibabel as nib
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from monai.transforms import LoadImaged

def _find_corresponding_image_file(label_filename, label_name_core, images_tr_path, is_label_nrrd_seg):
    """
    Finds a corresponding image NIfTI or NRRD file in images_tr_path for a given label_name_core.
    If is_label_nrrd_seg is True, it looks for label_name_core + '.seq.nrrd'.
    Otherwise, uses a broader search for NIfTI or generic NRRD files.
    Sorts candidates and picks the first one if multiple are found (for non-specific NRRD/NIfTI search).
    Prints warnings if no image or multiple images are found (for non-specific NRRD/NIfTI search).
    Returns the image filename (str) or None.
    """
    if label_name_core is None: 
        print(f"Warning: label_name_core is None for {label_filename}. Cannot find corresponding image.")
        return None

    candidate_image_files = []
    preferred_nrrd_image_suffix = ".seq.nrrd" # Used if is_label_nrrd_seg is true

    for f_name in os.listdir(images_tr_path):
        current_image_file_base = None
        # Determine base name based on extension
        if f_name.endswith(".nii.gz"):
            current_image_file_base = f_name[:-len(".nii.gz")]
        elif f_name.endswith(".nii"):
            current_image_file_base = f_name[:-len(".nii")]
        elif f_name.endswith(".nrrd"): # Catches .nrrd and .seq.nrrd
            current_image_file_base = f_name[:-len(".nrrd")]
        else:
            continue # Not a supported file type

        # If a base name was extracted and label_name_core is part of it
        if current_image_file_base is not None and label_name_core in current_image_file_base:
            candidate_image_files.append(f_name)

    if not candidate_image_files:
        # Universal message as search is now always substring based
        print(f"Warning: No corresponding image file found for label '{label_filename}' (using base '{label_name_core}' with substring search) in {images_tr_path}")
        return None
    
    candidate_image_files.sort() 
    
    image_filename = candidate_image_files[0] # Default to the first sorted candidate

    # If the label was a .seg.nrrd, check if a preferred .seq.nrrd image exists among candidates
    # and prioritize it if found.
    if is_label_nrrd_seg:
        preferred_match_name = label_name_core + preferred_nrrd_image_suffix
        # Iterate through candidates to find the exact preferred match
        for candidate in candidate_image_files:
            if candidate == preferred_match_name:
                image_filename = candidate # Prioritize this match
                break 
        # If preferred_match_name was not found, image_filename remains candidate_image_files[0]
    
    # Handle warnings for multiple candidates
    if len(candidate_image_files) > 1:
        is_preferred_chosen = False
        if is_label_nrrd_seg:
            # Check if the currently selected image_filename is the preferred one
            if image_filename == (label_name_core + preferred_nrrd_image_suffix):
                is_preferred_chosen = True
        
        if is_preferred_chosen:
            other_candidates = [c for c in candidate_image_files if c != image_filename]
            if other_candidates: # This should be true if len(candidate_image_files) > 1 and preferred was chosen
                print(f"Note: Preferred image '{image_filename}' selected for label '{label_filename}'. Other potential matches found based on '{label_name_core}': {other_candidates}.")
        else:
            # This covers cases where:
            # 1. Label is not .seg.nrrd (so no "preferred" selection logic was applied that would set is_preferred_chosen to True)
            # 2. Label is .seg.nrrd, but the specific .seq.nrrd was not among candidates or not selected (so image_filename is candidate_image_files[0])
            print(f"Warning: Multiple candidate image files found for label '{label_filename}' (matching base '{label_name_core}'): {candidate_image_files}. Selected '{image_filename}' (first after sorting, as preferred match was not applicable or not found).")

    return image_filename

def check_affine_consistency(root_dir):
    images_tr_path = os.path.join(root_dir, 'imagesTr')
    labels_tr_path = os.path.join(root_dir, 'labelsTr')

    if not os.path.isdir(images_tr_path):
        print(f"Error: Image directory not found at {images_tr_path}")
        return
    if not os.path.isdir(labels_tr_path):
        print(f"Error: Label directory not found at {labels_tr_path}")
        return

    print(f"Checking affine consistency for NIfTI/NRRD files in {root_dir}\n")

    for label_filename in os.listdir(labels_tr_path):
        label_name_core = None
        is_label_nrrd_seg = False

        if label_filename.endswith(".seg.nrrd"):
            label_name_core = label_filename[:-len(".seg.nrrd")]
            is_label_nrrd_seg = True
        elif label_filename.endswith(".nrrd"):
            label_name_core = label_filename[:-len(".nrrd")]
            # is_label_nrrd_seg remains False
        elif label_filename.endswith(".nii.gz"):
            label_name_core = label_filename[:-len(".nii.gz")]
            # is_label_nrrd_seg remains False
        elif label_filename.endswith(".nii"):
            label_name_core = label_filename[:-len(".nii")]
            # is_label_nrrd_seg remains False
        else:
            continue # Skip non-supported files
        
        if label_name_core is None or not label_name_core: # Ensure core is not empty
            print(f"Warning: Could not determine a valid base name for label file {label_filename}. Skipping.")
            continue

        label_filepath = os.path.join(labels_tr_path, label_filename)
        image_filename = _find_corresponding_image_file(label_filename, label_name_core, images_tr_path, is_label_nrrd_seg)

        if image_filename is None:
            continue
            
        image_filepath = os.path.join(images_tr_path, image_filename)

        try:
            label_nii = nib.load(label_filepath)
            image_nii = nib.load(image_filepath)

            label_affine = label_nii.affine
            image_affine = image_nii.affine

            if np.allclose(label_affine, image_affine):
                pass # Do nothing if consistent
            else:
                print(f"FAIL: Affine matrices for {label_filename} and {image_filename} are NOT consistent.")
                print(f"  Label Affine:\n{label_affine}")
                print(f"  Image Affine:\n{image_affine}")
        
        except Exception as e:
            print(f"Error processing file pair ({label_filename}, {image_filename}): {e}")

def generate_overlay_images(root_dir, slice_axis=-1, slice_position_ratio=0.5, overlay_dir_name="overlay_images", nrrd_is_4d_timeseries=True):
    """
    Generates and saves overlay images for image-label pairs from NIfTI or NRRD files.
    Uses MONAI LoadImaged for reading files.
    If nrrd_is_4d_timeseries is True and an NRRD file is 4D, the first time point is used.

    For each label file in the 'labelsTr' directory, it finds a corresponding image
    file in 'imagesTr'. It then loads both, extracts a 2D slice (defaulting to the
    middle slice of the last axis), normalizes the image slice, and overlays the
    label slice with distinct colors for different label values (0 is background).
    The resulting overlay is saved as a PNG in a subdirectory (default 'overlay_images')
    under the root_dir.
    """
    images_tr_path = os.path.join(root_dir, 'imagesTr')
    labels_tr_path = os.path.join(root_dir, 'labelsTr')
    output_path = os.path.join(root_dir, overlay_dir_name)

    if not os.path.isdir(images_tr_path):
        print(f"Error: Image directory not found at {images_tr_path}")
        return
    if not os.path.isdir(labels_tr_path):
        print(f"Error: Label directory not found at {labels_tr_path}")
        return

    os.makedirs(output_path, exist_ok=True)
    print(f"Generating overlay images in {output_path}\n")

    loader = LoadImaged(keys=['image', 'label'], ensure_channel_first=False, image_only=False, allow_missing_keys=False, reader=None)

    for label_filename in os.listdir(labels_tr_path):
        label_name_core = None
        is_label_nrrd_seg = False
        is_label_generic_nrrd = False # To distinguish between .seg.nrrd and other .nrrd for image data handling

        if label_filename.endswith(".seg.nrrd"):
            label_name_core = label_filename[:-len(".seg.nrrd")]
            is_label_nrrd_seg = True
        elif label_filename.endswith(".nrrd"):
            label_name_core = label_filename[:-len(".nrrd")]
            is_label_generic_nrrd = True # It's a .nrrd but not .seg.nrrd
            # is_label_nrrd_seg remains False
        elif label_filename.endswith(".nii.gz"):
            label_name_core = label_filename[:-len(".nii.gz")]
        elif label_filename.endswith(".nii"):
            label_name_core = label_filename[:-len(".nii")]
        else:
            continue # Skip non-supported files
        
        if label_name_core is None or not label_name_core:
            print(f"Warning: Could not determine a valid base name for label file {label_filename}. Skipping overlay generation.")
            continue
        
        label_filepath = os.path.join(labels_tr_path, label_filename)
        # For _find_corresponding_image_file, the is_label_nrrd_seg flag is key for the .seg.nrrd -> .seq.nrrd logic
        image_filename = _find_corresponding_image_file(label_filename, label_name_core, images_tr_path, is_label_nrrd_seg)

        if image_filename is None:
            print(f"Skipping overlay generation for {label_filename} as no corresponding image was confirmed (details from _find_corresponding_image_file).")
            continue
            
        image_filepath = os.path.join(images_tr_path, image_filename)
        
        # Determine if the specific image file found is NRRD for 4D handling later
        # This is independent of how the label was named (e.g. label could be .nii, image .nrrd)
        image_is_nrrd = image_filename.endswith(".nrrd") 
        # If the label was .seg.nrrd, then image_filename must be .seq.nrrd for a match
        # if is_label_nrrd_seg and not image_filename.endswith(".seq.nrrd"):
        #     print(f"Error: Label was {label_filename} but matched image {image_filename} is not .seq.nrrd. Skipping.")
        #     continue # This should be caught by _find_corresponding_image_file not finding a match

        try:
            data_dict = {'image': image_filepath, 'label': label_filepath}
            loaded_data = loader(data_dict)
            image_data = loaded_data['image'] 
            label_data = loaded_data['label']

            # Handle 4D NRRD time-series: select the first time point
            # This applies if the *loaded image* is NRRD and 4D, or *loaded label* is NRRD and 4D
            if image_is_nrrd and nrrd_is_4d_timeseries and image_data.ndim == 4:
                print(f"Info: Image {image_filename} is 4D NRRD, selecting first time point.")
                image_data = image_data[0, ...]
            
            # Check if the original label file was NRRD (either .seg.nrrd or generic .nrrd)
            original_label_is_nrrd = is_label_nrrd_seg or is_label_generic_nrrd
            if original_label_is_nrrd and nrrd_is_4d_timeseries and label_data.ndim == 4:
                print(f"Info: Label {label_filename} is 4D NRRD, selecting first time point.")
                label_data = label_data[0, ...]

            # 2) Decide a slice index and slicing the data array
            if not (0 <= slice_position_ratio <= 1):
                print(f"Warning: slice_position_ratio ({slice_position_ratio}) is out of [0,1] range. Clamping. For {label_filename}")
                slice_position_ratio = max(0, min(1, slice_position_ratio))

            actual_slice_axis = slice_axis
            if actual_slice_axis < 0: 
                actual_slice_axis += len(image_data.shape)
            
            if not (0 <= actual_slice_axis < len(image_data.shape)):
                print(f"Error: slice_axis {slice_axis} (resolved to {actual_slice_axis}) is out of bounds for image data with shape {image_data.shape}. Skipping {label_filename}.")
                continue
            
            num_slices_in_axis = image_data.shape[actual_slice_axis]
            if num_slices_in_axis == 0:
                print(f"Warning: Selected axis {actual_slice_axis} has 0 slices for {label_filename}. Skipping overlay.")
                continue
                
            slice_idx = int(num_slices_in_axis * slice_position_ratio)
            slice_idx = max(0, min(slice_idx, num_slices_in_axis - 1)) 

            slicer = [slice(None)] * len(image_data.shape)
            slicer[actual_slice_axis] = slice_idx
            
            image_slice_raw = image_data[tuple(slicer)]
            label_slice_raw = label_data[tuple(slicer)]

            # The slicing operation itself results in image_slice_raw and label_slice_raw.
            # If the original data was 3D, these will be 2D.
            # The ndim check below will catch cases where they are not 2D (e.g. from 4D input).
            image_slice = image_slice_raw
            label_slice = label_slice_raw

            if image_slice.ndim != 2 or label_slice.ndim != 2:
                print(f"Warning: Sliced data for {label_filename} (slice {slice_idx} along resolved axis {actual_slice_axis}) is not 2D after slicing. Image shape: {image_slice.shape}, Label shape: {label_slice.shape}. Skipping overlay.")
                continue

            # 3) Overlay the label on image
            img_min = np.min(image_slice)
            img_max = np.max(image_slice)
            if img_max == img_min: 
                image_slice_normalized = np.zeros_like(image_slice, dtype=np.uint8)
            else:
                image_slice_normalized = ((image_slice - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)

            fig, ax = plt.subplots(figsize=(10, 10)) 
            
            rotated_image_slice = np.rot90(image_slice_normalized)
            ax.imshow(rotated_image_slice, cmap='gray', interpolation='none')

            label_colors = ['red', 'green', 'blue', 'yellow'] 
            masked_label_slice = np.ma.masked_where(label_slice == 0, label_slice)
            
            custom_cmap = mcolors.ListedColormap(label_colors)
            bounds = np.arange(0.5, len(label_colors) + 1.5, 1) 
            norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

            ax.imshow(np.rot90(masked_label_slice), cmap=custom_cmap, norm=norm, alpha=0.5, interpolation='nearest')
            
            # Add title with label filename and displayed image size
            displayed_height, displayed_width = rotated_image_slice.shape[0], rotated_image_slice.shape[1]
            title_str = f"{label_filename}\nSize: {displayed_height}x{displayed_width} (HxW)"
            ax.set_title(title_str, fontsize=10)
            
            ax.axis('off')

            output_filename_base = label_name_core 
            output_image_path = os.path.join(output_path, f"{output_filename_base}_slice{slice_idx}_axis{slice_axis}_overlay.png")
            
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig) 
            print(f"Saved overlay for {label_filename} (slice {slice_idx} on original axis {slice_axis}) to {output_image_path}")

        except Exception as e:
            print(f"Error generating overlay for pair ({label_filename}, {image_filename}): {e}")
            if 'fig' in locals() and plt.fignum_exists(fig.number):
                 plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check affine consistency or generate overlay images for NIfTI files.")
    parser.add_argument("root_dir", type=str, help="Root directory containing 'imagesTr' and 'labelsTr' folders.")
    parser.add_argument("--action", type=str, required=True, choices=['affine', 'overlay'], 
                        help="Action to perform: 'affine' for affine consistency check, 'overlay' for generating overlay images.")
    # Arguments specific to overlay generation, with defaults
    parser.add_argument("--slice_axis", type=int, default=-1, 
                        help="Axis along which to slice the NIfTI volume for overlay (default: -1, the last axis).")
    parser.add_argument("--slice_position_ratio", type=float, default=0.5, 
                        help="Relative position of the slice (0.0 to 1.0) along the chosen axis (default: 0.5, middle slice).")
    parser.add_argument("--overlay_dir_name", type=str, default="overlay_images", 
                        help="Name of the directory to save overlay images (default: 'overlay_images').")
    parser.add_argument("--nrrd_is_4d_timeseries", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="If processing NRRD files, treat them as 4D time-series and use the first time point (default: True).")

    args = parser.parse_args()
    
    if args.action == 'affine':
        check_affine_consistency(args.root_dir)
    elif args.action == 'overlay':
        generate_overlay_images(args.root_dir, 
                                slice_axis=args.slice_axis, 
                                slice_position_ratio=args.slice_position_ratio, 
                                overlay_dir_name=args.overlay_dir_name,
                                nrrd_is_4d_timeseries=args.nrrd_is_4d_timeseries)
    else:
        # This case should not be reached due to 'choices' in add_argument, but good for robustness
        print(f"Error: Unknown action '{args.action}'. Choose either 'affine' or 'overlay'.") 