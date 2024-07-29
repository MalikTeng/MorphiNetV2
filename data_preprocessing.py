import os, shutil
import nrrd
import nibabel as nib
import numpy as np
from natsort import natsorted
from scipy.interpolate import griddata
import tqdm


def main(source_nrrd_dir, output_nrrd_dir):

    if os.path.exists(output_nrrd_dir): # comment out this line if you want to overwrite the existing files
        shutil.rmtree(output_nrrd_dir)
    os.makedirs(output_nrrd_dir)

    # Iterate over the subfolders (PatientIDs) in the root directory
    for folder_name in tqdm.tqdm(os.listdir(source_nrrd_dir)):
        
        folder_path = os.path.join(source_nrrd_dir, folder_name)

        # Check if the subfolder is a directory
        if os.path.isdir(folder_path):

            # Get a list of all the nrrd files in the subfolder
            nrrd_files = natsorted([file for file in os.listdir(folder_path) if file.endswith('.seq.nrrd')])

            # Correct the meta info, rename the files, and save them in the output directory
            prefix = folder_path.split('/')[-1]
            nrrd_data = [nrrd.read(os.path.join(folder_path, nrrd_file)) for nrrd_file in nrrd_files]
            nrrd_hdr = [data[1] for data in nrrd_data]
            nrrd_data = [data[0] for data in nrrd_data]

            # Check all the SAX slices for consistency and save the corrected nifti files
            affine_check(nrrd_data, nrrd_hdr, output_nrrd_dir, prefix)


def apply_affine_matrix(label, affine_matrix):
    """
    apply affine matrix to the label.
    
    params:
        label: a 3D label.
        affine_matrix: affine matrix of the label.

    return:
        coord: the 3D coordinates of the label after applying the affine matrix.
    """
    x, y, z = np.meshgrid(np.arange(label.shape[0]), np.arange(label.shape[1]), np.arange(label.shape[2]), indexing='ij')
    coord = nib.affines.apply_affine(
        affine_matrix, np.concatenate([x, y, z], axis=-1)
        )
    return np.concatenate([coord, label], axis=-1)


def slice_fixer(current_slice_data, first_slice_affine, current_slice_affine, method="linear"):

    h, w = current_slice_data.shape[1:3]

    # use the first affine matrix to reverse the coordinates
    reverse_affine = np.linalg.inv(first_slice_affine)

    # apply the affine matrix to the current slice data
    for f, frame_data in enumerate(current_slice_data):
        point_cloud = apply_affine_matrix(frame_data, reverse_affine @ current_slice_affine)

        # renew the current slice data
        x, y, _, value = np.split(point_cloud, 4, axis=-1)

        # interpolate the point_cloud to avoid aliasing in outcomes
        fixed_frame_data = griddata((x.flatten(), y.flatten()), value.flatten(),
                            tuple(np.meshgrid(np.arange(h), np.arange(w), indexing='ij')),
                            method=method, fill_value=value.min())
        
        current_slice_data[f, ..., -1] = fixed_frame_data

    return current_slice_data


def affine_check(nrrd_data, nrrd_hdr, output_dir, filename):
    """
    double-check to see if SAX slice spacing field is correct. After correction, a few slices with an abnormal slice spacing should be removed, and the affine matrix should be chosen for output nifti file.
    """
    save_file_path = os.path.join(output_dir, f"{filename} - MR SAX_0000.seq.nrrd")
    if os.path.exists(save_file_path):
        os.remove(save_file_path)

    if len(nrrd_data) > 1:
        # check the size of all slices are the same
        if not all([all(nrrd_hdr[0]["sizes"] == hdr["sizes"]) for hdr in nrrd_hdr]):
            slice_sizes = [all(nrrd_hdr[0]["sizes"] == hdr["sizes"]) for hdr in nrrd_hdr]
            status = np.unique(slice_sizes, return_counts=True)
            slice_idx = np.where(slice_sizes == status[0][np.argmin(status[1])])[0].item()
            # drop the slice with different size
            nrrd_data.pop(slice_idx)
            nrrd_hdr.pop(slice_idx)

        # get the sort sequence of the slices
        slice_normal = nrrd_hdr[0]["space directions"][-1]
        if abs(np.linalg.norm(slice_normal) - 1) > 1e-3:
            slice_normal = slice_normal / np.linalg.norm(slice_normal)
            
        slice_position = np.array([np.dot(slice_normal, hdr["space origin"]) for hdr in nrrd_hdr])  # sort out the space position (D) of this slice using Ax + By + Cz = D, where (A, B, C) is the slice normal and (x, y, z) is the space origin.
        slice_sequence = np.argsort(slice_position)[::-1]   # from the base to apex, which is commonly the scanning order (in the direction of the slice normal)
        if np.dot(slice_normal, nrrd_hdr[slice_sequence[1]]["space origin"] - nrrd_hdr[slice_sequence[0]]["space origin"]) < 0: # if not, need to change the space direction from LPS (DICOM) to RAS (NRRD)
            for i in range(len(nrrd_hdr)):
                # change the space directions (affine)
                nrrd_hdr[i]["space directions"] = nrrd_hdr[i]["space directions"][[0, 2, 1, 3]]
                nrrd_hdr[i]["space directions"][-1] *= -1
                # nrrd_hdr[i]["space directions"][-1] = np.cross(nrrd_hdr[i]["space directions"][1], nrrd_hdr[i]["space directions"][2])
                # nrrd_hdr[i]["space directions"][-1] /= np.linalg.norm(nrrd_hdr[i]["space directions"][-1])
                # change the sizes (pixel array size)
                nrrd_hdr[i]["sizes"] = nrrd_hdr[i]["sizes"][[0, 2, 1, 3]]
            slice_normal = nrrd_hdr[0]["space directions"][-1]  # update the slice normal
            # change the slice sequence
            nrrd_data = [i.swapaxes(1, 2) for i in nrrd_data]
            print(f"Space directions of ({filename}) is changed from LPS to RAS.")
        slice_position = slice_position[slice_sequence].tolist()
        slice_sequence = slice_sequence.tolist()
        
        # find the most common slice spacing and drop the abnormal slices
        all_stacks = dict()                         # sometimes scanning been made with hops between consecutive slices (| | 'hop' | | | 'hop' | |, inconsistent spacing between two stack of slices with the same spacing), which should be considered as a stack.
        all_stacks_sequence = dict()
        cnt = 0
        accept_slices = [slice_position.pop(0)]     # the position of the selected slices
        accept_sequence = [slice_sequence.pop(0)]   # the index of the selected slices
        spacing = None
        while len(slice_sequence) > 0:  # loop through all slices
            position = slice_position.pop(0)
            sequence = slice_sequence.pop(0)
            if spacing is None:         # calculate the slice spacing using the first two slices
                accept_slices.append(position)
                accept_sequence.append(sequence)
                spacing = np.diff(accept_slices)
            else:                       # check the slice spacing of the rest slices
                if abs(position - accept_slices[-1]) < 1e-3:                    # drop the later slice if the slice spacing is too small (scanned multiple times on the same position)
                    continue
                elif abs(position - accept_slices[-1] - spacing[-1]) > 1e-3:    # drop the later slice if the slice spacing is not consistent
                    print(f"Slice spacing of ({filename}) is not consistent.")
                    all_stacks[cnt] = accept_slices
                    all_stacks_sequence[cnt] = accept_sequence
                    accept_slices = [position]
                    accept_sequence = [sequence]
                    spacing = None
                    cnt += 1
                else:                                                           # accept the slice if the slice spacing is consistent
                    accept_slices.append(position)
                    accept_sequence.append(sequence)
        all_stacks[cnt] = accept_slices
        all_stacks_sequence[cnt] = accept_sequence
        accept_slices = max(all_stacks.values(), key=len)                       # select the stack with the most slices
        accept_sequence = max(all_stacks_sequence.values(), key=len)
        spacing = np.mean(np.diff(accept_slices, axis=0))
        del all_stacks_sequence, accept_slices
        nrrd_data = [nrrd_data[i] for i in accept_sequence]
        nrrd_hdr = [nrrd_hdr[i] for i in accept_sequence]
        
        # update the slice spacing and size of data array in the affine matrix, in RAS+ coordinate system
        distance_vector = abs(spacing) * slice_normal
        nrrd_hdr[0]['sizes'][-1] = len(nrrd_data)
        nrrd_hdr[0]["space directions"][-1] = distance_vector

        # loopthrough all sax slices to check the space origin and space directions
        for i in range(1, len(nrrd_hdr)):
            if not np.allclose(nrrd_hdr[i]["space origin"], nrrd_hdr[0]["space origin"] + i * distance_vector):
                first_slice_affine = np.vstack([
                    np.hstack([nrrd_hdr[0]["space directions"][1:].T, nrrd_hdr[0]["space origin"][:, None]]), 
                    [0, 0, 0, 1]])
                current_slice_affine = np.vstack([
                    np.hstack([nrrd_hdr[i]["space directions"][1:].T, nrrd_hdr[i]["space origin"][:, None]]), 
                    [0, 0, 0, 1]])
                # correction is needed for some slice, resulted from a rotation for better field of view when scanning
                nrrd_data[i] = slice_fixer(nrrd_data[i], first_slice_affine, current_slice_affine, method="linear")
                print(f"Space origin of ({filename}) on SAX slice ({i+1}/{len(nrrd_data)}) is corrected.")

        # concatenate the nrrd data and update the shape of the nrrd data
        nrrd_data = np.concatenate(nrrd_data, axis=-1)
        assert nrrd_hdr[0]["sizes"][-1] == nrrd_data.shape[-1], "The number of slices in the nrrd data is not consistent with the header."

    elif len(nrrd_data) == 1:
        # transform from LPS to RAS
        nrrd_hdr[0]["space directions"] = nrrd_hdr[0]["space directions"][[0, 2, 1, 3]]
        nrrd_hdr[0]["space directions"][-1] *= -1
        # nrrd_hdr[0]["space directions"][-1] = np.cross(nrrd_hdr[0]["space directions"][1], nrrd_hdr[0]["space directions"][2])
        # nrrd_hdr[0]["space directions"][-1] /= np.linalg.norm(nrrd_hdr[0]["space directions"][-1])
        nrrd_hdr[0]["sizes"] = nrrd_hdr[0]["sizes"][[0, 2, 1, 3]]
        nrrd_data = np.flip(nrrd_data[0].swapaxes(1, 2), axis=-1)

    # save the corrected nrrd file
    nrrd.write(save_file_path, nrrd_data, header=nrrd_hdr[0])


if __name__ == "__main__":
    
    source_nrrd_dir = "/mnt/data/Experiment/Data/raw_data/Dataset010_CAP_SAX_NRRD/"    # Path to the root directory of the dataset
    
    # 1. Load your DICOM database into 3D Slicer and convert it to NRRD format volume images (operate in the DICOM module, 3D Slicer and convert only SAX series, toggling 'advance' if encountering any issues),
    #    and save the NRRD files in the above directory. This will create a structure of directory as follows:
    #    Abdul
    #    ├── PatientID1
    #    │   ├── PatientID1 - MR SAX_0000.seq.nrrd
    #    │   ├── PatientID1 - MR SAX_0001.seq.nrrd
    #    │   ├── ...
    #    ├── PatientID2
    #    │   ├── PatientID2 - MR SAX_0000.seq.nrrd
    #    │   ├── PatientID2 - MR SAX_0001.seq.nrrd
    #    │   ├── ...
    #    ├── ...
    #    ├── PatientIDN
    
    # 2. Check the image meta information, change directory, and rename NRRD files complying with Decathlon Medical Image Challenge Convention.
    output_nrrd_dir = "/mnt/data/Experiment/Data/MorphiNet-MR_CT/Dataset010_CAP_SAX_NRRD/imagesTr"   # Path to the output directory
    main(source_nrrd_dir, output_nrrd_dir)
