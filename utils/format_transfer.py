#!/usr/bin/env python

import nrrd
import nibabel as nib
from pydicom import dcmread
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse
import warnings
import os
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=FutureWarning)
import nibabel as nib

PRECISION = 17
np.set_printoptions(precision=PRECISION, suppress=True, floatmode='maxprec')


def _space2ras(space):
    '''Find the diagonal transform required to transform space to RAS'''

    if len(space)==3:
        # short definition LPI
        positive=[space[0],space[1],space[2]]

    else:
        # long definition left-posterior-inferior
        positive=space.split('-')

    xfrm=[ ]
    if positive[0][0].lower() == 'l': # 'left'
        xfrm.append(-1)
    else:
        xfrm.append(1)

    if positive[1][0].lower() == 'p': # 'posterior'
        xfrm.append(-1)
    else:
        xfrm.append(1)

    if positive[2][0].lower() == 'i': # 'inferior'
        xfrm.append(-1)
    else:
        xfrm.append(1)

    # return 4x4 diagonal matrix
    xfrm.append(1)
    return np.diag(xfrm)


def nifti_write(inImg, outDir):

    prefix = os.path.abspath(inImg).split('.')[0]

    img = nrrd.read(inImg)
    hdr = img[1]
    data = img[0]

    # # find the Patient Name (0x00100020), Study ID (0x00200010), Series Number (0x00200011) to form the filename
    # dcm = dcmread(hdr['MultiVolume.FrameFileList'].split(',')[0])   # first Instance in the Series folder
    # patient_id = dcm.PatientID
    # if patient_id == '':
    #     if dcm.PatientName != '':
    #         patient_id = dcm.PatientName
    #     else:
    #         patient_id = 'Unknown'
    # study_id = dcm.StudyID
    # series_number = dcm.SeriesNumber
    filename = os.path.basename(inImg).split(' - ')[0].replace('.nrrd', '')

    os.makedirs(outDir, exist_ok=True)
    if not os.path.exists(os.path.join(outDir, f"{filename}-00.nii.gz")):

        SPACE_UNITS = 2
        TIME_UNITS = 0

        SPACE2RAS = _space2ras(hdr['space'].split('-'))

        translation = hdr['space origin']
        
        if hdr['dimension'] == 4:
            axis_elements = hdr['kinds']
            for i in range(4):
                if axis_elements[i] == 'list' or axis_elements[i] == 'vector':
                    grad_axis = i
                    break
            
            volume_axes = [0,1,2,3]
            volume_axes.remove(grad_axis)
            rotation = hdr['space directions'][volume_axes,:3]
            
            xfrm_nhdr = np.array(np.vstack((np.hstack((rotation.T, np.reshape(translation,(3,1)))),[0,0,0,1])))

            # put the gradients along last axis
            if grad_axis != 3:
                data = np.moveaxis(data, grad_axis, 3)
            
            try:
                # DWMRI
                # write .bval and .bvec
                f_val = open(prefix + '.bval', 'w')
                f_vec = open(prefix + '.bvec', 'w')
                b_max = float(hdr['DWMRI_b-value'])

                mf = np.array(np.vstack((np.hstack((hdr['measurement frame'],
                                                    [[0],[0],[0]])),[0,0,0,1])))
                for ind in range(hdr['sizes'][grad_axis]):
                    bvec  = [float(num) for num in hdr[f'DWMRI_gradient_{ind:04}'].split()]
                    L_2 = np.linalg.norm(bvec[:3])
                    bval = round(L_2 ** 2 * b_max)

                    bvec.append(1)
                    # bvecINijk = RAS2IJK @ SPACE2RAS @ mf @ np.matrix(bvec).T
                    # simplified below
                    bvecINijk = xfrm_nhdr.T @ mf @ np.array(bvec).T

                    L_2 = np.linalg.norm(bvecINijk[:3])
                    if L_2:
                        bvec_norm = bvecINijk[:3]/L_2
                    else:
                        bvec_norm = [0, 0, 0]

                    f_val.write(str(bval)+' ')
                    f_vec.write(('  ').join(str(x) for x in np.array(bvec_norm).flatten())+'\n')

                f_val.close()
                f_vec.close()
            
            except:
                # fMRI
                pass
            
            TIME_UNITS = 8
        
        else:
            rotation = hdr['space directions']
            xfrm_nhdr = np.matrix(np.vstack((np.hstack((rotation.T, np.reshape(translation,(3,1)))),[0,0,0,1])))


        xfrm_nifti = SPACE2RAS @ xfrm_nhdr
        # RAS2IJK = xfrm_nifti.I

        if hdr['dimension'] == 3:
            img_nifti = nib.nifti1.Nifti1Image(data, affine=xfrm_nifti)
            hdr_nifti = img_nifti.header

            # now set xyzt_units, sform_code= qform_code= 2 (aligned)
            # https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/xyzt_units.html
            # simplification assuming 'mm' and 'sec'
            hdr_nifti.set_xyzt_units(xyz=SPACE_UNITS, t=TIME_UNITS)
            hdr_nifti['qform_code'] = 2
            hdr_nifti['sform_code'] = 2

            hdr_nifti['descrip'] = 'NRRD-->NIFTI transform by Tashrif Billah revised by Yu Deng'

            nib.save(img_nifti, os.path.join(outDir, f"{filename}-00.nii.gz"))
        else:
            # the following code treats the sequence axis as a volume axis, and the spacing for trigger time will be wrong if not convert the nifti back to nrrd
            for i in range(data.shape[2]):  # loop through the volume z-axis
                # automatically sets dim, data_type, pixdim, affine
                img_nifti = nib.nifti1.Nifti1Image(data[:, :, i], affine=xfrm_nifti)
                hdr_nifti = img_nifti.header

                # now set xyzt_units, sform_code= qform_code= 2 (aligned)
                # https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/xyzt_units.html
                # simplification assuming 'mm' and 'sec'
                hdr_nifti.set_xyzt_units(xyz=SPACE_UNITS, t=TIME_UNITS)
                hdr_nifti['qform_code'] = 2
                hdr_nifti['sform_code'] = 2

                hdr_nifti['descrip'] = 'NRRD-->NIFTI transform by Tashrif Billah revised by Yu Deng'

                nib.save(img_nifti, os.path.join(outDir, f"{filename}-{i:02d}.nii.gz"))


def nrrd_write(inDir, srcImg, outDir):
    inImgs = glob(inDir + '/*.nii.gz')    # list of sequence volume files
    inImgs = set([f.split('-')[0] for f in inImgs])
    os.makedirs(outDir, exist_ok=True)
    print(f'Converting {len(inImgs)} nifti files to nrrd format')
    for inImg in tqdm(inImgs):
        # Infer the source nrrd file from the inImg directory
        source_nrrd = glob(os.path.join(srcImg, os.path.basename(inImg).split('-')[0] + '*.seq.nrrd'))[0]

        # Read the header info from the source nrrd file
        source = nrrd.read(source_nrrd)
        source_img, source_hdr = source[0], source[1]

        # Combine the pixel data from the inImg file and the header data from the source nrrd file
        out_data = None
        for file in glob(os.path.join(inDir, inImg + '*.nii.gz')):
            if out_data is None:
                out_data = nib.load(file).get_fdata()[..., None]
            else:
                out_data = np.concatenate((out_data, nib.load(file).get_fdata()[..., None]), axis=3)
        out_data = np.moveaxis(out_data, 2, 0)  # restore the sequence axis
        out_data = out_data.astype(source_img.dtype)

        # Create a new nrrd file with the combined pixel data and header data
        outImg = os.path.join(outDir, os.path.basename(source_nrrd).replace('seq', 'seg'))
        nrrd.write(outImg, out_data, source_hdr)


def file_filter(args):
    '''filter the files in the input directory and pass only fMRI nrrd files for conversion'''
    if args.mode == 'nrrd2nii':
        inImgs = glob(args.input + '/*.seq.nrrd')    # list of sequence volume files
        if len(inImgs) == 0:
            inImgs = glob(args.input + '/*.nrrd')
        print(f'Converting {len(inImgs)} nrrd files to nifti format')
        for inImg in tqdm(inImgs):
            nifti_write(inImg, args.output)
    elif args.mode == 'nii2nrrd':
        nrrd_write(args.input, args.source, args.output)
    else:
        raise ValueError('Invalid mode of conversion, be either "nrrd2nii" or "nii2nrrd"')


def main():
    parser = argparse.ArgumentParser(description='NRRD to NIFTI conversion tool')
    parser.add_argument('-i', '--input', type=str, help='input nrrd/nifti file', default="/mnt/data/Experiment/nnUNet/nnUNet_raw/Dataset011_CAP_SAX/imagesTr")
    parser.add_argument('-o', '--output', type=str, help='output file directory', default="/mnt/data/Experiment/Data/CAP/Dataset011_CAP_SAX_NRRD/imagesTr")
    parser.add_argument('-m', '--mode', type=str, help='mode of conversion, be either "nrrd2nii" or "nii2nrrd"', default="nii2nrrd")
    # parser.add_argument('-i', '--input', type=str, help='input nrrd/nifti file', required=True)
    # parser.add_argument('-o', '--output', type=str, help='output file directory', required=True)
    # parser.add_argument('-m', '--mode', type=str, help='mode of conversion, be either "nrrd2nii" or "nii2nrrd"', required=True)
    parser.add_argument('-s', '--source', type=str, help='source nrrd file directory', default='')

    args = parser.parse_args()
    
    file_filter(args)


if __name__ == '__main__':
    main()