import os
from pathlib import Path
from dotenv import load_dotenv
import torch
import torchio as tio
import numpy as np
import ants
import antspynet
from nilearn import datasets
import nibabel as nib

ants_temp = ants.image_read('mni152_template_1mm.nii.gz')

def preprocess(nifti_file, target_shape=(128, 128, 128)):
    # N4 Bias Field Correction
    img = ants.image_read(nifti_file)
    brain_mask = ants.get_mask(img, low_thresh=None, cleanup=2)
    corrected_img = ants.n4_bias_field_correction(img, mask=brain_mask, shrink_factor=4, convergence={'iters': [50, 50, 30, 20], 'tol': 1e-7})

    # Skull Stripping using ANTsPyNet
    prob_mask = antspynet.brain_extraction(corrected_img, modality='t1', verbose=False)
    binary_mask = ants.threshold_image(prob_mask, low_thresh=0.5, high_thresh=1.0, inval=1, outval=0)
    stripped_img = corrected_img * binary_mask

    # Map to MNI152 space using ANTs registration
    reg = ants.registration(fixed=ants_temp, moving=stripped_img, type_of_transform='Affine')
    normalized_img = reg['warpedmovout']

    # Convert ANTs image to TorchIO ScalarImage
    nib_img = ants.utils.to_nibabel_nifti(normalized_img)  # Convert to nibabel format for TorchIO compatibility
    torchio_scalar_img = tio.ScalarImage(
        tensor=torch.from_numpy(nib_img.get_fdata(dtype=np.float32)).unsqueeze(0),
        affine=nib_img.affine
    )

    # Center Crop or Pad and Z-Score Normalization
    transforms = tio.Compose([
        tio.Resample(1.55, image_interpolation='linear'),  # Resample to isotropic 1.5mm voxel size
        tio.CropOrPad(target_shape, padding_mode=0),  # Center crop/pad with zeros
        tio.ZNormalization(masking_method=lambda x: x > 0)  # Z-score normalization inside brain mask
    ])

    processed_subject = transforms(torchio_scalar_img)

    # Explicitly zero out background voxels
    processed_tensor = processed_subject.data
    processed_tensor = torch.where(processed_tensor > processed_tensor.min(), processed_tensor, torch.tensor(0.0))  # Ensure background is zeroed out

    processed_tensor = processed_tensor.unsqueeze(0).float()
    return processed_tensor

def tensor_to_nifti(tensor: torch.Tensor, output_path: str, affine: np.ndarray = None):
    """
    Converts a 5D PyTorch tensor (1, 1, Depth, Height, Width) back to a .nii.gz file.
    """
    # 1. Remove batch and channel dimensions -> shape becomes (128, 128, 128)
    array_3d = tensor.squeeze().detach().cpu().numpy()
    
    # 2. Use identity matrix if no specific spatial affine is provided
    if affine is None:
        affine = np.eye(4)
        
    # 3. Create NIfTI image and save to disk
    nifti_img = nib.Nifti1Image(array_3d, affine=affine)
    nib.save(nifti_img, output_path)
    print(f" Saved NIfTI file to: {output_path}")

if __name__ == "__main__":
    patient_directory = Path(r'input_here') 
    
    output_directory = Path('processed_tensors') 
    output_directory.mkdir(parents=True, exist_ok=True)
    
    nifti_files = list(patient_directory.glob('*.nii.gz'))
    print(f" {len(nifti_files)} in {patient_directory.resolve()}")
    
    for idx, nifti_file in enumerate(nifti_files, 1):
        # Skip hidden operating system cache files
        if nifti_file.name.startswith('.'):
            continue
            
        print(f"[{idx}/{len(nifti_files)}] Preprocessing: {nifti_file.name}")
        
        try:
            processed_tensor = preprocess(
                nifti_file=str(nifti_file),
                target_shape=(128, 128, 128)
            )
            
            # Safely strip out the extension to create a clean output name
            clean_stem = nifti_file.name.split('.nii')[0]
            save_path = output_directory / f"{clean_stem}_preprocessed.pt"
            
            # Save the raw tensor array to your disk
            torch.save(processed_tensor, save_path)
            print(f"   Saved successfully! Shape: {tuple(processed_tensor.shape)}")
            
        except Exception as e:
            print(f"Error processing file {nifti_file.name}: {str(e)}")
            continue

    print(f"\n Tensors saved to: {output_directory.resolve()}")
