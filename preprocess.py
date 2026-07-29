import nibabel as nib        # For nifti files
import numpy as np           # For matrix math
import SimpleITK as sitk     # For N4 correction
import torchio as tio        # For deep learning
from dcm2niix import main as dcm2niix_run
from nilearn.datasets import MNI152_FILE_PATH
import torch
import os
import tempfile
import torchio as tio
import ants
from pathlib import Path
from dotenv import load_dotenv

def preprocess_mri_volume(
    nifti_path: str,
    mni_template_path: str,
    target_shape: tuple = (128, 128, 128),
    use_hdbet: bool = False  # Set to False to skip HD-BET entirely
) -> torch.Tensor:
    """
    3D MRI Preprocessing pipeline using ANTsPy for MNI152 registration
    and TorchIO for normalization and formatting.
    """
    subject = tio.Subject(t1w=tio.ScalarImage(nifti_path))
    ras_transform = tio.ToCanonical()
    subject = ras_transform(subject)
    
    # 1. ANTs Registration to MNI152 Template
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_temp = os.path.normpath(os.path.join(tmp_dir, "input_ras.nii.gz"))
        subject.t1w.save(input_temp)
        
        moving_ants = ants.image_read(input_temp)
        template_ants = ants.image_read(mni_template_path)

        n4_corrected_ants = ants.n4_bias_field_correction(moving_ants)

        # Affine Registration to MNI152 Template
        reg = ants.registration(
            fixed=template_ants,
            moving=n4_corrected_ants,
            type_of_transform='Affine'
        )
        registered_ants = reg['warpedmovout']

        # Convert C++ memory buffer to PyTorch Tensor in RAM
        reg_tensor = torch.from_numpy(registered_ants.numpy()).unsqueeze(0)
        
        # Attach MNI spatial orientation matrix
        template_io = tio.ScalarImage(mni_template_path)
        subject.t1w = tio.ScalarImage(tensor=reg_tensor, affine=template_io.affine)

    # 2. Simple Local Skull-Stripping (No HD-BET temp file bugs)
    foreground_mask = tio.ForegroundMask(mask_name='brain_mask')
    subject = foreground_mask(subject)
    subject.t1w.data = subject.t1w.data * subject.brain_mask.data

    # 3. Resampling & Intensity Normalization
    target_resample = tio.Resize(target_shape)
    subject = target_resample(subject)

    subject = tio.ForegroundMask(mask_name='final_brain_mask')(subject)
    z_norm = tio.ZNormalization(masking_method='final_brain_mask')
    subject = z_norm(subject)

    tensor_4d = subject.t1w.data        # Shape: (1, D, H, W)
    tensor_5d = tensor_4d.unsqueeze(0).float()  # Shape: (1, 1, D, H, W)

    return tensor_5d

load_dotenv()
folder_path_processed = os.getenv('PROCESSED')
DIMS = (128,128,128)

patient_directory = Path(folder_path_processed)
output_directory = Path(r'processed_tensors')
output_directory.mkdir(parents=True, exist_ok=True)

for nifti_file in patient_directory.rglob('*.nii.gz'):
    if nifti_file.name.startswith('.'):
        continue

    processed_tensor = preprocess_mri_volume(
        nifti_path=str(nifti_file),
        mni_template_path=str(MNI152_FILE_PATH),
        target_shape=DIMS,
        use_hdbet=False
    )

    # Save the resulting 5D tensor for model training
    clean_stem = nifti_file.name.replace('.nii.gz', '')
    save_path = output_directory / f"{clean_stem}_preprocessed.pt"
    torch.save(processed_tensor, save_path)