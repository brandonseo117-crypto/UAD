from preprocess import tensor_to_nifti
import nibabel as nib
import torch

mni_affine = nib.load('mni152_template_1mm.nii.gz').affine
tensor = torch.load(r"C:\Users\Owner\Downloads\processed_tensors_ad_test\153_S_4172_MPRAGE_20111031140639_I153_S_4172_preprocessed.pt")
tensor_to_nifti(tensor, 'here.nii.gz', affine=mni_affine)