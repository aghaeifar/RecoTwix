
import nibabel as nib
import torch
from recotwix.sequences import recoB0

filename = '/media/b0shim/basismaps.dat'

B0 = recoB0(filename)
B0map = B0.get_b0hz()
print(B0map.shape)

reshape_shape = [1] * B0map.ndim
reshape_shape[B0.dim_info['Rep']['ind']] = B0.dim_info['Rep']['len']
# scales used to measure the shim profiles; here up to 3rd order for 7T Terra
scales = torch.tensor([1, 300, 300, 300, 1500, 1500, 1500, 1500, 1500, 5000, 5000, 5000, 5000]).view(*reshape_shape)
print(scales.shape)

B0mapScaled = B0map / scales
# set freq. dimension to 1
B0mapScaled.index_fill_(B0.dim_info['Rep']['ind'], torch.tensor([0]), 1)

b0_nii = B0.make_nifti(B0mapScaled)
nib.save(b0_nii, '/media/b0shim/basismaps.nii.gz')  