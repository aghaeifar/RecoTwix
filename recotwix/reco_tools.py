import torch
from bart import bart
from tqdm import tqdm


cfl_order = [
    'READ_DIM',
    'PHS1_DIM',
    'PHS2_DIM',
    'COIL_DIM',
    'MAPS_DIM',
    'TE_DIM',
    'COEFF_DIM',
    'COEFF2_DIM',
    'ITER_DIM',
    'CSHIFT_DIM',
    'TIME_DIM',
    'TIME2_DIM',
    'LEVEL_DIM',
    'SLICE_DIM',
    'AVG_DIM',
    'BATCH_DIM'
]

dim_map = {
    'Ide': 'MAPS_DIM',
    'Idd': 'COEFF2_DIM',
    'Idc': 'COEFF_DIM',
    'Idb': 'ITER_DIM',
    'Ida': 'LEVEL_DIM',
    'Seg': 'BATCH_DIM',
    'Set': 'CSHIFT_DIM',
    'Rep': 'TIME_DIM',
    'Phs': 'TIME2_DIM',
    'Eco': 'TE_DIM',
    'Par': 'PHS2_DIM',
    'Sli': 'SLICE_DIM',
    'Ave': 'AVG_DIM',
    'Lin': 'PHS1_DIM',
    'Cha': 'COIL_DIM',
    'Col': 'READ_DIM'
}

axis_cha = 14

##########################################################
# FFT & iFFT
def ifftnd(kspace:torch.Tensor, axes=[-1]):
    from torch.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img  = fftshift(ifftn(ifftshift(kspace, dim=axes), dim=axes), dim=axes)
    return img


def fftnd(img:torch.Tensor, axes=[-1]):
    from torch.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace  = fftshift(fftn(ifftshift(img, dim=axes), dim=axes), dim=axes)
    return kspace

##########################################################
# BART wrapper
# adapting to bart CFL format, see https://bart-doc.readthedocs.io/en/latest/data.html or https://github.com/mrirecon/bart/blob/master/src/misc/mri.h
# Dimensions in BART (/src/misc/mri.h): [READ_DIM, PHS1_DIM, PHS2_DIM, COIL_DIM, MAPS_DIM, TE_DIM, COEFF_DIM, COEFF2_DIM, ITER_DIM, CSHIFT_DIM, TIME_DIM, TIME2_DIM, LEVEL_DIM, SLICE_DIM, AVG_DIM, BATCH_DIM]
def toBART(kspace:torch.Tensor):
    return torch.movedim(kspace, torch.arange(kspace.ndim).tolist(), [cfl_order.index(v) for v in dim_map.values()])

# converting BART data format to recoMRD data format
def fromBART(kspace:torch.Tensor):
    kspace = kspace[(...,) + (None,)*(len(dim_map) - kspace.ndim)] # add extra dims to match original format size if is not already!
    return torch.movedim(kspace, torch.arange(kspace.ndim).tolist(), [list(dim_map.values()).index(v) for v in cfl_order])


##########################################################
# applying iFFT to kspace and build image
def kspace_to_image(kspace:torch.Tensor, dim_enc=None, dim_loop=None):
    img = torch.zeros_like(kspace, dtype=kspace.dtype, device=kspace.device)
    for cha in tqdm(range(kspace.shape[dim_loop]), desc='k-space to image'):
        img.index_copy_(dim_loop, torch.Tensor([cha]).long().to(kspace.device), 
                        ifftnd(kspace.index_select(dim_loop, torch.Tensor([cha]).int().to(kspace.device)), axes=dim_enc)) 
    return img


def image_to_kspace(img:torch.Tensor, dim_enc=None, dim_loop=None):
    kspace = torch.zeros_like(img, dtype=img.dtype, device=img.device)
    for cha in tqdm(range(img.shape[dim_loop]), desc='image to k-space'):
        kspace.index_copy_(dim_loop, torch.Tensor([cha]).long().to(img.device), 
                            fftnd(img.index_select(dim_loop, torch.Tensor([cha]).int().to(img.device)), axes=dim_enc))
    return kspace

##########################################################
def calc_coil_sensitivity(acs:torch.Tensor, dim_enc, method='caldir'):
    print('Computing coil sensitivity...')
    torch.cuda.empty_cache()
    all_methods = ('espirit', 'caldir')
    
    if method.lower() not in all_methods:
        print(f'Given method is not valid. Choose between {", ".join(all_methods)}')
        return

    acs_bart = toBART(acs).numpy() # adapting to bart CFL format
    
    bart_slc_dim = cfl_order.index('SLICE_DIM')        
    if method.lower() == 'espirit'.lower():
        coil_sens = bart(1, f'-p {1<<bart_slc_dim} -e {acs_bart.shape[bart_slc_dim]} ecalib -m 1 -d 4', acs_bart)
        coil_sens = fromBART(torch.from_numpy(coil_sens))

    elif method.lower() == 'caldir'.lower():
        kernel_size = max([acs.shape[d] for d in dim_enc])//2
        if kernel_size >= acs.shape[dim_enc[0]] :
            kernel_size = acs.shape[dim_enc[0]] // 2
        print(f'kernel_size: {kernel_size}')
        coil_sens = bart(1, '-p {} -e {} caldir {}'.format(1<<bart_slc_dim, acs_bart.shape[bart_slc_dim], kernel_size), acs_bart)
        coil_sens = fromBART(torch.from_numpy(coil_sens))

    return coil_sens


def coil_combination(kspace:torch.Tensor, coil_sens:torch.Tensor, dim_enc, rss=False):
    print(f'Combining coils... ')
    torch.cuda.empty_cache()
    # sos    
    volume       = kspace_to_image(kspace, dim_enc=dim_enc, dim_loop=axis_cha)       
    volume_comb  = torch.sqrt(torch.sum(torch.abs(volume)**2, axis_cha, keepdims=True)) # is needed in 'bart' to calculate scale factor
    bart_slc_dim = cfl_order.index('SLICE_DIM')
    if rss == False:
        GPU          = '-g' # if kspace.device == 'cuda' else ''
        l2_reg       = 1e-4
        kspace       = toBART(kspace)
        coil_sens    = toBART(coil_sens)
        scale_factor = torch.quantile(volume_comb, 0.99).tolist()        
        recon        = bart(1, f'-p {1<<bart_slc_dim} -e {kspace.shape[bart_slc_dim]} pics {GPU} -d4 -w {scale_factor} -R Q:{l2_reg} -S', kspace.numpy(), coil_sens.numpy())
        volume_comb  = fromBART(torch.from_numpy(recon))
  
    return volume_comb


##########################################################
# def compress_coil(self, *kspace:torch.Tensor, virtual_channels=None): 
#     # kspace[0] is the reference input to create compression matrix for all inputs, it should be GRAPPA scan for example.    
#     print('Compressing Rx channels...')
#     torch.cuda.empty_cache()
#     if virtual_channels == None:
#         virtual_channels = int(kspace[0].shape[self.dim_info['cha']['ind']] * 0.75)

#     kspace_cfl = [torch.moveaxis(kspc, self.dim_info['cha']['ind'], BART_COIL_DIM) for kspc in kspace] # adapting to bart CFL format
#     cc_matrix  = [bart.bart(1, 'bart cc -A -M', kspace_cfl[0][...,cslc,0,0,0,0,0,0].numpy()) for cslc in range(self.dim_info['slc']['len'])]

#     kspace_compressed_cfl = []
#     for kspc in kspace_cfl:
#         n_extra1 = torch.prod(kspc.shape[self.dim_info['slc']['ind']:])   # number of extra dims  
#         n_extra2 = torch.prod(kspc.shape[self.dim_info['slc']['ind']+1:]) # number of extra dims excluing slice
#         kspc_r   = kspc.reshape(kspc.shape[:self.dim_info['slc']['ind']] + (-1,)) # flatten extra dims
#         kspc_cc  = [bart.bart(1, f'ccapply -p {virtual_channels}', k, cc_matrix[i%n_extra2]) for k, i in zip(kspc_r, range(n_extra1))]
#         kspace_compressed_cfl.append(torch.from_numpy(np.stack(kspc_cc, axis=4)).reshape(kspc.shape))

#     kspace_compressed = [torch.moveaxis(kspc, BART_COIL_DIM, self.dim_info['cha']['ind']) for kspc in kspace_compressed_cfl]
#     return (*kspace_compressed,)



    ##########################################################
    def noise_whitening(self, kspace:torch.Tensor, noise:torch.Tensor):
        pass

    ##########################################################
    def get_gfactor(self, kspace:torch.Tensor, coil_sens:torch.Tensor):
        pass
    
    ##########################################################

# Partial Fourier using Projection onto Convex Sets
def POCS(kspace:torch.Tensor, dim_enc, dim_pf=1, number_of_iterations=5, device='cuda'):
    print(f'POCS reconstruction along dim = {dim_pf} started...')
    torch.cuda.empty_cache()
    kspace = kspace.to(device)

    dim_nonpf     = tuple([int(x) for x in range(kspace.ndim) if x != dim_pf])        
    dim_nonpf_enc = tuple(set(dim_enc) & set(dim_nonpf))

    n_full = kspace.shape[dim_pf] 
    # mask for partial Fourier dimension taking accelleration into account
    mask    = torch.sum(torch.abs(kspace), dim_nonpf) > 0 # a mask along PF direction, considering acceleration, type: tensor
    
    mask_clone = mask.clone()
    ind_one = torch.nonzero(mask == True, as_tuple=True)[0] # index of mask_pf_acc, type: tensor
    acc_pf  = ind_one[1] - ind_one[0] # accelleration in partial Fourier direction
    # partial Fourier is at the beginning or end of the dimension
    ind_samples = torch.arange(ind_one[-1]+1) # index of samples in PF direction, without acceleration. ind_nopocs does not take accelleration into account, right?
    if ind_one[0] > (mask.numel() - ind_one[-1] - 1): # check which side has more zeros, beginning or end
        ind_samples = torch.arange(ind_one[0], mask.numel())
    # mask if there was no accelleration in PF direction
    mask[ind_samples] = True 
    # vector mask for central region
    mask_sym = mask & torch.flip(mask, dims=[0])     
    # gaussian mask for central region in partial Fourier dimension
    gauss_pdf = torch.signal.windows.gaussian(n_full, std=10, device=kspace.device) * mask_sym
    # kspace smoothed with gaussian profile and masked central region
    kspace_symmetric = kspace.clone()
    kspace_symmetric = torch.swapaxes(torch.swapaxes(kspace_symmetric, dim_pf, -1) * gauss_pdf, -1, dim_pf)
    angle_image_symmetric  = kspace_to_image(kspace_symmetric, dim_enc=dim_enc, dim_loop=axis_cha) # along non-pf encoding directions
    angle_image_symmetric /= torch.abs(angle_image_symmetric) # normalize to unit circle       

    kspace_full = kspace_to_image(kspace, dim_enc=dim_nonpf_enc, dim_loop=axis_cha) # along non-pf encoding directions
    kspace_full_clone = kspace_full.clone()
    # free memory
    del kspace_symmetric 
    del kspace
    torch.cuda.empty_cache()

    for ind in range(number_of_iterations):
        image_full  = kspace_to_image(kspace_full, dim_enc=[dim_pf], dim_loop=axis_cha)
        image_full  = torch.abs(image_full) * angle_image_symmetric
        kspace_full = image_to_kspace(image_full, dim_enc=[dim_pf], dim_loop=axis_cha)
        torch.moveaxis(kspace_full, dim_pf, 0)[mask] = torch.moveaxis(kspace_full_clone, dim_pf, 0)[mask] # replace elements of kspace_full from original kspace_full_clone

    kspace_full = image_to_kspace(kspace_full, dim_enc=dim_nonpf_enc, dim_loop=axis_cha)
    # remove all samples that was not part of the original dataset (e.g. acceleartion)        
    mask = mask_clone
    mask[ind_one[0]%acc_pf::acc_pf] = True
    torch.moveaxis(kspace_full, dim_pf, 0)[~mask] = 0       

    return kspace_full.to('cpu') 