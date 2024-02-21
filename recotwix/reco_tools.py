import sys, os
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

recotwix_order = [
    'Ide',
    'Idd',
    'Idc',
    'Idb',
    'Ida',
    'Seg',
    'Set',
    'Rep',
    'Phs',
    'Eco',
    'Par',
    'Sli',
    'Ave',
    'Lin',
    'Cha',
    'Col'
]

dim_map = {
    'Par': 'PHS2_DIM',
    'Sli': 'TE_DIM',
    'Lin': 'PHS1_DIM',
    'Cha': 'COIL_DIM',
    'Col': 'READ_DIM'
}

fft_loop_axis       = recotwix_order.index('Cha')
bart_flatten_axis   = cfl_order.index('COEFF_DIM')
bart_slc_axis       = cfl_order.index('TE_DIM')

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
    kspace = torch.unsqueeze(kspace, -1)
    kspace = torch.swapdims(kspace, recotwix_order.index('Ida'), -1)
    kspace = torch.movedim(kspace, [recotwix_order.index(v) for v in dim_map.keys()], [cfl_order.index(v) for v in dim_map.values()])
    unflatten_shape = [*kspace.shape][bart_flatten_axis:]
    kspace = torch.flatten(kspace, bart_flatten_axis, -1)
    kspace = kspace[(...,) + (None,)*(len(cfl_order) - kspace.ndim)] # add extra dims to match number of dimensions if is not already!
    return kspace, unflatten_shape

# converting BART data format to recoMRD data format
def fromBART(kspace:torch.Tensor, unflatten_shape):
    kspace = kspace[(...,) + (None,)*(len(cfl_order) - kspace.ndim)] # add extra dims to match number of dimensions if is not already!
    kspace = torch.squeeze(kspace, tuple(range(bart_flatten_axis+1, kspace.ndim)))
    kspace = torch.unflatten(kspace, bart_flatten_axis, unflatten_shape)
    kspace = torch.movedim(kspace, [cfl_order.index(v) for v in dim_map.values()], [recotwix_order.index(v) for v in dim_map.keys()])
    kspace = torch.swapaxes (kspace, -1, recotwix_order.index('Ida'))
    kspace = torch.squeeze(kspace, -1)
    return kspace


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
    acs_bart, unflatten_shape = toBART(acs)# adapting to bart CFL format
    if method.lower() == 'espirit'.lower():
        coil_sens = bart(1, f'-p {1<<bart_slc_axis} -e {acs_bart.shape[bart_slc_axis]} ecalib -m 1 -d 4', acs_bart.detach().cpu().numpy() )        
        coil_sens = fromBART(torch.from_numpy(coil_sens), unflatten_shape)

    elif method.lower() == 'caldir'.lower():
        kernel_size = max([acs.shape[d] for d in dim_enc])//2
        if kernel_size >= acs.shape[dim_enc[0]] :
            kernel_size = acs.shape[dim_enc[0]] // 2
        coil_sens = bart(1, '-p {} -e {} caldir {}'.format(1<<bart_slc_axis, acs_bart.shape[bart_slc_axis], kernel_size), acs_bart.detach().cpu().numpy())
        coil_sens = fromBART(torch.from_numpy(coil_sens), unflatten_shape)

    return coil_sens


def coil_combination(kspace:torch.Tensor, coil_sens:torch.Tensor, dim_enc, rss=False, supress_output=False):
    print(f'Combining coils... ')
    torch.cuda.empty_cache()     
    # sos    
    volume       = kspace_to_image(kspace, dim_enc=dim_enc, dim_loop=fft_loop_axis)       
    volume_comb  = torch.sqrt(torch.sum(torch.abs(volume)**2, fft_loop_axis, keepdims=True)) # is needed in 'bart' to calculate scale factor
    if rss == False:
        recon        = list() 
        GPU          = '-g' # if kspace.device == 'cuda' else ''
        l2_reg       = 1e-4
        scale_factor = torch.quantile(volume_comb, 0.99).tolist()         
        coil_sens    = toBART(coil_sens)[0].detach().cpu().numpy()
        kspace, unflatten_shape  = toBART(kspace)
        par_dim  = bart_slc_axis if bart_slc_axis>bart_flatten_axis else bart_flatten_axis
        loop_dim = bart_flatten_axis if bart_slc_axis>bart_flatten_axis else bart_slc_axis
        sys.stdout = open(os.devnull, 'w') if supress_output else sys.__stdout__
        for chunk in kspace.split(1, dim=loop_dim):             
            comb = bart(1, f'-p {1<<par_dim} -e {kspace.shape[par_dim]} pics {GPU} -d4 -w {scale_factor} -R Q:{l2_reg} -S', chunk.detach().cpu().numpy(), coil_sens)
            comb = torch.from_numpy(comb)
            recon.append(comb[(...,) + (None,)*(kspace.ndim - comb.ndim)]) # we need to add singleton dimensions to match the number of dimensions of kspace. It is needed for concatenation
        
        sys.stdout = sys.__stdout__
        recon = torch.cat(recon, dim=loop_dim)
        volume_comb  = fromBART(recon, unflatten_shape)
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
    angle_image_symmetric  = kspace_to_image(kspace_symmetric, dim_enc=dim_enc, dim_loop=fft_loop_axis) # along non-pf encoding directions
    angle_image_symmetric /= torch.abs(angle_image_symmetric) # normalize to unit circle       

    kspace_full = kspace_to_image(kspace, dim_enc=dim_nonpf_enc, dim_loop=fft_loop_axis) # along non-pf encoding directions
    kspace_full_clone = kspace_full.clone()
    # free memory
    del kspace_symmetric 
    del kspace
    torch.cuda.empty_cache()

    for ind in range(number_of_iterations):
        image_full  = kspace_to_image(kspace_full, dim_enc=[dim_pf], dim_loop=fft_loop_axis)
        image_full  = torch.abs(image_full) * angle_image_symmetric
        kspace_full = image_to_kspace(image_full, dim_enc=[dim_pf], dim_loop=fft_loop_axis)
        torch.moveaxis(kspace_full, dim_pf, 0)[mask] = torch.moveaxis(kspace_full_clone, dim_pf, 0)[mask] # replace elements of kspace_full from original kspace_full_clone

    kspace_full = image_to_kspace(kspace_full, dim_enc=dim_nonpf_enc, dim_loop=fft_loop_axis)
    # remove all samples that was not part of the original dataset (e.g. acceleartion)        
    mask = mask_clone
    mask[ind_one[0]%acc_pf::acc_pf] = True
    torch.moveaxis(kspace_full, dim_pf, 0)[~mask] = 0       

    return kspace_full.to('cpu') 