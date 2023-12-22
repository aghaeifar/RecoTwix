import os
import ctypes
import torch
import twixtools
import numpy as np
import nibabel as nib
from bart import bart
from tqdm import tqdm
from pytools import plot
from protocol import protocol_parse

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

# adapting to bart CFL format, see https://bart-doc.readthedocs.io/en/latest/data.html or https://github.com/mrirecon/bart/blob/master/src/misc/mri.h
# Dimensions in BART (/src/misc/mri.h): [READ_DIM, PHS1_DIM, PHS2_DIM, COIL_DIM, MAPS_DIM, TE_DIM, COEFF_DIM, COEFF2_DIM, ITER_DIM, CSHIFT_DIM, TIME_DIM, TIME2_DIM, LEVEL_DIM, SLICE_DIM, AVG_DIM, BATCH_DIM]
def toBART(kspace:torch.Tensor):
    return torch.movedim(kspace, torch.arange(kspace.ndim).tolist(), [cfl_order.index(v) for v in dim_map.values()])

# converting BART data format to recoMRD data format
def fromBART(kspace:torch.Tensor):
    kspace = kspace[(...,) + (None,)*(len(dim_map) - kspace.ndim)] # add extra dims to match original format size if is not already!
    return torch.movedim(kspace, torch.arange(kspace.ndim).tolist(), [list(dim_map.values()).index(v) for v in cfl_order])


class recotwix(): 
    hdr      = {}
    prot     = {}
    img      = None
    protName = None   
    twixobj  = None
    twixmap  = {}
    scans    = []
    device   = 'cpu'
    dim_enc  = None
    dim_free = None
    dim_size = None
    dim_info = None
    transformation = dict()
    nii_affine     = None
    
    def __init__(self, filename=None, device='cpu'):   
        self.device = device
        self.twixobj = twixtools.read_twix(filename)[-1]
        self.twixmap = twixtools.map_twix(self.twixobj)

        for key in self.twixmap.keys():
            if isinstance(self.twixmap[key], twixtools.twix_array):
                self.twixmap[key].flags['remove_os'] = True
        self.twixmap['image'].flags['zf_missing_lines'] = True

        self.scans = []
        for scan in self.twixmap:
            if isinstance(self.twixmap[scan], twixtools.twix_array):
                self.scans.append(scan)
        
        hdr = self.twixobj['hdr']        
        self.hdr = hdr

        self.dim_size = self.twixmap['image'].shape
        self.dim_info = {}
        for dim in self.twixmap['image'].dims: # ('Ide', 'Idd', 'Idc', 'Idb', 'Ida', 'Seg', 'Set', 'Rep', 'Phs', 'Eco', 'Par', 'Sli', 'Ave', 'Lin', 'Cha', 'Col')
            ind = self.twixmap['image'].dims.index(dim)
            self.dim_info[dim] = {'ind':ind, 'len':self.dim_size[ind]}

        self.dim_enc  = [self.dim_info['Col']['ind'], self.dim_info['Lin']['ind'], self.dim_info['Par']['ind']]
        self.dim_free = ('Ide', 'Idd', 'Idc', 'Idb', 'Ida', 'Seg', 'Set', 'Rep', 'Phs', 'Eco', 'Ave')
        
        self.prot = protocol_parse(self.twixmap)
        self._extract_transformation()


    def __str__(self):
        s = f"\n{self.__class__.__module__}.{self.__class__.__qualname__}:\n"
        s += f"  Scans: {self.scans}\n"
        s += f"  Dims: {list(self.dim_info.keys())}\n"
        s += f"  Size: {self.dim_size}\n"
        s += f"  Resolution (x y z): {self.prot.res['x']} {self.prot.res['y']} {self.prot.res['z']}\n"
        s += f"  PartialFourier (ro, pe, pe3d): {self.prot.isPartialFourierRO} {self.prot.isPartialFourierPE1} {self.prot.isPartialFourierPE2}\n"
        s += f"  isParallelImaging: {self.prot.isParallelImaging}\n"
        s += f"  Acceleration Factor: {self.prot.acceleration_factor}\n"
        return s

    def runReco(self, method_sensitivity='caldir'):     
        torch.cuda.empty_cache()   

        kspace = self._getkspace()
        print(f'kspace shape: {kspace.shape}')
        kspace = self.correct_scan_size(kspace, scantype='image')
        print(f'kspace corrected shape: {kspace.shape}')
        # Partial Fourier?
        if self.prot.isPartialFourierRO:
            kspace = self.POCS(kspace, dim_pf=self.dim_info['Col']['ind'])
        if self.prot.isPartialFourierPE1:
            kspace = self.POCS(kspace, dim_pf=self.dim_info['Lin']['ind'])
        if self.prot.isPartialFourierPE2:
            kspace = self.POCS(kspace, dim_pf=self.dim_info['Par']['ind'])

        coils_sensitivity = None
        # Parallel Imaging?
        if self.prot.isParallelImaging:
            self.twixmap['refscan'].flags['zf_missing_lines'] = False
            self.twixmap['refscan'].flags['skip_empty_lead']  = True
            acs = torch.from_numpy(self.twixmap['refscan'][:])
            acs = self.correct_scan_size(acs, scantype='refscan')
            coils_sensitivity = self.calc_coil_sensitivity(acs, method=method_sensitivity)
        else:
            coils_sensitivity = self.calc_coil_sensitivity(kspace, method=method_sensitivity)
        self.img = self.coil_combination(kspace, coil_sens=coils_sensitivity)

    ##########################################################
    # applying iFFT to kspace and build image
    def kspace_to_image(self, kspace:torch.Tensor, axes=None):
        if kspace.ndim != len(self.dim_size):
            print(f'Error! shape is wrong. {kspace.shape} vs {self.dim_size}')
            return
        if axes is None:
            axes = self.dim_enc

        img = torch.zeros_like(kspace, dtype=kspace.dtype, device=kspace.device)
        # this loop is slow because of order='F' and ind is in the first dimensions. see above, _create_kspace(). 
        for cha in tqdm(range(self.dim_info['Cha']['len']), desc='Fourier transform'):
            img.index_copy_(self.dim_info['Cha']['ind'], torch.Tensor([cha]).long().to(kspace.device), 
                            ifftnd(kspace.index_select(self.dim_info['Cha']['ind'], torch.Tensor([cha]).int().to(kspace.device)), axes=axes)) 
        return img


    def image_to_kspace(self, img:torch.Tensor, axes=None):
        if img.ndim != len(self.dim_size):
            print(f'Error! shape is wrong. {img.shape} vs {self.dim_size}')
            return
        if axes is None:
            axes = self.dim_enc

        kspace = torch.zeros_like(img, dtype=img.dtype, device=img.device)
        # this loop is slow because of order='F' and ind is in the first dimensions. see above, _create_kspace(). 
        for cha in tqdm(range(self.dim_info['Cha']['len']), desc='Fourier transform'):
            kspace.index_copy_(self.dim_info['Cha']['ind'], torch.Tensor([cha]).long().to(img.device), 
                               fftnd(img.index_select(self.dim_info['Cha']['ind'], torch.Tensor([cha]).int().to(img.device)), axes=axes))
        return kspace

    ##########################################################
    def correct_scan_size(self, kspace:torch.Tensor, scantype='image'):
        # Note: this function suppose oversampling is removed
        import torch.nn.functional as F
        # os_flag = self.twixmap[scantype].flags['remove_os']
        # self.twixmap[scantype].flags['remove_os'] = False
        col_diff = self.hdr['Meas']['iRoFTLength']//2 - self.twixmap[scantype].kspace_center_col
        lin_diff = self.hdr['Meas']['iPEFTLength']//2 - self.twixmap[scantype].kspace_center_lin
        par_diff = self.hdr['Meas']['i3DFTLength']//2 - self.twixmap[scantype].kspace_center_par
        # self.twixmap[scantype].flags['remove_os'] = os_flag # restore oversampling flag
        # print(f'iRoFTLength: {self.hdr["Meas"]["iRoFTLength"]}, iPEFTLength: {self.hdr["Meas"]["iPEFTLength"]}, i3DFTLength: {self.hdr["Meas"]["i3DFTLength"]}')
        # print(f'col_diff: {col_diff}, lin_diff: {lin_diff}, par_diff: {par_diff}')
        # print(f'kspace_center_col: {self.twixmap[scantype].kspace_center_col}, kspace_center_lin: {self.twixmap[scantype].kspace_center_lin}, kspace_center_par: {self.twixmap[scantype].kspace_center_par}')
        # print(f'res x: {self.prot.res["x"] }')
        # There might be asymmetric echo; thus, padding in the left and right differ. Note we suppose asymmetry is in the left side which should be valid in the most cases where shortening echo-time is the goal.
        col_diff_l = col_diff // 2 # because oversampling is removed before
        col_diff_r = self.prot.res['x'] - kspace.shape[self.dim_info['Col']['ind']] - col_diff_l

        pad = [0] * kspace.ndim * 2
        pad[self.dim_info['Col']['ind']*2]   = int(col_diff_r)
        pad[self.dim_info['Col']['ind']*2+1] = int(col_diff_l)
        pad[self.dim_info['Lin']['ind']*2]   = int(lin_diff)
        pad[self.dim_info['Lin']['ind']*2+1] = int(lin_diff)
        pad[self.dim_info['Par']['ind']*2]   = int(par_diff)
        pad[self.dim_info['Par']['ind']*2+1] = int(par_diff)
        pad.reverse()
        return F.pad(kspace, pad, 'constant', 0)

    ##########################################################
    def calc_coil_sensitivity(self, acs:torch.Tensor, method='caldir'):
        print('Computing coil sensitivity...')
        torch.cuda.empty_cache()
        all_methods = ('espirit', 'caldir')
        
        if method.lower() not in all_methods:
            print(f'Given method is not valid. Choose between {", ".join(all_methods)}')
            return

        # picking the 0th element of the free dimensions 
        for dim_free in self.dim_free:
            acs = acs.index_select(self.dim_info[dim_free]['ind'], torch.Tensor([0]).int()) 
        acs_bart = toBART(acs).numpy() # adapting to bart CFL format
        
        bart_slc_dim = cfl_order.index('SLICE_DIM')        
        if method.lower() == 'espirit'.lower():
            coil_sens = bart(1, f'-p {1<<bart_slc_dim} -e {acs_bart.shape[bart_slc_dim]} ecalib -m 1 -d 4', acs_bart)
            coil_sens = fromBART(torch.from_numpy(coil_sens))

        elif method.lower() == 'caldir'.lower():
            kernel_size = max([acs.shape[d] for d in self.dim_enc])//2
            if kernel_size >= acs.shape[self.dim_enc[0]] :
                kernel_size = acs.shape[self.dim_enc[0]] // 2
            print(f'kernel_size: {kernel_size}')
            coil_sens = bart(1, '-p {} -e {} caldir {}'.format(1<<bart_slc_dim, acs_bart.shape[bart_slc_dim], kernel_size), acs_bart)
            coil_sens = fromBART(torch.from_numpy(coil_sens))

        print('Done!')
        return coil_sens
    
    ##########################################################
    def coil_combination(self, kspace:torch.Tensor, coil_sens:torch.Tensor, rss=False):
        print(f'Combining coils... ')
        torch.cuda.empty_cache()
        if kspace.ndim != len(self.dim_size):
            print(f'Input size is not valid. {kspace.shape} != {self.dim_size}')
            return
                
        shp = (1,) + kspace.shape[1:] # coil is 1 after combining
        # sos    
        volume       = self.kspace_to_image(kspace)       
        volume_comb  = torch.sqrt(torch.sum(torch.abs(volume)**2, self.dim_info['Cha']['ind'], keepdims=True)) # is needed in 'bart' to calculate scale factor
        bart_slc_dim = cfl_order.index('SLICE_DIM')
        if rss == False:
            GPU          = '-g' if self.device == 'cuda' else ''
            l2_reg       = 1e-4
            kspace       = toBART(kspace)
            coil_sens    = toBART(coil_sens)
            scale_factor = torch.quantile(volume_comb, 0.99).tolist()        
            recon        = bart(1, f'-p {1<<bart_slc_dim} -e {kspace.shape[bart_slc_dim]} pics {GPU} -d4 -w {scale_factor} -R Q:{l2_reg} -S', kspace.numpy(), coil_sens.numpy())
            volume_comb  = fromBART(torch.from_numpy(recon))
            
        print('Done!')
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
    # Partial Fourier using Projection onto Convex Sets
    def POCS(self, kspace:torch.Tensor, dim_pf=1, number_of_iterations=5):
        print(f'POCS reconstruction along dim = {dim_pf} started...')
        torch.cuda.empty_cache()
        kspace = kspace.to(self.device)

        dim_nonpf     = tuple([int(x) for x in range(kspace.ndim) if x != dim_pf])        
        dim_nonpf_enc = tuple(set(self.dim_enc) & set(dim_nonpf))

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
        angle_image_symmetric  = self.kspace_to_image(kspace_symmetric) 
        angle_image_symmetric /= torch.abs(angle_image_symmetric) # normalize to unit circle       

        kspace_full = self.kspace_to_image(kspace, axes=dim_nonpf_enc) # along non-pf encoding directions
        kspace_full_clone = kspace_full.clone()
        # free memory
        del kspace_symmetric 
        del kspace
        torch.cuda.empty_cache()

        for ind in range(number_of_iterations):
            image_full  = self.kspace_to_image(kspace_full, axes=[dim_pf])
            image_full  = torch.abs(image_full) * angle_image_symmetric
            kspace_full = self.image_to_kspace(image_full, axes=[dim_pf])
            torch.moveaxis(kspace_full, dim_pf, 0)[mask] = torch.moveaxis(kspace_full_clone, dim_pf, 0)[mask] # replace elements of kspace_full from original kspace_full_clone

        kspace_full = self.image_to_kspace(kspace_full, axes=dim_nonpf_enc)
        # remove all samples that was not part of the original dataset (e.g. acceleartion)        
        mask = mask_clone
        mask[ind_one[0]%acc_pf::acc_pf] = True
        torch.moveaxis(kspace_full, dim_pf, 0)[~mask] = 0       
        print('Done!')

        return kspace_full.to('cpu') 

    ##########################################################
    def noise_whitening(self, kspace:torch.Tensor, noise:torch.Tensor):
        pass

    ##########################################################
    def get_gfactor(self, kspace:torch.Tensor, coil_sens:torch.Tensor):
        pass
    
    ##########################################################

    def _getkspace(self, reorder_slice=True):
        kspace = torch.from_numpy(self.twixmap['image'][:])
        if reorder_slice:
            kspace = self._reorder_slices(kspace)
        return kspace

    def _reorder_slices(self, data:torch.Tensor):
        unsorted_order = np.zeros((self.dim_info['Sli']['len']))
        transform_inv = np.linalg.inv(self.transformation['mat44'])
        for cSlc in range(self.dim_info['Sli']['len']):
            p = transform_inv @ self.transformation['soda'][cSlc,:,3]
            unsorted_order[cSlc] = p[2]  # z position before transformation

        ind_sorted = np.argsort(unsorted_order)
        self.transformation['soda'] = self.transformation['soda'][ind_sorted,...]
        return data.index_select(self.dim_info['Sli']['ind'], torch.from_numpy(ind_sorted))


    def _extract_transformation(self):
        from scipy.spatial.transform import Rotation as R

        self.transformation['soda'] = np.zeros((self.dim_info['Sli']['len'], 4, 4))
        flag_imagescan = np.array([mdb.is_image_scan() for mdb in self.twixobj['mdb']])
        slice_no = np.array([mdb.cSlc for mdb in self.twixobj['mdb']])
        for cSlc in range(self.dim_info['Sli']['len']):
            ind = np.where((slice_no == cSlc) & flag_imagescan)[0].tolist()[0]
            SliceData = self.twixobj['mdb'][ind].mdh.SliceData    
            position = [SliceData.SlicePos.Sag, SliceData.SlicePos.Cor, SliceData.SlicePos.Tra]
            dcm = R.from_quat(np.roll(SliceData.Quaternion, -1)).as_matrix()  #   {SliceData.SlicePos}')
            self.transformation['soda'][cSlc,:,:] =  np.row_stack((np.column_stack((dcm, position)), [0,0,0,1]))
            
        # rotation matrix should be identical for all slices, so mean does not matter here, but offcenter will be averaged
        self.transformation['mat44'] = self.transformation['soda'].mean(axis=0)


    ##########################################################
    # Save a custom volume as nifti
    # def make_nifti(self, volume:torch.Tensor, filename):        
    #     # input must have same dimension ordering as dim_info
    #     check_dims = [self.dim_info['ro']['ind'], self.dim_info['pe1']['ind'], self.dim_info['pe2']['ind'], self.dim_info['slc']['ind'], self.dim_info['rep']['ind']] # the dimensions that must be checked
    #     ds = [self.dim_size[y] for y in check_dims]
    #     vs = [volume.shape[y] for y in check_dims]
    #     if vs != ds and vs[0]*2 != ds[0]: # second condition is to account for oversampling
    #         print(f"Size mismatch (RO, PE1, PE2, SLC, REP)! {vs } vs {ds}")
    #         return

    #     vs = [y for y in volume.shape if y!=1]
    #     if len(vs) > 4 :
    #         print(f"{len(vs)}D data is not supported")
    #         return

    #     volume = torch.flip(volume, dims=[self.dim_info['ro']['ind'], 
    #                                       self.dim_info['pe1']['ind'],
    #                                       self.dim_info['slc']['ind']])
    #     #
    #     # creating permute indices
    #     #
    #     # prmt_ind = np.arange(0, len(self.dim_info), 1, dtype=int)
    #     # # swap ro and pe1
    #     # prmt_ind[[self.dim_info['ro']['ind'], self.dim_info['pe1']['ind']]] = prmt_ind[[self.dim_info['pe1']['ind'], self.dim_info['ro']['ind']]] 
    #     # # move cha to end
    #     # icha = self.dim_info['cha']['ind']
    #     # prmt_ind = np.hstack([prmt_ind[:icha], prmt_ind[icha+1:], prmt_ind[icha]])
    
    #     # volume = np.transpose(volume, prmt_ind)

    #     volume = volume.swapaxes(self.dim_info['ro']['ind'], self.dim_info['pe1']['ind']).moveaxis(self.dim_info['cha']['ind'], -1).squeeze()
       
    #     #
    #     # save to file
    #     #
    #     img = nib.Nifti1Image(volume.numpy(), self.nii_affine)
    #     nib.save(img, filename)
