import os
import torch
import twixtools
import numpy as np
from .protocol import protocol_parse
from .reco_tools import POCS, coil_combination, calc_coil_sensitivity
from .transformation import calc_nifti_affine

lib_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils', 'lib')

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
    transformation = None
    nii_affine     = None
    slice_reorder_ind = None
    
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
        
        self.transformation = dict()
        self.prot = protocol_parse(self.twixmap)
        self._extract_transformation()        


    def __str__(self):
        s = f"\n{self.__class__.__module__}.{self.__class__.__qualname__}:\n"
        s += f"  Scans: {self.scans}\n"
        s += f"  Dims: {list(self.dim_info.keys())}\n"
        s += f"  Size: {self.dim_size}\n"
        s += f"  Resolution (x y z): {self.prot.res['x']} {self.prot.res['y']} {self.prot.res['z']}\n"
        s += f"  Partial Fourier (ro, pe, pe3d): {self.prot.isPartialFourierRO} {self.prot.isPartialFourierPE1} {self.prot.isPartialFourierPE2}\n"
        s += f"  Parallel Imaging: {self.prot.isParallelImaging}\n"
        s += f"  Acceleration Factor: {self.prot.acceleration_factor}\n"
        return s


    def runReco(self, method_sensitivity='caldir'):     
        torch.cuda.empty_cache()   

        kspace = self._getkspace()        
        kspace = self.correct_scan_size(kspace, scantype='image')
        
        # Partial Fourier?
        if self.prot.isPartialFourierRO:
            kspace = POCS(kspace, dim_enc=self.dim_enc, dim_pf=self.dim_info['Col']['ind'])
        if self.prot.isPartialFourierPE1:
            kspace = POCS(kspace, dim_enc=self.dim_enc, dim_pf=self.dim_info['Lin']['ind'])
        if self.prot.isPartialFourierPE2:
            kspace = POCS(kspace, dim_enc=self.dim_enc, dim_pf=self.dim_info['Par']['ind'])

        # Parallel Imaging?
        if self.prot.isParallelImaging:
            self.twixmap['refscan'].flags['zf_missing_lines'] = not self.prot.isRefScanSeparate 
            acs = torch.from_numpy(self.twixmap['refscan'][:])
            acs = self.correct_scan_size(acs, scantype='refscan')
        else:
            # picking the 0th element of the free dimensions 
            acs = kspace.clone()
            for dim_free in self.dim_free:
                acs = acs.index_select(self.dim_info[dim_free]['ind'], torch.Tensor([0]).int()) 

        if method_sensitivity is not None:
            coils_sensitivity = calc_coil_sensitivity(acs, dim_enc=self.dim_enc, method=method_sensitivity)
            self.img = coil_combination(kspace, coil_sens=coils_sensitivity, dim_enc=self.dim_enc)
        else:
            self.img = coil_combination(kspace, coil_sens=None, dim_enc=self.dim_enc, rss=True)


    ##########################################################
    def correct_scan_size(self, kspace:torch.Tensor, scantype='image'):
        # Note: this function suppose oversampling is removed
        import torch.nn.functional as F
        print(f'kspace uncorrected shape: {kspace.shape}, scantype: {scantype}')
        col_diff = self.hdr['Meas']['iRoFTLength']//2 - self.twixmap[scantype].kspace_center_col
        lin_diff = self.hdr['Meas']['iPEFTLength']//2 - self.twixmap[scantype].kspace_center_lin
        par_diff = self.hdr['Meas']['i3DFTLength']//2 - self.twixmap[scantype].kspace_center_par
        if kspace.shape[self.dim_info['Lin']['ind']] == self.hdr['Meas']['iPEFTLength']:
            lin_diff = 0
        if kspace.shape[self.dim_info['Par']['ind']] == self.hdr['Meas']['i3DFTLength']:
            par_diff = 0
        # print(f'iRoFTLength: {self.hdr["Meas"]["iRoFTLength"]}, iPEFTLength: {self.hdr["Meas"]["iPEFTLength"]}, i3DFTLength: {self.hdr["Meas"]["i3DFTLength"]}')
        # print(f'kspace_center_col: {self.twixmap[scantype].kspace_center_col}, kspace_center_lin: {self.twixmap[scantype].kspace_center_lin}, kspace_center_par: {self.twixmap[scantype].kspace_center_par}')
        # print(f'col_diff: {col_diff}, lin_diff: {lin_diff}, par_diff: {par_diff}')
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
        kspace = F.pad(kspace, pad, 'constant', 0)
        print(f'kspace corrected shape  : {kspace.shape}, scantype: {scantype}')
        return kspace


    ##########################################################

    def _getkspace(self, scantype='image'):
        print('Extracting kspace...')
        kspace = torch.from_numpy(self.twixmap[scantype][:])
        return kspace.index_select(self.dim_info['Sli']['ind'], torch.from_numpy(self.slice_reorder_ind))

    def _slice_reorder(self):
        # sort slices based on position (e.g., when acqusition is interleaved).
        unsorted_order = np.zeros((self.dim_info['Sli']['len']))
        transform_inv = np.linalg.inv(self.transformation['mat44'])
        for cSlc in range(self.dim_info['Sli']['len']):
            p = transform_inv @ self.transformation['soda'][cSlc,:,3]
            unsorted_order[cSlc] = p[2]  # z position before transformation

        self.slice_reorder_ind = np.argsort(unsorted_order)
        self.transformation['soda'] = self.transformation['soda'][self.slice_reorder_ind,...]


    def _extract_transformation(self):
        from scipy.spatial.transform import Rotation

        self.transformation['soda'] = np.zeros((self.dim_info['Sli']['len'], 4, 4))
        flag_imagescan = np.array([mdb.is_image_scan() for mdb in self.twixobj['mdb']])
        slice_no = np.array([mdb.cSlc for mdb in self.twixobj['mdb']])
        for cSlc in range(self.dim_info['Sli']['len']):
            ind = np.where((slice_no == cSlc) & flag_imagescan)[0].tolist()[0]
            SliceData = self.twixobj['mdb'][ind].mdh.SliceData    
            position = [SliceData.SlicePos.Sag, SliceData.SlicePos.Cor, SliceData.SlicePos.Tra]
            dcm = Rotation.from_quat(np.roll(SliceData.Quaternion, -1)).as_matrix()  #   {SliceData.SlicePos}')
            self.transformation['soda'][cSlc,:,:] =  np.row_stack((np.column_stack((dcm, position)), [0,0,0,1]))
        
        # rotation matrix should be identical for all slices, so mean does not matter here, but offcenter will be averaged
        self.transformation['mat44'] = self.transformation['soda'].mean(axis=0)
        self._slice_reorder()
        
        # Here we swap order of x and y because we would like to have PE and RO as the first and the second dimensions, respectively, in nifti file.
        # flipping volume along its center 
        res =self.prot.res
        flip_affine = np.diag([-1, -1, -1 if self.prot.is3D else 1, 1]) 
        flip_affine[:,-1] = [res['y'], res['x'], res['z'] if self.prot.is3D else 0, 1]

        affine = calc_nifti_affine(self.transformation['mat44'], self.prot.fov, self.prot.res, self.prot.slice_thickness)
        self.transformation['nii_affine'] = affine @ flip_affine


    ##########################################################
    # Save a custom volume as nifti
    # input must have same dimension ordering as dim_info
    def write_nifti(self, volume:torch.Tensor, filename): 
        import nibabel as nib 
        if volume.squeeze().ndim > 4 :
            print(f"{volume.ndim}D data is not supported")
            return
        
        dim = self.dim_info
        # permute to match [PE, RO, SLC/PAR, REP]
        perm_ind = [dim['Lin']['ind'], dim['Col']['ind'], dim['Par']['ind'], dim['Sli']['ind'], dim['Rep']['ind'], dim['Cha']['ind']]
        perm_ind = perm_ind + [d['ind'] for d in dim.values() if d['ind'] not in perm_ind]
        volume = volume.permute(perm_ind) 
        volume = volume.squeeze()
        if self.dim_info['Par']['len'] == 1 and self.dim_info['Sli']['len'] == 1:
            volume = volume.unsqueeze(dim=2)

        img = nib.Nifti1Image(volume.detach().cpu().numpy(), self.transformation['nii_affine'])
        nib.save(img, filename)

        