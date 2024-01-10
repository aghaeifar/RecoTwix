import os
import math
import numpy as np
from . import transformation as T
import twixtools.twixprot as twixprot


class volume:
    _norm = None # [norm_sag, norm_cor, norm_tra]
    _rot = None # inplane rotation (radians)
    _pos = None # offset from origin (mm)
    _fov = None # FoV (mm)
    _res = None # resolution (voxels)
    _thickness = None # slice thickness (mm)
    _affine = None
    _transformation = None    
    _name = None

    def __init__(self, volume_structure=None, res=None, thickness=None, name=None, res2fov_ratio=1) -> None:
        
        v = volume_structure
        self._norm = [v['sNormal'].get('dSag',0), v['sNormal'].get('dCor',0), v['sNormal'].get('dTra',0)]
        self._rot = v.get('dInPlaneRot', 0)
        self._pos = [v.get('sPosition', {}).get('dSag',0), v.get('sPosition', {}).get('dCor',0), v.get('sPosition', {}).get('dTra',0)]
        self._fov = {'x':v['dReadoutFOV'], 'y':v['dPhaseFOV'], 'z':v['dThickness']}
        if res is None:
            res = {key: int(value*res2fov_ratio) for key, value in self._fov.items()}

        self._res = res  
        self._thickness = thickness if thickness is not None else (self._fov['z'] / self._res['z'])

        dcm = T.calc_norm2dcm(self._norm[0], self._norm[1], self._norm[2], self._rot)
        self._transformation = T.calc_tranformation_matrix(dcm, self._pos)
        self._affine = T.calc_nifti_affine(self._transformation, self._fov, self._res, self._thickness)
        self._name = name

    def write_nifti(self, filename):
        import nibabel as nib 
        vol = np.ones((self._res['y'], self._res['x'], self._res['z']))
        img = nib.Nifti1Image(vol, self._affine)
        nib.save(img, filename)
            
    @property
    def affine(self):
        return self._affine
    
    @property
    def transformation(self):
        return self._transformation
    
    @property
    def shape(self):
        return (self._res['y'], self._res['x'], self._res['z'])
    
    @property
    def fov(self):
        return (self._fov['y'], self._fov['x'], self._fov['z'])
    

class adjustment_volume(volume):
    def __init__(self, xprot) -> None:
        super().__init__(xprot['sAdjData']['sAdjVolume'], res2fov_ratio=0.5)  


class slice_volume():
    _slice_volume = None

    def __init__(self, xprot) -> None:
        if xprot.get('sSliceArray', None) is None:
            return
        if xprot['sSliceArray'].get('asSlice', None) is None:
            return

        # transformation per slice
        self._slice_volume = list()
        res  = {'x':xprot['sKSpace']['lBaseResolution'], 'y':xprot['sKSpace']['lPhaseEncodingLines'], 'z':xprot['sKSpace']['lPartitions']}
        res['z'] = res['z'] if xprot['sKSpace']['ucDimension'] == 4 else 1 # in case of 2D scans, lPartitions is not valid
        positions = list()
        for SlcVol in xprot['sSliceArray']['asSlice']:                      
            self._slice_volume.append(volume(SlcVol, res=res)) 
            positions.append([SlcVol.get('sPosition', {}).get('dSag',0) , SlcVol.get('sPosition', {}).get('dCor',0) , SlcVol.get('sPosition', {}).get('dTra',0)])
        
    def __getitem__(self, index):
        if index >= len(self._slice_volume):
            raise IndexError(f'Item out of range. Slice volume has only {len(self._slice_volume)} items but {index} is asked.')
        return self._slice_volume[index]
    
    def __len__(self):
        return len(self._slice_volume)


class ptx_volume():
    _ptx_volume = None

    def __init__(self, xprot) -> None:
        if xprot.get('sPTXData', None) is None:
            return
        if xprot['sPTXData'].get('asPTXVolume', None) is None:
            return
        
        self._ptx_volume = list()
        for pTxVol in xprot['sPTXData']['asPTXVolume']:            
            self._ptx_volume.append(volume(pTxVol))  

    def __getitem__(self, index):
        if index >= len(self._ptx_volume):
            raise IndexError(f'Item out of range. PTx volume has only {len(self._ptx_volume)} items but {index} is asked.')
        return self._ptx_volume[index]
    
    def __len__(self):
        return len(self._ptx_volume)


class prot_volumes:
    xprot = None
    _slc = None
    _ptx = None
    _adj = None

    def __init__(self, param=None) -> None:
        if param is None:
            return
        
        if isinstance(param, dict):
            if param.get('hdr', None) is not None:
                self.xprot = param['hdr']['MeasYaps']
            elif param.get('MeasYaps', None) is not None:
                self.xprot = param['MeasYaps']
            elif param.get('ulVersion', None) is not None:
                self.xprot = param
            else:
                raise ValueError(f'Unknown parameter type: {type(param)}')

        elif os.path.isfile(param):
            with open(param, "r") as my_file:
                str = my_file.read()
                self.xprot = twixprot.parse_buffer(str)
        else:
            raise ValueError(f'Unknown parameter type: {type(param)}')
        
        self._slc = slice_volume(self.xprot)
        self._ptx = ptx_volume(self.xprot)
        self._adj = adjustment_volume(self.xprot)

    @property
    def slc(self):
        return self._slc

    @property
    def ptx(self):
        return self._ptx

    @property
    def adj(self):
        return self._adj    