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
    

    def __init__(self, volume_structure=None, res=None, thickness=None) -> None:
        if volume_structure is None:
            return
        v = volume_structure
        self._norm = [v['sNormal'].get('dSag',0), v['sNormal'].get('dCor',0), v['sNormal'].get('dTra',0)]
        self._rot = v.get('dInPlaneRot', 0)
        self._pos = [v.get('sPosition', {}).get('dSag',0), v.get('sPosition', {}).get('dCor',0), v.get('sPosition', {}).get('dTra',0)]
        self._fov = [v['dReadoutFOV'], v['dPhaseFOV'], v['dThickness']]
        self._res = res      
        self._thickness = thickness  

        dcm = T.calc_dcm(self._norm[0], self._norm[1], self._norm[2], self._rot)
        self._transformation = T.calc_tranformation_matrix(dcm, self._pos)
        self._affine = T.calc_nifti_affine(self._transformation, self._fov, self._res, thickness)
    
    @property
    def affine(self):
        return self._affine
    
    @property
    def transformation(self):
        return self._transformation
    

class adjustment_volume(volume):
    def __init__(self, xprot) -> None:
        super().__init__(xprot['sAdjData']['sAdjVolume'])  


class slice_volume(volume):
    def __init__(self, xprot) -> None:
        if xprot.get('sSliceArray', None) is None:
            return
        if xprot['sSliceArray'].get('asSlice', None) is None:
            return

        # transformation per slice
        self._slice_volume = list()
        res  = [xprot['sKSpace']['lBaseResolution'], xprot['sKSpace']['lPhaseEncodingLines'], xprot['sKSpace']['lPartitions']]
        res[2] = res[2] if xprot['sKSpace']['ucDimension'] == 4 else 1 # in case of 2D scans, lPartitions is not valid
        positions = list()
        for SlcVol in xprot['sSliceArray']['asSlice']:                        
            self._slice_volume.append(volume(SlcVol, res)) 
            positions.append([SlcVol.get('sPosition', {}).get('dSag',0) , SlcVol.get('sPosition', {}).get('dCor',0) , SlcVol.get('sPosition', {}).get('dTra',0)])
        
        # transformation for the whole volume
        res[2] = res[2] if xprot['sKSpace']['ucDimension'] == 4 else xprot['sSliceArray']['lSize'] # in case of 2D scans
        SlcVol = xprot['sSliceArray']['asSlice'][0]
        slice_thickness = SlcVol['dThickness']
        pos_mean = np.mean(positions, axis=0).tolist()
        SlcVol['sPosition'] = {'dSag': pos_mean[0], 'dCor': pos_mean[1], 'dTra': pos_mean[2]}
        SlcVol['dThickness'] = math.dist(positions[0], positions[-1]) + slice_thickness
        super().__init__(SlcVol, res, slice_thickness)
        # print([SlcVol['sNormal'].get('dSag',0), SlcVol['sNormal'].get('dCor',0), SlcVol['sNormal'].get('dTra',0)], SlcVol.get('dInPlaneRot', 0))
        # print([SlcVol['sPosition'].get('dSag',0), SlcVol['sPosition'].get('dCor',0), SlcVol['sPosition'].get('dTra',0)])


    def __getitem__(self, index):
        if index >= len(self._slice_volume):
            raise IndexError(f'Item out of range. Slice volume has only {len(self._slice_volume)} items but {index} is asked.')
        return self._slice_volume[index]
    
    def __len__(self):
        return len(self._slice_volume)


class ptx_volume():
    
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