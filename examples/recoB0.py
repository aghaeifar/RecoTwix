import os
import ctypes
import torch
import numpy as np
from recotwix import recotwix, kspace_to_image
from math import pi as PI

class recoB0(recotwix):
    img_b0   = torch.empty([1])
    img_mag  = torch.empty([1])    
    img_mask = torch.empty([1])

    def __init__(self, filename=None, device='cpu'):
        super().__init__(filename, device)  
        self.runReco()

    def __str__(self):
        s = super().__str__()
        s += f"  TE = {self.prot.TE} μs\n"
        return s
        
    def runReco(self, method_sensitivity='caldir'):
        if self.dim_info['Eco']['len'] < 2:
            print(f"Error!\033[93mAt least two echoes are expected!\033[0m")
            return

        if self.prot.isParallelImaging:
            super().runReco(method_sensitivity=method_sensitivity)  
            self.img_mag = torch.abs(self.img)
        else:
            kspace = torch.from_numpy(self.twixmap['image'][:])
            kspace = self.correct_scan_size(kspace, scantype='image')
            self.img = kspace_to_image(kspace, dim_enc=self.dim_enc, dim_loop=self.dim_info['Cha']['ind'])
            self.img_mag = torch.sum(torch.abs(self.img)**2, self.dim_info['Cha']['ind'], keepdims=True)

        print(f"Calculating B0 map. \u0394TE = {(self.prot.TE[1] - self.prot.TE[0])} μs")

        dim_eco = self.dim_info['Eco']['ind']
        dim_rep = self.dim_info['Rep']['ind']
        idx = torch.tensor([0, 1])
        if self.dim_info['Rep']['len'] == 1 : #  % regular B0 mapping, b0map = (Eco2 - Eco1)                        
            self.img_b0 = self.img_mag.index_select(dim_eco, idx[1]) * self.img.index_select(dim_eco, idx[1]) * \
                          self.img_mag.index_select(dim_eco, idx[0]) * self.img.index_select(dim_eco, idx[0]).conj()  

        else: # shims basis-map, b0map = (Eco2Repn - Eco1Repn) - (Eco2Rep1 - Eco1Rep1)
            self.img_b0 = self.img_mag.index_select(dim_eco, idx[1]) * self.img.index_select(dim_rep, idx[1])        * \
                          self.img_mag.index_select(dim_eco, idx[0]) * self.img.index_select(dim_rep, idx[0]).conj() * \
                          self.img_mag.index_select(dim_eco, idx[1]).index_select(dim_rep, idx[0]) * self.img.index_select(dim_eco, idx[1]).index_select(dim_rep, idx[0]).conj() * \
                          self.img_mag.index_select(dim_eco, idx[0]).index_select(dim_rep, idx[0]) * self.img.index_select(dim_eco, idx[0]).index_select(dim_rep, idx[0])
            self.img_b0.moveaxis(dim_rep, 0)[0,...] = torch.tensor(complex(1,0))

        self.img_b0 = torch.angle(self.img_b0.sum(dim=self.dim_info['Cha']['ind'], keepdims=True)) # sum over coils
        self.img_mag = self.img_mag.index_select(self.dim_info['Eco']['ind'], idx[0])
        self.img = torch.empty((1)) # save memory

    ##########################################################
    def get_b0hz(self, b0_uw:torch.Tensor = None, offset = 0):
        if b0_uw is None:
            b0_uw = self._unwrap_b0()
        if b0_uw.shape != self.img_b0.shape:
            print(f"\033[93mUnwrapped image is not yet calculated. \033[0m")
            return None
        dTE = (self.prot.TE[1] - self.prot.TE[0]) * 1e-6 # s
        return (b0_uw + offset) / dTE / (2*PI)

    ##########################################################
    def _unwrap_b0(self):
        print('Unwrapping B0...')
        b0_size = (*self.img_b0.squeeze().shape,) + (1,) # add singleton dimensions
        if len(b0_size) != 4 and len(b0_size) != 5:
            print(f'Only 3D or 4D data is supported for unwrapping. Input shape is {b0_size}')
            return None

        dir_path = os.path.dirname(os.path.realpath(__file__))
        handle   = ctypes.CDLL(os.path.join(dir_path, "..", "utils", "lib", "libunwrap_b0.so")) 
        handle.unwrap_b0.argtypes = [np.ctypeslib.ndpointer(np.float32, ndim=self.img_b0.ndim, flags='F'),
                                     np.ctypeslib.ndpointer(np.float32, ndim=self.img_b0.ndim, flags='F'),
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        b0 = self.img_b0.numpy().copy(order='F')        
        b0_uw = np.zeros_like(b0)
        handle.unwrap_b0(b0, b0_uw, *b0_size[:4]) # [:4] -> always 4D shape to unwrapper
        return (torch.from_numpy(b0_uw.copy(order='C'))) 

                    

    
    
