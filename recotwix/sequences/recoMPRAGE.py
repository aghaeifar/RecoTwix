import os
import ctypes
import torch
import numpy as np
from math import pi as PI
from scipy import ndimage
from recotwix import recotwix, kspace_to_image, lib_folder


class recoMPRAGE(recotwix):
    img_mag  = torch.empty([1])   

    def __init__(self, filename=None, device='cpu', method_sensitivity='caldir'):
        super().__init__(filename, device)  
        self.runReco(method_sensitivity)

    def __str__(self):
        s = super().__str__()
        # s += f"  TE = {self.prot.TE} μs\n"
        return s
        
    def runReco(self, method_sensitivity='caldir'):
        super().runReco(method_sensitivity=method_sensitivity)  
        self.img_mag = torch.abs(self.img)


                    

    
    
