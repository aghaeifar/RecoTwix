
import torch
from recotwix import recotwix


#
# This class implementes reconstruction of the B1 mapping technique as described in https://onlinelibrary.wiley.com/doi/10.1002/mrm.29459
#

GAMMA_RAD       = 267.52218744e6  # rad/(sT)
GAMMA_HZ        = 42.577478518e6  # Hz/T

HAS_ABSOLUTE_MAP = 1<<0
HAS_RELATIVE_MAP = 1<<1
HAS_B0_MAP       = 1<<3
HAS_SINC_SAT     = 1<<4

PTX_MODE_CP     = 1
PTX_MODE_ONEON  = 2
PTX_MODE_ONEOFF = 3
PTX_MODE_ONEINV = 4

mapping_mode = {PTX_MODE_CP       :{'diag':complex( 1.0, 0.0), 'offdiag':complex(1.0, 0.0)}, \
                PTX_MODE_ONEON    :{'diag':complex( 1.0, 0.0), 'offdiag':complex(0.0, 0.0)}, \
                PTX_MODE_ONEOFF   :{'diag':complex( 0.0, 0.0), 'offdiag':complex(1.0, 0.0)}, \
                PTX_MODE_ONEINV   :{'diag':complex(-1.0, 0.0), 'offdiag':complex(1.0, 0.0)}}

class recoB1TFLpTx(recotwix):
    nTx      = 0
    img_cp   = torch.empty([1]) # CP map
    img_fa   = torch.empty([1]) # FA map
    img_b    = torch.empty([1]) # B1 map, nT per Volt
    img_mask = torch.empty([1])
    params   = {'operating_mode':0, 'nTx_abs':0, 'abs_mnode':0, 'pulse_duration':0, 'rel_mode':0, 'nTx_rel':0, 'pulse_integral':0, 'b0_dTE':0}
    seqTxScaleFactor = torch.empty([1])

    def __init__(self, filename=None, device='cuda'):
        super().__init__(filename, device)        
        self.parseHeader()
        self.runReco()    

    def __str__(self):
        s = super().__str__()
        s += f"  Operating Mode: {self.params['operating_mode']:05b}\n" + \
             f"  Absolute Mode : {self.params['abs_mode']}\n" + \
             f"  Relative Mode : {self.params['rel_mode']}\n" + \
             f"  Num. Tx (Abs, Rel) : {self.params['nTx_abs']}, {self.params['nTx_rel']}\n" + \
             f"  Pulse Duration(sec.) and Integral(volt*sec.): {self.params['pulse_duration']}, {self.params['pulse_integral']}\n" + \
             f"  Sequence TxScaleFactor = {self.seqTxScaleFactor.tolist()}\n"
        
        return s
    
    def parseHeader(self):
        ICEProgramPara = self.hdr['Meas']['alICEProgramPara']
        self.params['operating_mode'] = ICEProgramPara[10]
        self.params['nTx_abs']        = ICEProgramPara[11]
        self.params['abs_mode']       = ICEProgramPara[12]
        self.params['pulse_duration'] = ICEProgramPara[14] * 1e-6 # convert to seconds
        self.params['rel_mode']       = ICEProgramPara[15]
        self.params['nTx_rel']        = ICEProgramPara[16]
        self.params['pulse_integral'] = ICEProgramPara[16] * 1e-6 # convert to Volt * second
        self.params['b0_dTE']         = ICEProgramPara[17] * 1e-6 # convert to seconds
        self.nTx = self.hdr['Meas']['lNumberOfTXCalibDateTime']
        
        if self.params['nTx_abs'] != self.params['nTx_rel']:
            print(f'\033[93mNumber of absolute and relative transmit channels do not match: {self.params["nTx_abs"]} vs {self.params["nTx_rel"]}\033[0m')
            return
        if self.params['operating_mode'] & HAS_ABSOLUTE_MAP == 0 or self.params['operating_mode'] & HAS_RELATIVE_MAP == 0:
            print(f'\033[93mAbsolute and relative maps must present.\033[0m')
            return  
    

        self.seqTxScaleFactor = torch.complex(torch.zeros(self.nTx), torch.zeros(self.nTx)) # filling with zero, just in case some channels are missing in the header
        for i in range(self.nTx):
            self.seqTxScaleFactor[i].real = self.hdr['MeasYaps']['sTXSPEC']['aTxScaleFactor'][i].get('dRe', 0)
            self.seqTxScaleFactor[i].imag = self.hdr['MeasYaps']['sTXSPEC']['aTxScaleFactor'][i].get('dIm', 0)


    def runReco(self, method_sensitivity='caldir'):
        super().runReco(method_sensitivity=method_sensitivity)

        rep_ind = self.dim_info['Rep']['ind']
        sat     = self.img.index_select(rep_ind, torch.arange(0, self.nTx))
        ref     = self.img.index_select(rep_ind, torch.tensor([self.nTx]))
        rel_map = self.img.index_select(rep_ind, torch.arange(self.nTx+1, 2*self.nTx+1))
        fa_map  = torch.arccos( sat / ref).abs().rad2deg()        
        fa_map  = fa_map * rel_map / rel_map.abs() 
        fa_map  = torch.reshape(fa_map.moveaxis(rep_ind, 0), (self.nTx, -1))  # reshape to 2D matrix, [nTx, nVoxels]

        # create the matrix of scales factors
        scale_factor = torch.full((self.nTx, self.nTx), mapping_mode[self.params['abs_mode']]['offdiag'])
        scale_factor.fill_diagonal_(mapping_mode[self.params['abs_mode']]['diag'])
        scale_factor = scale_factor * self.seqTxScaleFactor
        
        self.img_fa = torch.linalg.solve(scale_factor, fa_map)# solve the linear system of equations
        # restore the original shape
        rshp = list(sat.shape)
        rshp.insert(0, rshp.pop(rep_ind))
        self.img_fa = torch.moveaxis(self.img_fa.reshape(rshp), 0, rep_ind) 
        self.img_b1 = self.img_fa / (GAMMA_HZ * 360.0 * self.params['pulse_integral']) * 1e9 # convert to nT/Volt 
        self.img_cp = torch.sum(self.img_fa * self.seqTxScaleFactor.view((-1,)+(1,)*(self.img_fa.dim()-rep_ind-1)), dim=rep_ind, keepdim=True) # sum over all transmit channels
        