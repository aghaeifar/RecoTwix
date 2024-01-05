
import torch
from recotwix import recotwix

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

class recoB1TFLcp(recotwix):
    nTx      = 0
    img_cp   = torch.empty([1]) # CP map
    img_fa   = torch.empty([1]) # FA map
    img_b    = torch.empty([1]) # B1 map, nT per Volt
    img_mask = torch.empty([1])
    params   = {'pulse_duration':0, 'pulse_integral':0}
    seqTxScaleFactor = torch.empty([1])

    def __init__(self, filename=None, device='cuda'):
        super().__init__(filename, device)        
        self.parseHeader()
        self.runReco()    

    def __str__(self):
        s = super().__str__()
        s += f"  Pulse Duration(sec.) and Integral(volt*sec.): {self.params['pulse_duration']}, {self.params['pulse_integral']}\n" + \
             f"  Sequence TxScaleFactor = {self.seqTxScaleFactor.tolist()}\n"
        return s
    
    def parseHeader(self):
        ICEProgramPara = self.hdr['Meas']['alICEProgramPara']
        self.params['pulse_duration'] = ICEProgramPara[14] * 1e-6 # convert to seconds
        self.params['pulse_integral'] = ICEProgramPara[16] * 1e-6 # convert to Volt * second
        self.nTx = self.hdr['Meas']['lNumberOfTXCalibDateTime']
        
        self.seqTxScaleFactor = torch.complex(torch.zeros(self.nTx), torch.zeros(self.nTx)) # filling with zero, just in case some channels are missing in the header
        for i in range(self.nTx):
            self.seqTxScaleFactor[i].real = self.hdr['MeasYaps']['sTXSPEC']['aTxScaleFactor'][i].get('dRe', 0)
            self.seqTxScaleFactor[i].imag = self.hdr['MeasYaps']['sTXSPEC']['aTxScaleFactor'][i].get('dIm', 0)


    def runReco(self, method_sensitivity='caldir'):
        super().runReco(method_sensitivity=method_sensitivity)

        ida_ind = self.dim_info['Ida']['ind']
        sat     = self.img.index_select(ida_ind, torch.Tensor([self.dim_info['Ida']['len']-1]).int())
        ref     = self.img.index_select(ida_ind, torch.Tensor(torch.arange(0, self.dim_info['Ida']['len']-1)))
        self.fa_map  = torch.arccos( sat / ref).abs().rad2deg()        
