#!/usr/bin/env python
# coding: utf-8

# imports
import twixtools
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import time

from scipy.io import savemat, loadmat
import scipy.io as spio


from mrtools import remove_RO_oversampling, twix_get_image_data, twix_get_acs_data
from mrtools import fftnd, ifftnd, ifftnd_split, fftnd_split, POCS
from grappa import grappa3d_conv, create_grappa_weight_set
from coilcombine_cpp import sum_of_squares, adaptive_combine

# define
def reco_dbTFLb1(infile,outfile,dev=torch.device('cpu')):

    start_time = time.time()

    #%% read raw data
    twix = twixtools.read_twix(infile);
    meta = dbMeta(twix)

    #%% Generate k-space data

    # get twix metadata
    al_ICE_para = twix[-1]['hdr']['Meas']['alICEProgramPara']

    settings = al_ICE_para[10]
    nModes = al_ICE_para[11]
    presatMode = al_ICE_para[12]
    tDelay = al_ICE_para[13]
    pulse_dur = al_ICE_para[14] * 1e-6
    presatMode = al_ICE_para[15]
    nRelModes = al_ICE_para[16]
    AmplitudeIntegralVs = al_ICE_para[17] / 1e6
    TEshift = al_ICE_para[18] / 1e6
    VersionNumber = al_ICE_para[19]

    # from matlab, adapt
    #isPrep = bitget(settings,1);
    #hasRelMaps = bitget(settings,2);
    #isHybridMap = bitget(settings,3);
    #hasB0map = bitget(settings,4);


    #%% generate k-space
    kspace = twix_get_image_data(twix).to(dev)
    kspace_acs = twix_get_acs_data(twix).to(dev)


    #%% remove RO oversampling
    kspace_acs = remove_RO_oversampling(kspace_acs)
    kspace = remove_RO_oversampling(kspace)

    #%% Handle asymmetric echo

    if twix[-1]['hdr']['MeasYaps']['sKSpace']['ucAsymmetricEchoMode']:
        for i in range(kspace.shape[4]):
            kspace[:,:,:,:,i] = POCS(kspace[:,:,:,:,i],dim=1)


    #%% Calculate GRAPPA kernel

    # get acceleration factor from twix header
    R_accel = (twix[-1]['hdr']['MeasYaps']['sPat']['lAccelFactPE'],
               twix[-1]['hdr']['MeasYaps']['sPat']['lAccelFact3D'])

    print('calculating GRAPPA kernel... ',end='')
    kernel_size = None # None=automatic (according to some heuristic)
    lambd = 0.01 # Tikhonov reg for weight fitting
    delta = 0 # CAIPI shift (probably not fully working yet)
    t1 = time.time()
    # target matrix size
    matSz = kspace.shape[1:4]

    # create grappa kernel
    wsKernel,sbl = create_grappa_weight_set(
        kspace_acs[:,:,:,:,0], R_accel[0], R_accel[1], delta, matSz, 
        lambd=lambd, kernel_size=kernel_size
        )

    el = time.time() - t1
    print(f'Calculating GRAPPA kernel took {el:.3f} s')


    #%% Apply GRAPPA kernel
    # apply grappa kernel to all images
    print('applying GRAPPA kernel...  ',end='')
    t1 = time.time()
    torch.cuda.empty_cache()
    n_rep = kspace.shape[4]
    reco_grappa = torch.zeros(kspace.shape,dtype=torch.complex64,
                              device=kspace.device)
    for i in range(n_rep):
        print('{}/{}  '.format(i+1,n_rep),end='')
        reco_grappa[:,:,:,:,i] = grappa3d_conv(kspace[:,:,:,:,i], wsKernel,
                                               sbl,R_accel[0], R_accel[1])
    el = time.time() - t1
    print(f'{el:.3f} s')


    #%% Apply gaussian filter to reduce ringing
    # right now this is set rather aggressive. Increase sigma to reduce effect


    sigma=0.4
    filtD1 = gaussian(reco_grappa.shape[1],sigma=sigma)#.reshape(1,-1)
    filtD2 = gaussian(reco_grappa.shape[2],sigma=sigma)#.reshape(1,1,-1)
    filtD3 = gaussian(reco_grappa.shape[3],sigma=sigma)#.reshape(1,1,1,-1)
    filt3D = torch.ones(reco_grappa.shape[1:4])
    filt3D = torch.einsum('ghj,g->ghj',filt3D,filtD1)#filt3D.mul(filtD1,1)# * filtD3
    filt3D = torch.einsum('ghj,h->ghj',filt3D,filtD2)#filt3D.mul(filtD1,1)# * filtD3
    filt3D = torch.einsum('ghj,j->ghj',filt3D,filtD3)#filt3D.mul(filtD1,1)# * filtD3

    filt3D = filt3D.to(reco_grappa.device)
    reco_grappa = torch.einsum('kghjy,ghj->kghjy',reco_grappa,filt3D)
    del(filt3D)
    
    #%% fourier-transform (to image-space)

    image = fftnd(reco_grappa,(1,2,3))
    
    #%% Coil combination with adaptive combine

    t1 = time.time()
    imSum = torch.zeros(image.shape[1:],dtype=torch.complex64)
    imSum[:,:,:,0],ws = adaptive_combine(image[:,:,:,:,0].cpu())
    for i in range(1,reco_grappa.shape[4]):
        imSum[:,:,:,i],_ = adaptive_combine(image[:,:,:,:,i].cpu(),ws)
    el = time.time() - t1
    print(f'Adaptive combine took {el:.3f} s')


    # #%% Determine brain mask and tissue mask
    # # mask in matlab
    # ## 1: export into matlab file
    # imRot = imSum[...,nModes].permute([2,1,0]).flip(1)
    # mat = {"img":imRot.numpy()}
    # savemat('image.mat',mat)
    # ## 2.: call matlab
    # mat_cmd = 'matlab -nosplash -nodesktop -noFigureWindows -r "try; classify_matlab(); catch; end; quit"'
    # os.system(mat_cmd)
    # ## 3.: read result
    # r = loadmat('result.mat')
    # brain_mask = torch.from_numpy(r['brain_mask']).to(torch.bool).flip(1).permute([2,1,0])

    # # tissue mask based on mean image intensity
    # meanval = imSum.abs().mean()
    # tissue_mask = imSum.abs() > meanval
    # tissue_mask = tissue_mask.any(3)
    
    #%% calculate FA maps
    sz = imSum.shape[0:3]
    FA = torch.zeros(sz,device=imSum.device,dtype=imSum.dtype)
    # FA = FA.reshape(64,64,64,1).repeat_interleave(nModes,3)
    FA = FA.unsqueeze(-1).repeat_interleave(nModes,3) 
    for i in range(nModes):
        FA[:,:,:,i] = torch.arccos(imSum[:,:,:,i]/imSum[:,:,:,nModes]).abs()
        if nRelModes > 0:
            FA[:,:,:,i] *= torch.exp (1j * imSum[:,:,:,i+nModes+1].angle())
    # solve intereferometry (one-inv mode)
    if nModes == meta['nTx']:
        I = (torch.ones(nModes)-2*torch.eye(nModes)).to(FA.device).to(FA.dtype)
        W = torch.eye(nModes).to(FA.dtype).to(FA.device)
        M = torch.inverse(torch.t(I) @ W @ torch.t(I)) @ torch.t(I) @ W
        maps = torch.reshape(FA, (-1, nModes))
        res = torch.einsum('ij, kj -> ki', M, maps)
        FA_res = torch.reshape(res, FA.shape)
    else:
        FA_res = FA.clone()
    # make phases relative to first phase
    #for i in range(1,FA_res.shape[3]):
    #    FA_res[...,i] = FA_res[...,i].abs() .* exp(1j * (FA_res[...,i]*FA_res[...,0].conj()).angle())
    #FA_res[...,0] = FA_res[...,0].abs() .* exp(1j * (FA_res[...,0]*FA_res[...,0].conj()).angle())

    #%% calc B0 map
    deltaPhi = (imSum[:,:,:,nModes] * imSum[:,:,:,-1].conj()).angle()
    deltaB0_Hz = deltaPhi / TEshift / 2 / torch.pi * -1
    
    #%% Export relevant data to Matlab
    gamma_rad = 267522189.9851
    nT_per_V = FA_res / (gamma_rad * AmplitudeIntegralVs) * 1e9;
    if nT_per_V.shape[3] == meta['nTx']:
        # scale by each TXScaleFactor individually
        nT_per_V = nT_per_V.conj().resolve_conj() / meta['TxScaleFactors']
    else:
        #nT_per_V = nT_per_V.conj().resolve_conj()# / np.abs(meta['TxScaleFactors']).mean()
        nT_per_V *=  np.sqrt(meta['nTx']);

    mydic = {"alpha"      : FA_res.permute([2,1,0,3]).flip(1).abs().rad2deg().numpy(),
             "im"         : imSum.permute([2,1,0,3]).flip(1).numpy(),
            #  "brainMask"  : brain_mask.permute([2,1,0]).flip(1).numpy(),
            #  "tissueMask" : tissue_mask.permute([2,1,0]).flip(1).numpy(),
             "nT_per_V"   : nT_per_V.permute([2,1,0,3]).flip(1).numpy(),
             "deltaB0_Hz" : deltaB0_Hz.permute([2,1,0]).flip(1).numpy(),
             "meta"       : meta}

    savemat(outfile,mydic)
    return mydic
    #os.system('scp B1_maps.mat gadgetron:darioTest/python-pulsedesign/data/test.mat')

def newloadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def gaussian(N, mu=.5, sigma=.1):
    x = torch.Tensor([float(x)/(N-1) for x in range(N)])
    w = torch.exp(-(x-mu)**2 / (2 * sigma**2))
    #plt.figure()
    #plt.plot(np.abs(w))
    #plt.show()
    return w

def dbMeta(twix):
    hdr = twix[-1]['hdr']
    meob = hdr['Meas']
    nPar = hdr['Phoenix']['sKSpace']['lPartitions']
    nSlc = 1
    try:
        nSlc = meob['NSlc']
    except:
        pass
    matSize = np.array([meob['NImageLins'], meob['NImageCols'], nSlc*nPar])
    
    slice = hdr['MeasYaps']['sSliceArray']['asSlice'][0];
    fov = np.array([meob['PeFOV'], meob['RoFOV'], nSlc * slice['dThickness']]) / 1e3;
    
    voxSz = fov / matSize
    
    #nTx = int(hdr['Config']['NCoilSelects'])
    #print(hdr['MeasYaps'])
    try:
        nTx = len(hdr['MeasYaps']['sTXSPEC']['asNucleusInfo'][0]['CompProtectionValues']['MaxOnlineTxAmpl'])
        #nTx = int( hdr['Config']['TXANumberOfChannels'])
    except:
        try:
            nTx = int( hdr['Config']['lNumberOfTXCalibDateTime'])
        except:
            nTx = int( hdr['Config']['NCoilSelects'])
        
    sPos = hdr['Phoenix']['sSliceArray']['asSlice'][0]['sPosition']
    offset = np.zeros(3)
    try:
        offset[0] =     sPos['dSag']/1e3
    except:
        pass
    try:
        offset[1] = -1 * sPos['dCor']/1e3
    except:
        pass
    try:
        offset[2] =     sPos['dTra']/1e3
    except:
        pass
    
            
    #hdr.MeasYaps.sTXSPEC.asNucleusInfo{1,1}.flReferenceAmplitude;
    Uref = hdr['MeasYaps']['sTXSPEC']['asNucleusInfo'][0]['flReferenceAmplitude']
    
    # Isocenter in voxelspace:
    isocenter = np.round ( matSize/2 - offset/voxSz )
    
    # Tx Scale Factors
    TxScaleFactors = np.ones((1,1,1,nTx),dtype=np.complex64)
    #try:
    for i in range(nTx):
        try:
            re = hdr['MeasYaps']['sTXSPEC']['aTxScaleFactor'][i]['dRe']
        except:
            re = 0
        try:
            im = hdr['MeasYaps']['sTXSPEC']['aTxScaleFactor'][i]['dIm']
        except:
            im = 0
        if re==0 and im==0:
            re = 1
        TxScaleFactors[...,i] = re + 1j * im
    print(f'Tx scale factors: {TxScaleFactors}')
    #except:
    #    pass
        #TxScaleFactors = np.ones((1,1,1,nTx));
    
    
    meta = {"matSize":matSize,
            "FOV"    : fov,
            "voxSz"  : voxSz,
            "nTx"    : nTx,
            "offset":offset,
            "isocenter": isocenter,
            "TxScaleFactors": TxScaleFactors,
            "Uref"   : Uref
           }
    return meta