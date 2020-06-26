import sys
import numpy as np
import os
from copy import deepcopy
import h5py
import f90nml
import time
import ctypes
from math import pi, ceil
from copy import deepcopy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from multiprocessing import Array
from multiprocessing import cpu_count
from multiprocessing import Process
from multiprocessing import Pool

from .utils import eu2qu, qu2eu
from .utils.imgprocess import *

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'EMsoft_DIR.txt'), mode='r') as f:
    EMsoft_DIR = f.read()
sys.path.append(EMsoft_DIR)

from PyEMEBSDDI import PyEMEBSDDI, PyEMEBSDRefine


ipar = [
    240,        #1 nx  = (numsx-1)/2; --this should be computed internally; from numsx in NMLparameters/MCCLNameList in master.h5
    150,        #2 globalworkgrpsz; in NMLparameters/MCCLNameList in master.h5
    10,         #3 num_el;          in NMLparameters/MCCLNameList in master.h5
    2000000000, #4 totnum_el;       in EMData/MCOpenCLL           in master.h5
    1,          #5 multiplier;      in EMData/MCOpenCL            in master.h5
    2,          #6 devid;           in NMLparameters/MCCLNameList in master.h5
    2,          #7 platid;          in NMLparameters/MCCLNameList in master.h5
    
    1,          #8 CrystalSystem;      in CrystalData in master.h5
    1,          #9 Natomtypes;         in CrystalData in master.h5
    225,        #10 SpaceGroupNumber;  in CrystalData in master.h5
    1,          #11 SpaceGroupSetting; in CrystalData in master.h5
    
    11,         #12 numEbins; in EMData/MCOpenCL & EMData/EBSDmaster in master.h5
    101,        #13 numzbins; in EMData/MCOpenCL in master.h5
    
    1,     #14 mcmode  ( 1 = 'full', 2 = 'bse1' ); in MCOpenCl.nml -- you have to do the translation to integer
    12,         #15 numangle; in EMdata/EBSD in EBSDout.h5
    24,         #16 nxten = nx/10; --this should be computed internally;
    
    # the following are only used in the master routine
    500,        #17 npx;     in NMLparameters/EBSDMasterNameList in master.h5
    12,         #18 nthreads in NMLparameters/EBSDMasterNameList in master.h5
    
    # the following are only used for EBSD patterns
    480,        #19 numx; in NMLparameters/EBSDNameList in EBSDout.h5
    480,        #20 numy; in NMLparameters/EBSDNameList in EBSDout.h5
    0, #21 number of orientation in quaternion set
    0, #22 binning factor (0-3)
    0, #23 binned x-dimension
    0, #24 binned y-dimension
    0, #25 anglemode  (0 for quaternions, 1 for Euler angles)
    151,        #26 ipf_wd in NMLparameters in DIout.h5
    186,        #27 ipf_ht in NMLparameters in DIout.h5
    10,         #28 nregions in NMLparameters in DIout.h5
    0,        #29 maskpattern in NMLparameters in DIout.h5
    0, #30 useROI   (1 or 0) -- I don't think this is used
    0,          #31 ROI1; ROI in NMLparameters in DIout.h5
    0,          #32 ROI2; ROI in NMLparameters in DIout.h5
    0,          #33 ROI3; ROI in NMLparameters in DIout.h5
    0,          #34 ROI4; ROI in NMLparameters in DIout.h5
    4,   #35 inputtype in NMLparameters in DIout.h5 -- must be translated to int internally
    0,          #36 uniform  ['1' = yes (background only), '0' = no ]; in NMLparameters/EBSDMasterNameList in master.h5
                    # EMsoft/Source/EMsoftHDFLib/patternmod.f90
                    # if (trim(inputtype).eq."Binary") itype = 1
                    # if (trim(inputtype).eq."TSLup1") itype = 2
                    # if (trim(inputtype).eq."TSLup2") itype = 3
                    # if (trim(inputtype).eq."TSLHDF") itype = 4
    4096,       #37 numexptsingle; in NMLparameters in DIout.h5
    4096,       #38 numdictsingle; in NMLparameters in DIout.h5
    5,         #39 nnk;           in NMLparameters in DIout.h5
    0, #40 totnumexpt     (number of experimental patterns in current batch) --this  is computed internally
    0, #41 numexptsingle*ceiling(float(totnumexpt)/float(numexptsingle))  -- this should be computed internally
   
    3600, #42 16*ceiling(float(numsx*numsy)/16.0) --this is computed internally 
    0, #43 neulers  (number of Euler angle triplets in the dictionary)-- this is read from eulerangle file
    0, #44 nvariants (number of variants for refinement wrapper) -- this is read from PSvaraiantfile if used
    
    # EMEBSDDIpreview.nml
    1,      #45 nregionsmin
    1,      #46 nregionsstepsize
    0,      #47 numav
    1,      #48 patx
    1,      #49 paty
    
    0, #50 numw (number of hipass parameters)
    0, #51 numr (number of regions parameters)
   
    0, #52 unused from here
    0, #53
    0, #54
    0, #55
    0, #56
    0, #57
    0, #58
    0, #59
    0, #60
    0, #61
    0, #62
    0, #63
    0, #64
    0, #65
    0, #66
    0, #67
    0, #68
    0, #69
    0, #70
    0, #71
    0, #72
    0, #73
    0, #74
    0, #75
    0, #76
    0, #77
    0, #78
    0, #79
    0, #80
]
fpar = [
    75.7,       #1 sig;       in MCOpenCL.nml
    0.0,        #2 omega;     in NMLparams/MCCLNameList in master.h5
    20.0,       #3 EkeV;      in NMLparams/MCCLNameList in master.h5
    10.0,       #4 Ehistmin;  in NMLparams/MCCLNameList in master.h5
    1.0,        #5 Ebinsize;  in NMLparams/MCCLNameList in master.h5
    100.0,      #6 depthmax;  in NMLparams/MCCLNameList in master.h5
    1.0,        #7 depthstep; in NMLparams/MCCLNameList in master.h5
    
    0.0,        #8 sigstart; in MCOpenCL.nml
    30.0,       #9 sigend;   in MCOpenCL.nml
    2.0,        #10 sigstep; in MCOpenCL.nml
    
    # parameters only used in the master pattern routine
    0.05,       #11 dmin;      in NMLparameters/EBSDMasterNameList in master.h5
    4.0,        #12 Bethe  c1; in NMLparameters/BetheList in master.h5
    8.0,        #13 Bethe  c2; in NMLparameters/BetheList in master.h5
    50.0,       #14 Bethe  c3; in NMLparameters/BetheList in master.h5
    
    # parameters only used in the EBSD pattern routine
    0.0,       #15 xpc;         in NMLparameters/EBSDNameList in EBSDout.h5
    0.0,       #16 yps;         in NMLparameters/EBSDNameList in EBSDout.h5
    50.0,      #17 delta;       in NMLparameters/EBSDNameList in EBSDout.h5
    10.0,      #18 thetac;      in NMLparameters/EBSDNameList in EBSDout.h5
    15000.0,   #19 L;           in NMLparameters/EBSDNameList in EBSDout.h5
    150.0,     #20 beamcurrent; in NMLparameters/EBSDNameList in EBSDout.h5
    100.0,     #21 dwelltime;   in NMLparameters/EBSDNameList in EBSDout.h5
    1.0,       #22 gamma value; in NMLparameters/EBSDNameList in EBSDout.h5
    240,       #23 maskradius;  in NMLparameters/EBSDNameList in EBSDout.h5
    0.05,      #24 hipassw;     in NMLparameters/EBSDNameList in EBSDout.h5
    
    # refinement parameters
    0.03,      #25 step; in EMFitOrientation.nml
    
    # preview parameters
    0.5,      #26 hipasswmax; in EMEBSDDIpreview.nml
    
    0.0, #27 unused from here
    0.0, #28
    0.0, #29
    0.0, #30
    0.0, #31
    0.0, #32
    0.0, #33
    0.0, #34
    0.0, #35
    0.0, #36
    0.0, #37
    0.0, #38
    0.0, #39
    0.0, #40
    0.0, #41
    0.0, #42
    0.0, #43
    0.0, #44
    0.0, #45
    0.0, #46
    0.0, #47
    0.0, #48
    0.0, #49
    0.0, #50
    0.0, #51
    0.0, #52
    0.0, #53
    0.0, #54
    0.0, #55
    0.0, #56
    0.0, #57
    0.0, #58
    0.0, #59
    0.0, #60
    0.0, #61
    0.0, #62
    0.0, #63
    0.0, #64
    0.0, #65
    0.0, #66
    0.0, #67
    0.0, #68
    0.0, #69
    0.0, #70
    0.0, #71
    0.0, #72
    0.0, #73
    0.0, #74
    0.0, #75
    0.0, #76
    0.0, #77
    0.0, #78
    0.0, #79
    0.0, #80
]
spar = [
    # config file
    '/home/EMsoft/EMsoftPublic',      #1 EMsoftpathname
    '/home/Elena/DItutorial/xtals',   #2 EMXtalFolderpathname
    '/home/Elena/EMData',             #3 EMdatapathname
    '/home/Elena/.config/EMsoft/tmp', #4 EMtmppathname
    '/home/Elena/Zihao_NN',           #5 EMsoftLibraryLocation
    '',                                #6 EMSlackWebHookURL
    '',                                #7 EMSlackChannel
    'Elena',             #8 UserName;     in EMheader in DIout.h5
    'Pittsburgh',        #9 UserLocation; in EMheader in DIout.h5
    'epascal@gmail.com', #10 UserEmail;   in EMheader in DIout.h5
    'Email',             #11 EMNotify
    'Yes',               #12 Develop
    'No',                #13 Release
    '', #14 h5copypath
    '', #15 EMsoftplatform
    '', #16 EMsofttestpath
    '', #17 EMsoftTestingPath
    '4_1_0_49b5275',     #18 EMsoftversion; Version in EMheader in DIout.h5
    '', #19 Configpath
    '', #20 Templatepathname
    '', #21 Resourcepathname
    '', #22 Homepathname
    '/home/zihaod/EMsoftPublic/opencl', #23 OpenCLpathname
    '', #24 Templatecodefilename
    '', #25 WyckoffPositionsfilename
    '', #26 Randomseedfilename
    '', #27 EMsoftnativedelimiter
    
    '', #28 unused from here
    '', #29
    '', #30
    '', #31
    '', #32
    '', #33
    '', #34
    '', #35
    '', #36
    '', #37
    '', #38
    '', #39
    '', #40
    '', #41
    '', #42
    '', #43
    '', #44
    '', #45
    '', #46
    '', #47
    '', #48
    '', #49
    '', #50
    '', #51
    '', #52
    '', #53
    '', #54
    '', #55
    '', #56
    '', #57
    '', #58
    '', #59
    '', #60
    '', #61
    '', #62
    '', #63
    '', #64
    '', #65
    '', #66
    '', #67
    '', #68
    '', #69
    '', #70
    '', #71
    '', #72
    '', #73
    '', #74
    '', #75
    '', #76
    '', #77
    '', #78
    '', #79
    '', #80
]

def PyEMEBSDDI_singlethread(ipar, fpar, spar, dp_shape, ep_shape, ep_idx):
    '''
    Singlethread wrapper for PyEMEBSDDI_wrapper when multiple gpu are used.

    Should not be called directly from outside.

    Will read patterns from multiprocessing shared memory __dp, __ep.

    Input:
    ----------
        ipar: int params, list;
        fpar: float params, list;
        spar: string params, list;
        dp_shape: shape of numpy array dpatterns, tuple, (row, column);
        ep_shape: shape of numpy array epatterns, tuple, (row, column);
        ep_idx: index of epatterns for current thread, list, [start, end];

    Output:
    ----------
        orientations output by PyEMEBSDDI, 2darray, (n,4)
    '''

    # read dpatterns, epatterns from shared memory
    # need to reshape
    dp = np.frombuffer(__dp, dtype=float)
    ep = np.frombuffer(__ep, dtype=float)
    dp.shape = dp_shape
    ep.shape = ep_shape

    # slice epatterns
    ep = ep[ep_idx[0]:ep_idx[1]]

    return PyEMEBSDDI(ipar, fpar, spar, dp, ep)


def PyEMEBSDDI_wrapper(epatterns, dpatterns, orientations, nml_dir,
                        epatterns_processing=None, dpatterns_processing=None,
                        gpu=None, refine=False):
    '''
    Higher-level wrapper for PyEMEBSDDI. Must have PyEMEBSDDI library before calling.

    Input:
    ----------
        epatterns: experimental patterns, 3darray, 8bit, (n,numsx*numsy);
        dpatterns: dictionary patterns, 3darray, 8bit, (n,numsx*numsy);
        orientations: orientations of dictionary patterns, unit quaternions, 2darray, (n,len(dpatterns));
        nml_dir: nml file dir, string;
        epatterns_processing: img processing recipe for epatterns, list of strings;
        dpatterns_processing: img processing recipe for dpatterns, list of strings;
        gpu: multiple gpu choice, None or list of device id (int);
            if single gpu, gpu = None or len(gpu) = 1, actual gpu used is determined by devid (ipar[5]);
            if multiple gpu, gpu is the list of devid of all gpu used;
            Attention: full use of multiplt gpu needs more cpu resource and memory!
        refine: whether to refine indexing, bool, currently unavailable;

    Output:
    ----------
        pred: orientations output by PyEMEBSDDI, 2darray, (n,4);
        resultmain: dot products for each orientation prediction, 1darray, (n,);
    '''

    # *********************************************
    # 1. Check patterns, orientations, do processing
    # *********************************************
    assert dpatterns.shape[0] == orientations.shape[0], 'dpatterns and orientations have different length.'
    assert len(dpatterns.shape) == 3, 'dpatterns is not 3darray.'
    assert len(epatterns.shape) == 3, 'epatterns is not 3darray.'
    assert np.max(epatterns) >= 200., 'check whether epatterns are 8bit.'
    assert np.max(dpatterns) >= 200., 'check whether dpatterns are 8bit.'
    assert dpatterns.shape[1] == epatterns.shape[1], 'dpatterns and epatterns have different rows'
    assert dpatterns.shape[2] == epatterns.shape[2], 'dpatterns and epatterns have different columns'
    assert (gpu == None or isinstance(gpu, list)), 'param gpu must be None or list.'

    # start = time.time()
    if epatterns_processing:
        for i in epatterns_processing:
            epatterns = eval(i.replace('(','(epatterns,',1))
    if dpatterns_processing:
        for i in dpatterns_processing:
            dpatterns = eval(i.replace('(','(dpatterns,',1))
    # print('img processing time:', time.time()-start)

    # start = time.time()
    epatterns = epatterns.astype(float)
    # epatterns = np.clip(np.nan_to_num(epatterns),0.,255.)
    dpatterns = dpatterns.astype(float)
    # dpatterns = np.clip(np.nan_to_num(dpatterns),0.,255.)
    epatterns = epatterns / 255.
    dpatterns = dpatterns / 255.
    # print('clip time:', time.time()-start)

    # start = time.time()
    epatterns = epatterns.reshape((epatterns.shape[0],-1))
    dpatterns = dpatterns.reshape((dpatterns.shape[0],-1))
    # print('reshape time:', time.time()-start)

    numepatterns = deepcopy(epatterns.shape[0])
    numdpatterns = deepcopy(dpatterns.shape[0])

    print('epatterns shape:', epatterns.shape)
    print('dpatterns shape:', dpatterns.shape)
    print('orientations shape:', orientations.shape)


    # *********************************************
    # 2. Read nml, set params
    # *********************************************
    nml = f90nml.read(nml_dir)
    temp_dict = nml[list(nml.keys())[0]]
    
    ipar[5] = int(temp_dict['devid'])
    ipar[6] = int(temp_dict['platid'])
    ipar[17] = int(temp_dict['nthreads'])
    ipar[36] = int(temp_dict['numexptsingle'])
    ipar[37] = int(temp_dict['numdictsingle'])
    ipar[38] = int(temp_dict['nnk'])
    ipar[39] = int(temp_dict['totnumexpt'])
    ipar[42] = int(temp_dict['neulers'])

    spar[22] = str(temp_dict['OpenCLpathname'])

    # do padding based on numexptsingle/numdictsingle
    numexptsingle = ipar[36]
    numdictsingle = ipar[37]

    # ipar(42): 16*ceiling(float(numsx*numsy)/16.0)
    ipar[41] = 16 * ceil(dpatterns.shape[1]/16.)

    pad_d = numdictsingle - numdpatterns%numdictsingle
    pad_p = ipar[41] - dpatterns.shape[1]

    # only pad_e is affected by multi-gpu
    if not gpu or len(gpu) == 0:
        pad_e = numexptsingle - numepatterns%numexptsingle
    else:
        pad_e = numexptsingle*len(gpu) - numepatterns%(numexptsingle*len(gpu))

    # start = time.time()
    # epatterns = np.pad(epatterns, 
    #             ((0, pad_e), (0, pad_p)),
    #             mode='constant',
    #             constant_values=0.)
    epatterns_pad = np.zeros((numepatterns+pad_e, ipar[41]))
    epatterns_pad[:numepatterns, :epatterns.shape[1]] = epatterns
    epatterns = epatterns_pad

    # dpatterns = np.pad(dpatterns, 
    #             ((0, pad_d), (0, pad_p)),
    #             mode='constant',
    #             constant_values=0.)
    dpatterns_pad = np.zeros((numdpatterns+pad_d, ipar[41]))
    dpatterns_pad[:numdpatterns, :dpatterns.shape[1]] = dpatterns
    dpatterns = dpatterns_pad
    # print('padding time', time.time()-start)

    if not gpu or len(gpu) == 0:
        ipar[39] = epatterns.shape[0]
    else:
        ipar[39] = ceil(epatterns.shape[0] / len(gpu))
    ipar[42] = dpatterns.shape[0]
    # ipar(41): numexptsingle*ceiling(float(totnumexpt)/float(numexptsingle)) 
    ipar[40] = ipar[36] * ceil(ipar[39]/ipar[36])

    # for debug
    # print(ipar)
    # print(spar[22])
    # print(epatterns.shape)
    # print(dpatterns.shape)

    obj = 0
    cancel = False


    # *********************************************
    # 3. call PyEMEBSDDI
    # *********************************************
    # start = time.time()
    if not gpu or len(gpu) == 0:
        resultmain, indexmain = PyEMEBSDDI(ipar=ipar, fpar=fpar, spar=spar,
                                            dpatterns=dpatterns, epatterns=epatterns,
                                            obj=obj, cancel=cancel)
    else:
        # put dpatterns, epatterns in shared Array
        # can be inherited by all child process
        # must be global variable
        global __dp, __ep
        __dp = Array(ctypes.c_double, dpatterns.size, lock=False)
        __ep = Array(ctypes.c_double, epatterns.size, lock=False)
        _dp = np.frombuffer(__dp)
        _ep = np.frombuffer(__ep)
        _dp.shape = dpatterns.shape
        _ep.shape = epatterns.shape
        _dp[:] = dpatterns
        _ep[:] = epatterns
        
        epatterns_each_process = ceil(epatterns.shape[0] / len(gpu))
        p = Pool()
        res = [p.apply_async(PyEMEBSDDI_singlethread,
            (ipar[:5]+[gpu[i],]+ipar[6:], fpar, spar,
            dpatterns.shape, epatterns.shape, [i*epatterns_each_process,(i+1)*epatterns_each_process])) for i in range(len(gpu))]
        p.close()
        p.join()
        resultmain = np.concatenate([j.get()[0] for j in res], axis=0)
        indexmain = np.concatenate([j.get()[1] for j in res], axis=0)
    # print('PyEMEBSDDI time', time.time()-start)
    
    resultmain = resultmain[:numepatterns]
    # Fortran index from 1
    indexmain = indexmain[:numepatterns].astype(int)
    indexmain -= 1
    pred = orientations[indexmain[:,0]]

    # for debug
    # for i in range(len(indexmain)):
    #     print(indexmain[i])

    return [pred, resultmain[:,0]]


def PyEMEBSDRefine_wrapper(epatterns, startOrientations, startdps, variants, 
                        nml_dir, h5_dir, epatterns_processing=None):
    '''
    Higher-level wrapper for PyEMEBSDDI. Must have PyEMEBSDDI library before calling.

    Input:
    ----------
        epatterns: experimental patterns, 3darray, 8bit, (n,numsx*numsy);
        startOrientations:
        startdps:
        variants:
        nml_dir: nml file dir, string;
        h5_dir:
        epatterns_processing: img processing recipe for epatterns, list of strings;

    Output:
    ----------
        orientations output by PyEMEBSDRefine, 2darray, (n,4)
    '''

    # *********************************************
    # 1. Check patterns, orientations, do processing
    # *********************************************
    assert epatterns.shape[0] == startOrientations.shape[0], 'len(epatterns) != len(startOrientations)'
    assert epatterns.shape[0] == startdps.shape[0], 'len(epatterns) != len(startdps)'
    assert len(startdps.shape) == 1, 'startdps should be 1darray'
    assert startOrientations.shape[1] == 4, 'start_orientation is not in the form of quaternions'

    # start = time.time()
    if epatterns_processing:
        for i in epatterns_processing:
            epatterns = eval(i.replace('(','(epatterns,',1))
    # print('img processing time:', time.time()-start)

    # start = time.time()
    if epatterns.dtype != float:
        epatterns = epatterns.astype(float)
    # epatterns = np.clip(np.nan_to_num(epatterns),0.,255.)
    if np.max(epatterns) > 1.5:
        epatterns = epatterns / 255.
    # print('clip time:', time.time()-start)

    # start = time.time()
    epatterns = epatterns.reshape((epatterns.shape[0],-1))
    # print('reshape time:', time.time()-start)

    numepatterns = deepcopy(epatterns.shape[0])

    print('epatterns shape:', epatterns.shape)


    # *********************************************
    # 2. Read accum_e, mLPNH, mLPSH from h5
    # *********************************************
    h5 = h5py.File(h5_dir, 'r')
    accum_e = np.array(h5['EMData']['MCOpenCL']['accum_e'], dtype=np.int32)
    mLPNH = np.array(h5['EMData']['EBSDmaster']['mLPNH'], dtype=np.float32)
    mLPSH = np.array(h5['EMData']['EBSDmaster']['mLPSH'], dtype=np.float32)
    mLPNH = mLPNH.reshape(mLPNH.shape[1:])
    mLPSH = mLPSH.reshape(mLPSH.shape[1:])

    assert mLPNH.shape == mLPSH.shape, 'mLPNH.shape != mLPSH.shape'
    assert accum_e.shape[-1] == mLPNH.shape[0], 'numEbins inconsistent in accum_e and mLPNH'

    # *********************************************
    # 3. Read nml, set params
    # *********************************************
    nml = f90nml.read(nml_dir)
    temp_dict = nml[list(nml.keys())[0]]

    ipar[3] = int(temp_dict['totnum_el'])
    ipar[4] = int(temp_dict['multiplier'])
    ipar[9] = int(temp_dict['SpaceGroupNumber'])
    ipar[16] = int(temp_dict['npx'])
    ipar[17] = int(temp_dict['nthreads'])
    ipar[18] = int(temp_dict['numx'])
    ipar[19] = int(temp_dict['numy'])
    ipar[25] = int(temp_dict['ipf_wd'])
    ipar[26] = int(temp_dict['ipf_ht'])
    ipar[27] = int(temp_dict['nregions'])
    ipar[28] = int(temp_dict['maskpattern'])
    ipar[36] = int(temp_dict['numexptsingle'])
    ipar[37] = int(temp_dict['numdictsingle'])
    ipar[43] = int(temp_dict['nvariants'])

    fpar[0] = float(temp_dict['sig'])
    fpar[1] = float(temp_dict['omega'])
    fpar[2] = float(temp_dict['EkeV'])
    fpar[3] = float(temp_dict['Ehistmin'])
    fpar[4] = float(temp_dict['Ebinsize'])
    fpar[14] = float(temp_dict['xpc'])
    fpar[15] = float(temp_dict['ypc'])
    fpar[16] = float(temp_dict['delta'])
    fpar[17] = float(temp_dict['thetac'])
    fpar[18] = float(temp_dict['L'])
    fpar[19] = float(temp_dict['beamcurrent'])
    fpar[20] = float(temp_dict['dwelltime'])
    fpar[21] = float(temp_dict['gammavalue'])
    fpar[22] = float(temp_dict['maskradius'])
    fpar[24] = float(temp_dict['step'])

    ipar[0] = int((accum_e.shape[0]-1)/2.)
    ipar[11] = int(mLPNH.shape[0])
    ipar[16] = int((mLPNH.shape[-1]-1)/2.)
    ipar[41] = 16 * ceil(epatterns.shape[1]/16.)

    if variants is None or len(variants) == 0:
        variants = np.array([[1.,0.,0.,0.]], dtype=float)
    ipar[43] = variants.shape[0]

    # do padding based on numexptsingle
    numepatterns = deepcopy(epatterns.shape[0])
    numexptsingle = ipar[36]

    pad_e = numexptsingle - numepatterns%numexptsingle
    pad_p = ipar[41] - epatterns.shape[1]

    epatterns_pad = np.zeros((numepatterns+pad_e, ipar[41]))
    epatterns_pad[:numepatterns, :epatterns.shape[1]] = epatterns
    epatterns = epatterns_pad
    
    ipar[39] = epatterns.shape[0]
    ipar[40] = ipar[36] * ceil(ipar[39]/ipar[36])

    obj = 0
    cancel = False


    # *********************************************
    # 4. start_orientation -> startEulers, pad startdps
    # *********************************************
    startEulers = np.zeros((epatterns.shape[0],3), dtype=float)
    for i in range(numepatterns):
        startEulers[i] = qu2eu(startOrientations[i])

    # pad startdps
    startdps = np.concatenate((startdps, np.zeros((epatterns.shape[0]-startdps.shape[0],), dtype=float)), axis=0)


    # *********************************************
    # 5. call PyEMEBSDRefine
    # *********************************************
    # start = time.time()
    eumain, dpmain = PyEMEBSDRefine(ipar=ipar, fpar=fpar, 
                                    accum_e=accum_e, mLPNH=mLPNH, mLPSH=mLPSH,
                                    variants=variants, epatterns=epatterns,
                                    startEulers=startEulers, startdps=startdps,
                                    obj=obj, cancel=cancel)
    # print('PyEMEBSDRefine time', time.time()-start)
    print('eumain.shape:', eumain.shape)
    print('dpmain.shape:', dpmain.shape)

    eumain = eumain[:numepatterns]
    dpmain = dpmain[:numepatterns]

    pred = np.zeros((numepatterns,4), dtype=float)
    for i in range(numepatterns):
        pred[i] = eu2qu(eumain[i])

    return [pred, dpmain]