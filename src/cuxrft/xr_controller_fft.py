#Written by Tim Vogel
import xarray as xr
import numpy as np
import time
import math
import warnings
import dask.delayed
from dask.array import from_delayed, from_array
from random import randrange
import socket
from subprocess import run, PIPE
import pickle
import os
import concurrent.futures
import sys
from pathlib import Path
import cupy
from byte_converter import convertToBytesByUnit

def castable_int(value):
    try:
        int(value)
        return True
    except ValueError:
        False

def try_cast_int(value):
    try:
        return int(value)
    except ValueError:
        return value

def check_data_sufficient(data, FFT_dims, data_vars): 
    if ((not isinstance(data, xr.Dataset)) == (not isinstance(data, xr.DataArray))):
        raise TypeError('Data as ' + str(type(data)) + ' type provided.\nPlease provide as xarray.Dataset or xarray.DataArray.')
    if (isinstance(FFT_dims, list) and False in [False if dim  not in list(data.dims) else True for dim  in FFT_dims]):
        raise ValueError('FFT dimension(s) "' + FFT_dims + '" not found in ' + str(type(data)))
    #elif (isinstance(FFT_dims, dict) and False in [False if dim[1] not in list(data[dim[0]].dims) else True for dim in FFT_dim FFT_dim in FFT_dims[data_var] for data_var in list(FFT_dims.keys())]):
    elif (isinstance(FFT_dims, dict) and False in [False if FFT_dim not in list(data[data_var].dims) else True for data_var in list(FFT_dims.keys()) for FFT_dim in FFT_dims[data_var]]):
        raise ValueError('(Not all) FFT dimension(s) "' + FFT_dims + '" found in ' + str(type(data)))
    if (isinstance(FFT_dims, list) and isinstance(data, xr.Dataset) and False in [False if data_var not in list(data.keys()) else True for data_var in data_vars]):
        raise ValueError('Not all data_vars "' + str(data_vars) + '" found in ' + str(type(data)))
    elif (isinstance(FFT_dims, dict) and isinstance(data, xr.Dataset) and False in [False if data_var not in list(data.keys()) else True for data_var in list(FFT_dims.keys())]):
        raise ValueError('Not all data_vars "' + str(list(FFT_dims.keys())) + '" from FFT_dims dict found in ' + str(type(data)))

def get_maximalMemoryGPUs(availableGPUs):
    nvidiaGPUAnswer = run(['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv'], stdout=PIPE).stdout.decode().split("\n")
    if (isinstance(availableGPUs, list)):
        return [GPU.split(', ')[1] for GPU in nvidiaGPUAnswer if try_cast_int(GPU.split(', ')[0]) in availableGPUs]
    elif (isinstance(availableGPUs, str)):
        return [GPU.split(', ')[1] for GPU in nvidiaGPUAnswer if castable_int(GPU.split(', ')[0])]
    else:
        raise ValueError('GPUs must either be list, or str with the value "all".')

def get_smallestMemoryOfAllGPUs(availableGPUs):
    memoryGPUs = get_maximalMemoryGPUs(availableGPUs)
    smallest = convertToBytesByUnit(memoryGPUs[0])
    for mem in memoryGPUs[1:]:
        if (convertToBytesByUnit(mem) < smallest):
            smallest = convertToBytesByUnit(mem)
    return smallest

def get_freeGPU(availableGPUs, filterList=['mumax3', 'mumax3me'], filterListCheckGPU=[], treshHoldGPUUsage=0.1):
    programsToFilterGPUs = [] + filterList
    checkUsageFor = [] + filterListCheckGPU
    pythonVersion = sys.version.split(' ')[0].replace("'", '')
    programsToFilterGPUs.append('python')
    checkUsageFor.append('python')
    for i in range(1, len(pythonVersion.split('.'))+1):
        programsToFilterGPUs.append('python' + '.'.join(pythonVersion.split('.')[:i]))
        checkUsageFor.append('python' + '.'.join(pythonVersion.split('.')[:i]))
    nvidiaGPUAnswer = run(["nvidia-smi"], stdout=PIPE).stdout.decode().split("\n")
    if (os.name == "nt"):
        nvidiaGPUAnswer = run(["nvidia-smi"], stdout=PIPE).stdout.decode().split("\n") 
        nvidiaGPUAnswer = [list(filter(None, line.split(" "))) for line in nvidiaGPUAnswer]
        nvidiaGPUAnswerLower = [list(filter(lambda char: char != "|" and char != "N/A", char)) for char in nvidiaGPUAnswer[nvidiaGPUAnswer.index(['|=======================================================================================|\r'])+1:-2]]
        nvidiaGPUAnswerUpper = [list(filter(lambda char: char != "|" and char != "N/A", char)) for char in nvidiaGPUAnswer[nvidiaGPUAnswer.index(['|=========================================+======================+======================|\r'])+1:nvidiaGPUAnswer.index(['|=======================================================================================|\r'])-2]]
        nvidiaGPUIndex = []
        hardwareIndexAvailable = []
        for upperIndex in list(range(0, len(nvidiaGPUAnswerUpper)-5, 4)):
            if (castable_int(nvidiaGPUAnswerUpper[upperIndex][0])):
                hardwareIndexAvailable.append(int(nvidiaGPUAnswerUpper[upperIndex][0]))
        if (len(nvidiaGPUAnswerLower) > 0 and nvidiaGPUAnswerLower[0][0] != 'No'):
            for i in range(len(nvidiaGPUAnswerLower)):
                upperListConverter = list(range(1, len(nvidiaGPUAnswerUpper)-5, 4))
                if Path(nvidiaGPUAnswerLower[i][3]).stem.lower() in programsToFilterGPUs:
                    if (Path(nvidiaGPUAnswerLower[i][3]).stem.lower() in checkUsageFor and float(nvidiaGPUAnswerUpper[upperListConverter[int(nvidiaGPUAnswerLower[i][0])]][9].replace('%', ""))/100 > treshHoldGPUUsage or Path(nvidiaGPUAnswerLower[i][3]).stem.lower() not in checkUsageFor and Path(nvidiaGPUAnswerLower[i][3]).stem.lower() in programsToFilterGPUs):
                        try:
                            hardwareIndexAvailable.remove(int(nvidiaGPUAnswerLower[i][0]))
                        except ValueError:
                            pass
            nvidiaGPUIndex = np.unique(np.asarray(hardwareIndexAvailable)).tolist()
        elif (len(nvidiaGPUAnswerLower) == 0 or nvidiaGPUAnswerLower[0][0] == 'No'):
            nvidiaGPUIndex = hardwareIndexAvailable
        if (isinstance(availableGPUs, list)):
            nvidiaGPUIndex = [int(index) for index in nvidiaGPUIndex if int(index) in availableGPUs]
        return nvidiaGPUIndex
    
    elif (os.name == "posix"):
        nvidiaGPUAnswer = run(["nvidia-smi"], stdout=PIPE).stdout.decode().split("\n") 
        nvidiaGPUAnswer = [list(filter(None, line.split(" "))) for line in nvidiaGPUAnswer]
        systemDetails = str(os.uname())
        if ('WSL' not in systemDetails):
            nvidiaGPUAnswerLower = [list(filter(lambda char: char != "|" and char != "N/A", char)) for char in nvidiaGPUAnswer[nvidiaGPUAnswer.index(['|=============================================================================|'])+1:-2]]
            nvidiaGPUAnswerUpper = [list(filter(lambda char: char != "|" and char != "N/A", char)) for char in nvidiaGPUAnswer[nvidiaGPUAnswer.index(['|===============================+======================+======================|'])+1:nvidiaGPUAnswer.index(['|=============================================================================|'])-2]]
        else:
            nvidiaGPUAnswerLower = [list(filter(lambda char: char != "|" and char != "N/A", char)) for char in nvidiaGPUAnswer[nvidiaGPUAnswer.index(['|=======================================================================================|'])+1:-2]]
            nvidiaGPUAnswerUpper = [list(filter(lambda char: char != "|" and char != "N/A", char)) for char in nvidiaGPUAnswer[nvidiaGPUAnswer.index(['|=========================================+======================+======================|'])+1:nvidiaGPUAnswer.index(['|=======================================================================================|'])-2]]
        
        nvidiaGPUIndex = []
        hardwareIndexAvailable = []
        for upperIndex in list(range(0, len(nvidiaGPUAnswerUpper)-5, 4)):
            if (castable_int(nvidiaGPUAnswerUpper[upperIndex][0])):
                hardwareIndexAvailable.append(int(nvidiaGPUAnswerUpper[upperIndex][0]))
        if (len(nvidiaGPUAnswerLower) > 0 and nvidiaGPUAnswerLower[0][0] != 'No'):
            for i in range(len(nvidiaGPUAnswerLower)):
                upperListConverter = list(range(1, len(nvidiaGPUAnswerUpper)-5, 4))
                if Path(nvidiaGPUAnswerLower[i][3]).stem.lower() in programsToFilterGPUs:
                    if (Path(nvidiaGPUAnswerLower[i][3]).stem.lower() in checkUsageFor and float(nvidiaGPUAnswerUpper[upperListConverter[int(nvidiaGPUAnswerLower[i][0])]][9].replace('%', ""))/100 > treshHoldGPUUsage or Path(nvidiaGPUAnswerLower[i][3]).stem.lower() not in checkUsageFor and Path(nvidiaGPUAnswerLower[i][3]).stem.lower() in programsToFilterGPUs):
                        try:
                            hardwareIndexAvailable.remove(int(nvidiaGPUAnswerLower[i][0]))
                        except ValueError:
                            pass
            nvidiaGPUIndex = np.unique(np.asarray(hardwareIndexAvailable)).tolist()
        elif (len(nvidiaGPUAnswerLower) == 0 or nvidiaGPUAnswerLower[0][0] == 'No'):
            nvidiaGPUIndex = hardwareIndexAvailable
        if (isinstance(availableGPUs, list)):
            nvidiaGPUIndex = [int(index) for index in nvidiaGPUIndex if int(index) in availableGPUs]
        return nvidiaGPUIndex


def waitForGPU(availableGPUs):
    while (len(get_freeGPU(availableGPUs)) == 0):
        time.sleep(1)
    return get_freeGPU(availableGPUs)


def GPU_client(message):
    
    # Create a UDP socket
    sock = None
    server_address = ''
    client_address = ''
    #windows uses different socket type -> necessary for docker
    if (os.name == 'nt'):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        # Connect the socket to the port where the server is listening
        server_address = ('localhost', 8070)
        sock.connect(server_address)
    elif (os.name == 'posix'):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        server_address = './socket_server_file_FFT'
        if not os.path.isdir('clientAdress'):
            os.mkdir('clientAdress')
        client_address = './clientAdress/socket_client_file' + str(randrange(20000))
        try:
            os.unlink(client_address)
        except FileNotFoundError:
            pass

        sock.connect(server_address)
        sock.bind(client_address)

    else:
        raise TypeError('Unknown OS.')
    # Send data
    sent = sock.sendto(pickle.dumps(message), server_address)
    # Receive response
    data, server = sock.recvfrom(4096)
    sock.close()
    if (os.name == 'posix'):
        try:
            os.unlink(client_address)
        except FileNotFoundError:
            pass
    return pickle.loads(data)

def GPUcontrollingServer():
    # Make sure file doesn't exist already
    print('Starting up GPU controlling Server ...')
    sock = None
    server_address = ''
    if (os.name == 'nt'):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = ('localhost', 8070)
        sock.bind(server_address)
    elif (os.name == 'posix'):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        server_address = './socket_server_file_FFT'
        try:
            os.unlink(server_address)
        except FileNotFoundError:
            pass
        sock.bind(server_address)
    GPUsInUse = {}
    freeGPUsFirstRun = []
    while True:
        pickledData, address =  sock.recvfrom(4096)
        data = pickle.loads(pickledData)
        answer = {}
        if (isinstance(data, dict) and 'requestGPU' in data):
            freeGPUs = get_freeGPU(data['requestGPU'])
            if (freeGPUsFirstRun == []):
                freeGPUsFirstRun = freeGPUs
            else:
                freeGPUs = freeGPUsFirstRun
            for GPU in freeGPUs:
                if (GPU not in GPUsInUse or GPU in GPUsInUse and GPUsInUse[GPU] != True):
                    answer['useGPU'] = GPU
                    GPUsInUse[GPU] = True
                    break
            if ('useGPU' not in answer):
                answer['useGPU'] = -1
        elif (isinstance(data, dict) and 'freeGPU' in data):
            if (data['freeGPU'] in GPUsInUse and GPUsInUse[data['freeGPU']] == True):
                GPUsInUse[data['freeGPU']] = False
                answer['freedGPU'] = data['freeGPU']
            else:
                answer['freedGPU'] = -1
        elif (isinstance(data, dict) and 'checkHealth' in data):
            answer['alive'] = 0
        elif (isinstance(data, dict) and 'exit' in data):
            pass
        else:
            print(data)
        if data:
            sent = sock.sendto(pickle.dumps(answer), address)
        if (isinstance(data, dict) and 'exit' in data):
            if (data['exit'] != 0):
                raise ValueError
            else:
                break

def get_DimsLengthOne(dataset, data_var):
    dimsLengthOne = [dim for dim in dataset[data_var].dims if dataset.dims[dim] == 1]
    return dimsLengthOne

def get_CoordsLengthOne(dataset, data_var):
    dimsLengthOne = get_DimsLengthOne(dataset, data_var)
    coordsLengthOne = []
    for dim in dimsLengthOne:
        coordsLengthOne.append(dataset.coords[dim].values.tolist()[0])
    return coordsLengthOne

def sel_lengthOne(dataset, data_var):
    dimsToSelect = get_DimsLengthOne(dataset, data_var)
    for dim in dimsToSelect:
        dataset[data_var] = dataset[data_var].drop_vars(dim)
    dataset[data_var] = dataset[data_var].squeeze()
    return dataset

def sel_chunk(data, chunks, out, resultDim, fftAxis, fftDirection="forward", fftNorm=None, chunkedDims={}, useGPU=0, sc=None, multiple_GPUs=False):
    if (chunkedDims.keys() != chunks.keys()):
        for key in chunks.keys():
            if (key in chunkedDims):
                continue
            chunkLength = chunks[key]
            scList = []
            for keyChunked in chunkedDims.keys():
                scList.append((list(data[resultDim].dims).index(keyChunked), slice(chunkedDims[keyChunked]["chunkIndex"], chunkedDims[keyChunked]["chunkIndex"] + chunkedDims[keyChunked]["chunkLength"], 1)))
            scList.sort(key=lambda scTuple: scTuple[0]) 
            scExt = tuple([slice(0, None, 1)]*[d for d in list(data[resultDim].dims)].index(key))
            for i in range(0, data.dims[key], chunkLength): 
                chunkAdded = False
                sc = None
                if (chunkedDims != {} and not (key in chunkedDims and len(list(chunkedDims.keys())) == 1)):
                    for scIndex in range(len(scList)):
                        if (scIndex == 0 and scList[scIndex][0] != 0 and list(data[resultDim].dims).index(key) != 0):
                            if (scList[scIndex][0] <= list(data[resultDim].dims).index(key)):
                                if (i == 0):
                                    sc = tuple([slice(0, None, 1)]*(scList[scIndex][0])) + tuple([scList[scIndex][1]])
                                else:
                                    sc = tuple([slice(0, None, 1)]*(scList[scIndex][0])) + tuple([scList[scIndex][1]])
                            elif ((scList[scIndex][0] > list(data[resultDim].dims).index(key))):
                                    chunkAdded = True
                                    if (i == 0):
                                        sc = tuple([slice(0, None, 1)]*(list(data[resultDim].dims).index(key))) + tuple([slice(i, i+chunkLength, 1)]) + tuple([slice(0, None, 1)]*(list(data[resultDim].dims).index(key) - scList[scIndex][0])) + tuple([scList[scIndex][1]]) 
                                    else:
                                        sc = tuple([slice(0, None, 1)]*(list(data[resultDim].dims).index(key))) + tuple([slice(i, i+chunkLength, 1)]) + tuple([slice(0, None, 1)]*(list(data[resultDim].dims).index(key) - scList[scIndex][0])) + tuple([scList[scIndex][1]])
                        elif (scIndex == 0  and scList[scIndex][0] == 0):
                            sc = tuple([scList[scIndex][1]])
                        elif (len(scList) > scIndex + 1 and scList[scIndex][0] != scList[scIndex + 1][0]-1):
                            if (list(data[resultDim].keys()).index(key) < scList[scIndex + 1][0]):
                                sc = tuple([slice(0, None, 1)]*(scList[scIndex + 1][0] - list(data[resultDim].keys()).index(key))) + tuple([scList[scIndex + 1][1]])
                            else:
                                sc += tuple([slice(0, None, 1)]*(scList[scIndex + 1][0] - scList[scIndex][0])) + tuple([scList[scIndex + 1][1]]) 
                        elif (len(scList) > scIndex + 1 and scList[scIndex][0] == scList[scIndex + 1][0]-1):
                            sc += tuple([scList[scIndex + 1][1]])
                else:
                    sc = scExt
                if (chunkAdded == False):
                    if (sc != None):
                        sc += tuple([slice(i, i+chunkLength, 1)])
                    else:
                        sc = tuple([slice(i, i+chunkLength, 1)])
                chunkedDims[key] = {"chunkIndex": i, "chunkLength": chunkLength}
                out[sc] = sel_chunk(data.isel({key: slice(i, i+chunkLength)}), chunks, out, resultDim, fftAxis, fftDirection, fftNorm, chunkedDims, useGPU=useGPU, sc=sc, multiple_GPUs=multiple_GPUs)
            del chunkedDims[key]
            sc = list(sc)
            sc[list(data[resultDim].dims).index(key)] = slice(0, None, 1)
            sc = tuple(sc)
            return out[sc]
    else:
        @dask.delayed
        def calcFFT(data, resultDim, fftDirection, fftAxis, fftNorm):
            GPU_avail = {}
            if (multiple_GPUs == False):
                if (isinstance(useGPU, int)):
                    waitForGPU([useGPU])
                    GPU_avail['useGPU'] = useGPU
                elif (isinstance(useGPU, list) or isinstance(useGPU, str) and useGPU == 'all'):
                    GPU_avail['useGPU'] = waitForGPU(useGPU)[0]
            else:
                if (isinstance(useGPU, int)):
                    GPU_avail = GPU_client({'requestGPU': [useGPU]})
                    cupy.fft.config.use_multi_gpus = False
                elif (isinstance(useGPU, list)):
                    GPU_avail = GPU_client({'requestGPU': useGPU})
                    cupy.fft.config.use_multi_gpus = True
                elif (isinstance(useGPU, str) and useGPU == 'all'):
                    GPU_avail = GPU_client({'requestGPU': useGPU})
                    cupy.fft.config.use_multi_gpus = True
                else:
                    raise ValueError
                if ('useGPU' in GPU_avail):
                    while (GPU_avail['useGPU'] == -1):
                        time.sleep(1)
                        GPU_avail = GPU_client({'requestGPU': useGPU})
                print('Executing on GPU ' + str(GPU_avail['useGPU']))
            with cupy.cuda.Device(GPU_avail['useGPU']):
                fft = cupy.array(data[resultDim].values)
                # cp.fft.config.show_plan_cache_info()
                #print("Current gpu memory usage: %s / %s" % (mempool.used_bytes()*1e-9, mempool.total_bytes()*1e-9))
                if (fftDirection.lower() == "forward"):
                    fft_d = cupy.fft.fft(fft, axis=fftAxis, norm=fftNorm) 
                elif (fftDirection.lower() == "inverse"):
                    fft_d = cupy.fft.ifft(fft, axis=fftAxis, norm=fftNorm) 
                else:
                    raise ValueError("No proper direction provided.")
                del fft
                fftReturn = fft_d.get()
                del fft_d
                if (multiple_GPUs == True):
                    time.sleep(3)
                    GPU_client({'freeGPU': GPU_avail['useGPU']})
            return fftReturn
        delayedObject = from_delayed(calcFFT(data, resultDim, fftDirection, fftAxis, fftNorm), dtype=np.complex128, shape=out[sc].shape)
        return delayedObject

def fft_cellwise(data, chunks='auto', FFT_dims='', data_vars='', delayed=False, multiple_GPUs=False, GPUs=[0], keepGPUcontrollingServerRunning=False):
    """
    Performs FFT along FFT_dim for data_var
        Parameters
        ----------
        data                                : xarray.Dataset or xarray.DataArray, xarray data containing the provided FFT_dims and data_vars name(s).
                                            If data is an xarray.DataArray data_var will be ignored.

        chunks                              : str or dict, will split data into chunks. If chunks='auto' the chunk size will be determined automatically comparing the size of the numpy array
                                            with the smallest available memory on the graphic card(s).

        FFT_dims                            : str or list or dict, str and list: the dimension(s) to calculate the FFT(s) along in data_var(s) (if it is and xarray.Dataset).
                                                                           dict: the dimensions(s) to calculate the FFT(s) along in the keys of FFT_dims and using the values
                                                                                  as the dims to transform. Str and list as values allowed.
                                            The returned xarray.Dataset or xarray.DataArray will contain these dimension(s) with the name(s) being prolonged by "_freq".
                                            If multiple data_vars are present in the data and the others are dependent on FFT_dims and will not be transformed a new dim will be created.

        data_vars                           : str or list, the name(s) of the data_var(s) to calculate the FFT from. Will be ignored if data is a xarray.DataArray. If no string is,
                                            the first entry will be used. Will be ignored if FFT_dims is a dict.

        
        delayed                             : bool, wherether the returned dataset should be made up by dask.delayed arrays. The GPU(s) are going to be reserved from python until the computation has been executed.

        multiple_GPUs                       : bool, if multiple GPUs should be used. Even uf GPUs is a list, this flag needs to be set true to use all of the GPUs provided in the list.
                                            This flag starts a controlling server that manages the access of the GPUs.

        GPUs                                : list or int, contains the index of the GPU(s) to use.

        keepGPUcontrollingServerRunning     : bool, wherether the GPU controlling server should continue to run, or not. Sometimes cuda does not closes itself properly.
                                            By setting keepGPUcontrollingServerRunning=True this can be resolved for the case that multiple FFTs are supposed to be computed during different calls
                                            of this method.

        Returns
        -------
        Dataset, or DataArray
    """
    #convert data into datarray and create list of data_vars and FFT_dims if not given or dataarray
    wasDataArray = False
    if (isinstance(FFT_dims, dict) and data_vars != '' and isinstance(data, xr.Dataset)):
        warnings.warn('FFT_dims and data_vars both set. Using FFT_dims.')
    elif (isinstance(FFT_dims, dict) and data_vars != '' and isinstance(data, xr.DataArray)):
        raise ValueError('Cannot use dict for FFT_dims for xarray.DataArray')
    if (isinstance(FFT_dims, dict)):
        data_vars = list(FFT_dims.keys())
        for key in FFT_dims.keys():
            if not isinstance(FFT_dims[key], list) and isinstance(FFT_dims[key], str):
                FFT_dims[key] = [FFT_dims[key]]
            if isinstance(FFT_dims[key], list):
                pass
            else:
                raise ValueError('Dict of FFT dims must contain list or string as value.')
    
    if (data_vars == ''  and isinstance(data, xr.Dataset) and not isinstance(FFT_dims, dict)):
        try:
            print('No name for data_vars provided. Transforming all data_vars.') 
            data_vars = list(data.keys())
        except IndexError:
            print(str(type(data)) + ' contains no data')
    elif (isinstance(data_vars, str) and isinstance(data, xr.Dataset)):
        data_vars = [data_vars] 
    elif (isinstance(data, xr.DataArray)): 
        data_vars = ['raw']
        wasDataArray = True 
        data = data.to_dataset().rename({0: 'raw'}) 
    if (FFT_dims == ''):
        dataTmp = data
        for data_var in data_vars:
            dataTmp = sel_lengthOne(dataTmp, data_var)
        FFT_dims = {}
        for data_var in dataTmp.keys():
            FFT_dims[data_var] = list(dataTmp[data_var].dims)
    elif (isinstance(FFT_dims, str)):
        FFT_dims = [FFT_dims]
    
    check_data_sufficient(data, FFT_dims, data_vars)
    chunksOrg = chunks
    cupy.fft.config.enable_nd_planning = False
    cupy.fft.config.use_multi_gpus = False
    FFT_dimsFlatten = FFT_dims
    if (isinstance(FFT_dimsFlatten, dict)):
        FFT_dimsFlatten = np.unique(sum(list(FFT_dimsFlatten.values()), [])).tolist()
    for FFT_dim in FFT_dimsFlatten:
        d_dim = float(np.mean(np.diff(data.coords[FFT_dim].to_numpy())))
        fft_freq = cupy.fft.rfftfreq(data.dims[FFT_dim]-1, d_dim).get()
        negativeFreq = -np.flip(fft_freq)
        if ((data.dims[FFT_dim] % 2) != 0):
            negativeFreq = negativeFreq[:-1]
        fft_freq = np.concatenate((negativeFreq, fft_freq))
        if (isinstance(data, xr.Dataset)):
            keepTAxis = False
            if (len(list(data.keys())) > 1):
                for data_var in list(data.keys()):
                    if (not isinstance(FFT_dims, dict) and data_var not in data_vars and FFT_dim in data[data_var].dims or isinstance(FFT_dims, dict) and FFT_dim in data[data_var].dims and (data_var not in FFT_dims or data_var in FFT_dims and FFT_dim not in FFT_dims[data_var])):
                        keepTAxis = True
                    elif (not isinstance(FFT_dims, dict) and data_var in data_vars and FFT_dim in data[data_var].dims and FFT_dim in FFT_dims or isinstance(FFT_dims, dict) and (data_var in FFT_dims and FFT_dim in data[data_var].dims and FFT_dim in FFT_dims[data_var])):
                        data[data_var] = data[data_var].rename({FFT_dim: FFT_dim + '_freq'})
                if (keepTAxis == True):
                    data[FFT_dim + '_freq'] = fft_freq.tolist()
                else:
                    data = data.assign_coords({FFT_dim + '_freq': fft_freq.tolist()})
                    if FFT_dim in list(data.dims):
                        data = data.drop_dims(FFT_dim)
            else:
                data = data.rename({FFT_dim: FFT_dim + '_freq'}).assign_coords({FFT_dim + '_freq': fft_freq.tolist()})
        else:
            data = data.rename({FFT_dim: FFT_dim + '_freq'}).assign_coords({FFT_dim + '_freq': fft_freq.tolist()})    
    #reduce amount of dimensions to minimum
    orgOrderDimsDataVars = []
    posOfDimsLengthOne = []
    dimsLengthOne = []
    coordsLengthOne = []
    for data_var in data_vars:
        orgOrderDimsDataVars.append(list(data[data_var].dims))
        dimsLengthOne.append(get_DimsLengthOne(data, data_var))
        coordsLengthOne.append(get_CoordsLengthOne(data, data_var))
        posOfDimsLengthOne.append([list(data[data_var].dims).index(dim) for dim in dimsLengthOne[-1]])
    for data_var in data_vars:
        data = sel_lengthOne(data, data_var)
    # Get steps of transform axis
    # Initialize mempool
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    if (multiple_GPUs == True):
        if (len(GPUs) == 1):
            GPUs = 'all'
        try:
            GPU_client({'checkHealth': 0})
            GPUServerStarted = True
        except (ConnectionResetError, ConnectionRefusedError, FileNotFoundError):
            GPUServerStarted = False
        if (GPUServerStarted == False):
            executor = concurrent.futures.ProcessPoolExecutor()
            GPUcontrollingServerThread = executor.submit(GPUcontrollingServer)
            while not GPUServerStarted:
                try:
                    GPU_client({'checkHealth': 0})
                    GPUServerStarted = True
                except (ConnectionResetError, ConnectionRefusedError, FileNotFoundError):
                    GPUServerStarted = False
                time.sleep(1)

    dataTransposed = False
    #transform axis has to be the 0th axis
    for data_var in data_vars:
        FFT_dimsTmp = FFT_dims
        if (isinstance(FFT_dimsTmp, dict)):
            if (data_var in FFT_dimsTmp):
                FFT_dimsTmp = FFT_dimsTmp[data_var]
            else:
                continue
        for FFT_dim in FFT_dimsTmp:
            rechunkDict = {}
            if (list(data[data_var].dims)[0] != FFT_dim):
                newDimOrder = list(data[data_var].dims)
                newDimOrder[newDimOrder.index(FFT_dim + '_freq')] = newDimOrder[0]
                newDimOrder[0] = FFT_dim + '_freq'
                data[data_var] = data[data_var].transpose(*newDimOrder)
                dataTransposed = True
            fftAxis = [dim for dim in data[data_var].dims].index(FFT_dim + '_freq')
            out = dask.array.empty(shape=tuple(data.dims[d] for d in data[data_var].dims), dtype=np.complex128)
            chunksDict = {}
            if (chunksOrg == 'auto'):
                memoryConsumption = out.itemsize*out.size
                minimalMemoryAvailable = get_smallestMemoryOfAllGPUs(GPUs)
                shape = list(out.shape)
                for dim in data[data_var].dims:
                    if (dim == FFT_dim + '_freq'):
                        continue
                    while (memoryConsumption > minimalMemoryAvailable / 3):
                        if (shape[list(data[data_var].dims).index(dim)]/ 2 > 1):
                            shape[list(data[data_var].dims).index(dim)] = int(shape[list(data[data_var].dims).index(dim)] / 2)
                            memoryConsumption = out.itemsize*math.prod(shape)
                            if (memoryConsumption <= minimalMemoryAvailable/3): 
                                chunksDict[dim] = int(shape[list(data[data_var].dims).index(dim)])
                        else:
                            chunksDict[dim] = int(shape[list(data[data_var].dims).index(dim)]) 
                            break
                if (chunksDict != {}):
                    print("Chunking with: " + str(chunksDict))
                chunks = chunksDict
            else:
                if (not isinstance(chunksOrg, dict)):
                    raise ValueError('chunks must either be dict or str.')
                from copy import deepcopy
                chunks = deepcopy(chunksOrg)
                chunksPrint = deepcopy(chunksOrg)
                for key in chunksOrg.keys():
                    if key in FFT_dimsFlatten:
                        if (key + '_freq' not in data[data_var].dims):
                            del chunks[key]
                            del chunksPrint[key]
                        else:
                            chunks[key + '_freq'] = chunks.pop(key)
                    else:
                        if (key not in data[data_var].dims):
                            del chunks[key]
                            del chunksPrint[key]
                    if (key == FFT_dim):
                        del chunks[key + '_freq']
                        del chunksPrint[key]
                    
                if (chunks != {}):
                    print("Chunking with: " + str(chunksPrint))
                
            if (chunks != {}):
                for key in chunks.keys():
                    rechunkDict[list(data[data_var].dims).index(key)] = chunks[key]
            cache = cupy.fft.config.get_plan_cache() 
            cache.set_size(0)
            out = dask.array.rechunk(sel_chunk(data, chunks, out, data_var, fftAxis, "forward", useGPU=GPUs, multiple_GPUs=multiple_GPUs), rechunkDict)
            if (delayed == False):
                print('Computing FFT in ' + data_var + ' along ' + FFT_dim)
                out = out.compute()
            data = data.update({data_var: (data[data_var].dims, out)})
    for data_var in data_vars:
        dataVarIndex = data_vars.index(data_var)
        for dim in dimsLengthOne[dataVarIndex]:
            data[data_var] = data[data_var].expand_dims(dim, posOfDimsLengthOne[dataVarIndex][dimsLengthOne[dataVarIndex].index(dim)]).assign_coords({dim: [coordsLengthOne[dataVarIndex][dimsLengthOne[dataVarIndex].index(dim)]]})
    if (dataTransposed == True):
        for data_var in data_vars:
            data[data_var] = data[data_var].transpose(*(orgOrderDimsDataVars[data_vars.index(data_var)]))
    if (wasDataArray == True):
        data = data.to_dataarray('raw')
    if (multiple_GPUs == True and keepGPUcontrollingServerRunning == False and delayed == False):
        GPU_client({'exit': 0}) 
    return data

def ifft_cellwise(data, chunks='auto', FFT_dims='', data_vars='', delayed=False, multiple_GPUs=False, GPUs=[0], keepGPUcontrollingServerRunning=False):
    """
    Performs iFFT along FFT_dim for data_var
        Parameters
        ----------
        data                                : xarray.Dataset or xarray.DataArray, xarray data containing the provided FFT_dims and data_vars name(s).
                                            If data is an xarray.DataArray data_var will be ignored.

        chunks                              : str or dict, will split data into chunks. If chunks='auto' the chunk size will be determined automatically comparing the size of the numpy array
                                            with the smallest available memory on the graphic card(s).
                                            If chunks is a dict, the dict shall contain the name of the dim(s) to chunk along as the key and the size as the entry

        FFT_dims                            : str or list or dict,  str and list: the dimension(s) to calculate the iFFT(s) along in data_var(s) (if it is and xarray.Dataset).
                                                                            dict: the dimensions(s) to calculate the iFFT(s) along in the keys of FFT_dims and using the values
                                                                                  as the dims to transform. Str and list as values allowed.
                                            The returned xarray.Dataset or xarray.DataArray will contain these dimension(s) with the name(s) being prolonged by "_freq".
                                            If multiple data_vars are present in the data and the others are dependent on FFT_dims and will not be transformed a new dim will be created.

        data_vars                           : str or list, the name(s) of the data_var(s) to calculate the iFFT from. Will be ignored if data is a xarray.DataArray. If no string is,
                                            the first entry will be used. Will be ignored if FFT_dims is a dict.

        
        delayed                             : bool, wherether the returned dataset should be made up by dask.delayed arrays. The GPU(s) are going to be reserved from 
                                            python until the computation has been executed.

        multiple_GPUs                       : bool, if multiple GPUs should be used. Even uf GPUs is a list, this flag needs to be set true to use all of the GPUs provided in the list.
                                            This flag starts a controlling server that manages the access of the GPUs.

        GPUs                                : list or int, contains the index of the GPU(s) to use.

        keepGPUcontrollingServerRunning     : bool, wherether the GPU controlling server should continue to run, or not. Sometimes cuda does not closes itself properly.
                                            By setting keepGPUcontrollingServerRunning=True this can be resolved for the case that multiple FFTs are supposed to be computed during different calls
                                            of this method.

        Returns
        -------
        Dataset, or DataArray
    """
    #convert data into datarray and create list of data_vars and FFT_dims if not given or dataarray
    wasDataArray = False
    if (isinstance(FFT_dims, dict) and data_vars != '' and isinstance(data, xr.Dataset)):
        warnings.warn('FFT_dims and data_vars both set. Using FFT_dims.')
    elif (isinstance(FFT_dims, dict) and data_vars != '' and isinstance(data, xr.DataArray)):
        raise ValueError('Cannot use dict for FFT_dims for xarray.DataArray')
    if (isinstance(FFT_dims, dict)):
        data_vars = list(FFT_dims.keys())
        for key in FFT_dims.keys():
            if not isinstance(FFT_dims[key], list) and isinstance(FFT_dims[key], str):
                FFT_dims[key] = [FFT_dims[key]]
            if isinstance(FFT_dims[key], list):
                pass
            else:
                raise ValueError('Dict of FFT dims must contain list or string as value.')
    
    if (data_vars == ''  and isinstance(data, xr.Dataset) and not isinstance(FFT_dims, dict)):
        try:
            print('No name for data_vars provided. Transforming all data_vars.') 
            data_vars = list(data.keys())
        except IndexError:
            print(str(type(data)) + ' contains no data')
    elif (isinstance(data_vars, str) and isinstance(data, xr.Dataset)):
        data_vars = [data_vars] 
    elif (isinstance(data, xr.DataArray)): 
        data_vars = ['raw']
        wasDataArray = True 
        data = data.to_dataset().rename({0: 'raw'}) 
    if (FFT_dims == ''):
        dataTmp = data
        for data_var in data_vars:
            dataTmp = sel_lengthOne(dataTmp, data_var)
        FFT_dims = {}
        for data_var in dataTmp.keys():
            FFT_dims[data_var] = list(dataTmp[data_var].dims)
    elif (isinstance(FFT_dims, str)):
        FFT_dims = [FFT_dims]
    
    check_data_sufficient(data, FFT_dims, data_vars)
    chunksOrg = chunks
    cupy.fft.config.enable_nd_planning = False
    cupy.fft.config.use_multi_gpus = False
    FFT_dimsFlatten = FFT_dims
    if (isinstance(FFT_dimsFlatten, dict)):
        FFT_dimsFlatten = np.unique(sum(list(FFT_dimsFlatten.values()), [])).tolist()
    for FFT_dim in FFT_dimsFlatten:
        d_dim = float(np.mean(np.diff(data.coords[FFT_dim].to_numpy())))
        fft_freq = cupy.fft.rfftfreq(2*(data.dims[FFT_dim]-1), d_dim).get()
        if ((data.dims[FFT_dim] % 2) != 0):
            fft_freq[:-1]
        if (isinstance(data, xr.Dataset)):
            keepTAxis = False
            if (len(list(data.keys())) > 1):
                for data_var in list(data.keys()):
                    if (not isinstance(FFT_dims, dict) and data_var not in data_vars and FFT_dim in data[data_var].dims or isinstance(FFT_dims, dict) and FFT_dim in data[data_var].dims and (data_var not in FFT_dims or data_var in FFT_dims and FFT_dim not in FFT_dims[data_var])):
                        keepTAxis = True
                    elif (not isinstance(FFT_dims, dict) and data_var in data_vars and FFT_dim in data[data_var].dims and FFT_dim in FFT_dims or isinstance(FFT_dims, dict) and (data_var in FFT_dims and FFT_dim in data[data_var].dims and FFT_dim in FFT_dims[data_var])):
                        data[data_var] = data[data_var].rename({FFT_dim: FFT_dim + '_freq'})
                if (keepTAxis == True):
                    data[FFT_dim + '_freq'] = fft_freq.tolist()
                else:
                    data = data.assign_coords({FFT_dim + '_freq': fft_freq.tolist()})
                    if FFT_dim in list(data.dims):
                        data = data.drop_dims(FFT_dim)
            else:
                data = data.rename({FFT_dim: FFT_dim + '_freq'}).assign_coords({FFT_dim + '_freq': fft_freq.tolist()})
        else:
            data = data.rename({FFT_dim: FFT_dim + '_freq'}).assign_coords({FFT_dim + '_freq': fft_freq.tolist()})    
    #reduce amount of dimensions to minimum
    orgOrderDimsDataVars = []
    posOfDimsLengthOne = []
    dimsLengthOne = []
    coordsLengthOne = []
    for data_var in data_vars:
        orgOrderDimsDataVars.append(list(data[data_var].dims))
        dimsLengthOne.append(get_DimsLengthOne(data, data_var))
        coordsLengthOne.append(get_CoordsLengthOne(data, data_var))
        posOfDimsLengthOne.append([list(data[data_var].dims).index(dim) for dim in dimsLengthOne[-1]])
    for data_var in data_vars:
        data = sel_lengthOne(data, data_var)
    # Get steps of transform axis
    # Initialize mempool
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    if (multiple_GPUs == True):
        if (len(GPUs) == 1):
            GPUs = 'all'
        try:
            GPU_client({'checkHealth': 0})
            GPUServerStarted = True
        except (ConnectionResetError, ConnectionRefusedError, FileNotFoundError):
            GPUServerStarted = False
        if (GPUServerStarted == False):
            executor = concurrent.futures.ProcessPoolExecutor()
            GPUcontrollingServerThread = executor.submit(GPUcontrollingServer)
            while not GPUServerStarted:
                try:
                    GPU_client({'checkHealth': 0})
                    GPUServerStarted = True
                except (ConnectionResetError, ConnectionRefusedError, FileNotFoundError):
                    GPUServerStarted = False
                time.sleep(1)
    dataTransposed = False
    #transform axis has to be the 0th axis
    for data_var in data_vars:
        FFT_dimsTmp = FFT_dims
        if (isinstance(FFT_dimsTmp, dict)):
            if (data_var in FFT_dimsTmp):
                FFT_dimsTmp = FFT_dimsTmp[data_var]
            else:
                continue
        for FFT_dim in FFT_dimsTmp:
            rechunkDict = {}
            if (list(data[data_var].dims)[0] != FFT_dim):
                newDimOrder = list(data[data_var].dims)
                newDimOrder[newDimOrder.index(FFT_dim + '_freq')] = newDimOrder[0]
                newDimOrder[0] = FFT_dim + '_freq'
                data[data_var] = data[data_var].transpose(*newDimOrder)
                dataTransposed = True
            fftAxis = [dim for dim in data[data_var].dims].index(FFT_dim + '_freq')
            out = dask.array.empty(shape=tuple(data.dims[d] for d in data[data_var].dims), dtype=np.complex128)
            chunksDict = {}
            if (chunksOrg == 'auto'):
                memoryConsumption = out.itemsize*out.size
                minimalMemoryAvailable = get_smallestMemoryOfAllGPUs(GPUs)
                shape = list(out.shape)
                for dim in data[data_var].dims:
                    if (dim == FFT_dim + '_freq'):
                        continue
                    while (memoryConsumption > minimalMemoryAvailable / 3):
                        if (shape[list(data[data_var].dims).index(dim)] / 2 > 1):
                            shape[list(data[data_var].dims).index(dim)] = int(shape[list(data[data_var].dims).index(dim)] / 2)
                            memoryConsumption = out.itemsize*math.prod(shape)
                            if (memoryConsumption <= minimalMemoryAvailable / 3): 
                                chunksDict[dim] = int(shape[list(data[data_var].dims).index(dim)])
                        else:
                            chunksDict[dim] = int(shape[list(data[data_var].dims).index(dim)]) 
                            break
                if (chunksDict != {}):
                    print("Chunking with: " + str(chunksDict))
                chunks = chunksDict
            else:
                if (not isinstance(chunksOrg, dict)):
                    raise ValueError('chunks must either be dict or str.')
                from copy import deepcopy
                chunks = deepcopy(chunksOrg)
                chunksPrint = deepcopy(chunksOrg)
                for key in chunksOrg.keys():
                    if key in FFT_dimsFlatten:
                        if (key + '_freq' not in data[data_var].dims):
                            del chunks[key]
                            del chunksPrint[key]
                        else:
                            chunks[key + '_freq'] = chunks.pop(key)
                    else:
                        if (key not in data[data_var].dims):
                            del chunks[key]
                            del chunksPrint[key]
                    if (key == FFT_dim):
                        del chunks[key + '_freq']
                        del chunksPrint[key]
                    
                if (chunks != {}):
                    print("Chunking with: " + str(chunksPrint))
            
            if (chunks != {}):
                for key in chunks.keys():
                    rechunkDict[list(data[data_var].dims).index(key)] = chunks[key]

            cache = cupy.fft.config.get_plan_cache() 
            cache.set_size(0)
            out = dask.array.rechunk(sel_chunk(data, chunks, out, data_var, fftAxis, "inverse", useGPU=GPUs, multiple_GPUs=multiple_GPUs), rechunkDict)
            if (delayed == False):
                print('Computing iFFT in ' + data_var + ' along ' + FFT_dim)
                out = out.compute()
            data = data.update({data_var: (data[data_var].dims, out)})
    for data_var in data_vars:
        dataVarIndex = data_vars.index(data_var)
        for dim in dimsLengthOne[dataVarIndex]:
            data[data_var] = data[data_var].expand_dims(dim, posOfDimsLengthOne[dataVarIndex][dimsLengthOne[dataVarIndex].index(dim)]).assign_coords({dim: [coordsLengthOne[dataVarIndex][dimsLengthOne[dataVarIndex].index(dim)]]})
    if (dataTransposed == True):
        for data_var in data_vars:
            data[data_var] = data[data_var].transpose(*(orgOrderDimsDataVars[data_vars.index(data_var)]))
    for dim in data.dims:
        if ('_freq' in dim):
            if (dim.replace('_freq_freq', '').replace('_freq', '') in data.dims):
                for data_var in data_vars:
                    if (dim in data[data_var].dims):
                        data[data_var] = data[data_var].swap_dims(({dim: dim.replace('_freq_freq', '').replace('_freq', '')})).drop_vars(dim)
                data = data.drop_dims(dim)
            else:
                data = data.rename({dim: dim.replace('_freq_freq', '').replace('_freq', '_ifft')})
    if (wasDataArray == True):
        data = data.to_dataarray('raw')
    if (multiple_GPUs == True and keepGPUcontrollingServerRunning == False and delayed == False):
        GPU_client({'exit': 0}) 
    return data

def closeGPUController():
    GPU_client({'exit': 0}) 

if __name__ == '__main__':
    """for parent in parent_nodes:
        xr_file = parent.get_file("data.nc")
        xr_file.retrieve()
    dataset = xr.open_dataset(xr_file.get_path())
    print(dataset.load())
    datasetfft = fft_cellwise(dataset, FFT_dims='t', data_vars=['raw'])
    #print(datasetfft)
    datasetifft = ifft_cellwise(datasetfft, FFT_dims='t_freq', data_vars=['raw'])
    print(datasetifft)"""
    print(xr.open_dataset(r"D:\tvo\final_nc_files\11-12-2023_18-20-45_script_whispering_gallery_mode_vortex_displacement_higher_space_res.nc").isel({'x': slice(256, 768), 'y': slice(256, 768), 't': slice(0, 1000)}))
    print(ifft_cellwise(fft_cellwise(xr.open_dataset(r"D:\tvo\final_nc_files\11-12-2023_18-20-45_script_whispering_gallery_mode_vortex_displacement_higher_space_res.nc").isel({'x': slice(256, 768), 'y': slice(256, 768), 't': slice(0, 1000)}), chunks='auto', FFT_dims='t', multiple_GPUs=True, keepGPUcontrollingServerRunning=True), chunks='auto', FFT_dims='t_freq', multiple_GPUs=True))
else:
    try:
        xr.Dataset.fft_cellwise = fft_cellwise
        xr.DataArray.fft_cellwise = fft_cellwise
        xr.Dataset.ifft_cellwise = ifft_cellwise
        xr.DataArray.ifft_cellwise = ifft_cellwise
    except AttributeError:
        print('Could not provide cuxrft methods as internal functions of xarray.')