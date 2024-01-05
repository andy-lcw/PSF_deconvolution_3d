#%%

__all__ = ['loadBinary', 'loadBin', 'loadBinLong', 'cut']


#%%

import typing as tp
from itertools import product, repeat

import numpy as np

import torch


#%%

def loadBinary( path: str, dtype):
    with open( path, 'rb') as f:
        return torch.tensor( np.fromfile( f, dtype=dtype))


def loadBin( path: str):
    return loadBinary( path, np.float32)


def loadBinLong( path: str):
    return loadBinary( path, np.int64)


#%%

def cut( data: torch.Tensor, center: tp.Sequence[ int],
         length: tp.Sequence[ int], stride: tp.Sequence[ int],
         interleaved=False, onlyShape=False):

    tcenter = torch.as_tensor( center, dtype=torch.long)
    tlength = torch.as_tensor( length, dtype=torch.long)
    tstride = torch.as_tensor( stride, dtype=torch.long)

    cutDims = len( tcenter)
    assert 0 < cutDims == len( tlength) == len( tstride) <= data.ndim
    dataShape = data.shape[ cutDims :]

    tleft = tcenter - ( tlength >> 1)
    assert tleft.ge( 0).all()

    cutShape = torch.tensor( data.shape[ : cutDims], dtype=torch.long) - tleft + tstride - tlength
    cutShape.div_( tstride, rounding_mode='trunc')
    assert cutShape.ge( 1).all()
    cutShape = tuple( cutShape.tolist())

    if interleaved:
        leadingShape = tuple( zip( cutShape, length))
    else:
        leadingShape = cutShape + tuple( length)

    retShape = leadingShape + dataShape
    if onlyShape:
        return retShape
    ret = torch.empty( retShape, dtype=data.dtype, device=data.device)

    for ids in product( * tuple( map( range, cutShape))):
        if interleaved:
            retSlices = tuple( zip( ids, repeat( slice( None), cutDims)))
        else:
            retSlices = tuple( ids)
        sleft = tleft + tstride * torch.tensor( ids, dtype=torch.long)
        sright = sleft + tlength
        dataSlices = tuple( slice( sl, sr) for sl, sr in zip( sleft, sright))
        ret[ retSlices] = data[ dataSlices]
    return ret
