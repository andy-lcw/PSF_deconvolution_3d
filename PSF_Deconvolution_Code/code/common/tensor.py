#%%

__all__ = ['deepCopy', 'TensorCollector']


#%%

import enum
import os
import pathlib as pl
import typing as tp
from collections import OrderedDict as OD

import torch


#%%

def deepCopy( sth):
    if sth is None:
        return sth
    if isinstance( sth, ( bool, int, float, str, enum.Enum)):
        return sth
    if isinstance( sth, torch.Tensor):
        return sth.detach().to( 'cpu', copy=True)

    if isinstance( sth, ( tuple, list, set)):
        ret = sth.__class__( deepCopy( th) for th in sth)
    elif isinstance( sth, ( dict, OD)):
        ret = sth.__class__( { k: deepCopy( v) for k, v in sth.items()})
    else:
        raise NotImplementedError( type( sth))

    return ret


#%%

class TensorCollector:

    def __init__(self, fileName: str = None):
        self.fileName = fileName
        self.batches = 0
        self.dataShape = None

        if self.fileName is None:
            self.list = []
        else:
            file = pl.Path( self.fileName)
            assert not file.exists()
            file.touch()

    def append(self, tensor: torch.Tensor, batched=True):
        if batched:
            batches, dataShape = tensor.shape[ 0], tensor.shape[ 1:]
        else:
            batches, dataShape = 1, tensor.shape

        if self.dataShape is None:
            self.dataShape = dataShape
        else:
            assert self.dataShape == dataShape
        self.batches += batches

        if self.fileName is None:
            self.list.append( tensor)
        else:
            with open( self.fileName, 'ab') as file:
                file.write( tensor.numpy().tobytes())
        return self

    def bake(self, leadShape: tp.Union[ None, int, tp.Sequence[ int]]):
        if hasattr( leadShape, '__len__') and len( leadShape) == 0:
            leadShape = None
        if leadShape is None:
            assert self.batches <= 1
            shape = self.dataShape if self.batches else ( 0,)
        else:
            if isinstance( leadShape, int):
                leadShape = leadShape,
            shape = torch.empty( self.batches, device='meta')
            shape = shape.view( leadShape).shape
            if self.dataShape is not None:
                shape += self.dataShape

        if self.fileName is not None:
            return dict( file=self.fileName, shape=shape)

        ret = torch.cat( self.list) if self.batches > 0 else torch.empty( 0)
        if shape is not None:
            ret = ret.view( shape)
        return ret

    @classmethod
    def single(cls, inMemory: bool, root: str, name: str):
        return cls() if inMemory else cls( os.path.join( root, name + '.bin'))

    @classmethod
    def wrap(cls, tensor: torch.Tensor, inMemory: bool, root: str, name: str):
        if inMemory:
            return tensor
        return cls.single( False, root, name) \
            .append( tensor, batched=False).bake( None)

    @classmethod
    def list(cls, inMemory: bool, root: str, name: str, subNames: list):
        if inMemory:
            return [ cls() for _ in subNames]

        root = os.path.join( root, name)
        os.makedirs( root)
        return [ cls( os.path.join( root, f'{sn}.bin'))
                 for sn in subNames]

    @classmethod
    def wrapList(cls, tensors: tp.Sequence[ torch.Tensor],
                 inMemory: bool, root: str, name: str, subNames: list):
        assert len( tensors) == len( subNames)
        if inMemory:
            return list( tensors)
        return [ tc.append( t, batched=False).bake( None) for tc, t in
                 zip( cls.list( False, root, name, subNames), tensors)]
