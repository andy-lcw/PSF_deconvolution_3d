#%%

__all__ = ['InterpolateBase',
           'RuntimeInterpolate', 'MaskedInterpolate']


#%%

from ..norm import PSFNorm
# noinspection PyUnresolvedReferences
from .interpolate import interpolate2d, interpolate3d

import abc
import random
import typing as tp

import torch
import torch.nn.functional as F
import torch.utils.data as tud


#%%

class InterpolateBase( abc.ABC, tud.Dataset):

    def __init__(self, data: torch.Tensor,
                 trilinear: bool,
                 normType: str,
                 samples: tp.Optional[ int]):

        self.data = data
        self.trilinear = trilinear
        self.normType = normType
        self.samples = samples

        self.intDims = 3 if self.trilinear else 2
        self.intFunc = eval( f'interpolate{self.intDims}d')
        self.normFunc = PSFNorm.norm( self.normType)
        if self.samples is not None:
            assert isinstance( self.samples, int) and self.samples > 0

        shapes = torch.tensor( self.data.shape[ : self.intDims], dtype=torch.long)
        assert shapes.ge( 2).all()
        self.ranges = tuple( shapes.sub( 1).tolist())

        self.kerMask = None

    def registerKerMask(self, kerMask):
        assert self.kerMask is None
        assert kerMask.shape[ : self.intDims] == self.data.shape[ : self.intDims]
        self.kerMask = kerMask

    def __len__(self):
        return self.samples

    def _checkIndex(self, index):
        if self.samples is not None:
            if index >= self.samples or index < - self.samples:
                raise IndexError

    def _interpolate(self, coord: tp.Optional[ tp.Sequence[ int]] = None,
                     coeff: tp.Optional[ tp.Sequence[ float]] = None):
        if coord is None:
            coord = [ random.randrange( r) for r in self.ranges]
        if coeff is None:
            coeff = [ random.random() for _ in range( self.intDims)]
        raw = self.intFunc( self.data, * coord, * coeff)
        normed = self.normFunc( raw)
        if self.kerMask is None:
            return normed
        kerMask = self.intFunc( self.kerMask, * coord, * coeff)
        return normed, PSFNorm.absmax( kerMask)


#%%

class RuntimeInterpolate( InterpolateBase):

    def __getitem__(self, index):
        self._checkIndex( index)
        return self._interpolate()


#%%

class MaskedInterpolate( InterpolateBase):

    def __init__(self, data: torch.Tensor,
                 mask: torch.BoolTensor,
                 trilinear: bool,
                 normType: str,
                 samples: tp.Optional[ int]):

        super().__init__( data, trilinear, normType, samples)
        self.mask = mask

        assert self.mask.shape == self.ranges and self.mask.dtype is torch.bool
        s = self.mask.sum()
        assert s > 0

        if s / self.mask.numel() >= 0.25:
            self.indices = None
        else:
            indices = torch.arange( self.mask.numel()).view_as( self.mask)
            indices = indices[ self.mask]
            self.indices = self.flat2Index( self.ranges, indices)

    def __getitem__(self, index):
        self._checkIndex( index)

        if self.indices is None:
            while True:
                coord = tuple( random.randrange( r) for r in self.ranges)
                if self.mask[ coord]:
                    break
        else:
            cid = random.randrange( self.indices.shape[ 0])
            coord = self.indices[ cid].tolist()

        return self._interpolate( coord)

    @staticmethod
    def flat2Index( shape: tp.Sequence[ int], indices: torch.Tensor):
        assert not indices.dtype.is_floating_point

        prod = torch.as_tensor( shape)
        assert prod.ndim == 1 and not prod.dtype.is_floating_point
        prod = prod.flip( 0)[ :-1].cumprod( 0).flip( 0).tolist()

        r = indices
        indices = []
        for p in prod:
            q = r.div( p, rounding_mode='trunc')
            indices.append( q)
            r = r - q * p
        indices.append( r)

        return torch.stack( indices, -1)

    @classmethod
    def createMask4Nan(cls, data: torch.Tensor, trilinear: bool):
        intDims = 3 if trilinear else 2
        nan = data.view( * data.shape[ : intDims], -1)
        nan = nan.isnan().any( -1).float()
        ker = torch.ones( ( 2,) * intDims)
        mask = getattr( F, f'conv{intDims}d')( nan[ None], ker[ None, None])[ 0]
        return mask.lt( 0.5)
