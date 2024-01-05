#%%

__all__ = ['RuntimeInterpolateDataset']


#%%

from ..common import PSFNorm, RuntimeInterpolate

import torch


#%%

class RuntimeInterpolateDataset:

    def __init__(self, data: torch.Tensor, trilinear: bool,
                 forSample: bool, normType: str):

        self.data = data
        self.trilinear = trilinear
        self.forSample = forSample
        self.normType = normType

        self.dim = 3 if self.trilinear else 2
        assert self.data.ndim == self.dim * 2
        self.slices = ( slice( None, None, 2),) * self.dim

        self.allNormFunc = PSFNorm.allNorm( self.normType)

        self.kerMask = None

    def registerKerMask(self, kerMask: torch.Tensor):
        assert self.kerMask is None and \
               kerMask.ndim == self.data.ndim and \
               kerMask.shape[ : self.dim] == self.data.shape[ : self.dim]
        self.kerMask = kerMask

    def train(self, samples: int):
        if not self.forSample:
            raise NotImplementedError
        ret = RuntimeInterpolate( self.data[ self.slices], self.trilinear,
                                  self.normType, samples)
        if self.kerMask is not None:
            ret.registerKerMask( self.kerMask[ self.slices])
        return ret

    def test(self, samples: int):
        if not self.forSample:
            raise NotImplementedError
        ret = RuntimeInterpolate( self.data, self.trilinear,
                                  self.normType, samples)
        if self.kerMask is not None:
            ret.registerKerMask( self.kerMask)
        return ret

    def all(self):
        ret = self.allNormFunc(
            self.data.flatten( 0, self.dim - 1), self.dim)
        if self.kerMask is None:
            return ret
        return ret, PSFNorm.allAbsMax(
            self.kerMask.flatten( 0, self.dim - 1), self.dim)

    def novel(self):
        if not self.forSample:
            raise NotImplementedError
        novelMask = torch.ones( self.data.shape[ : self.dim], dtype=torch.bool)
        novelMask[ self.slices] = False
        ret = self.allNormFunc( self.data[ novelMask], self.dim)
        if self.kerMask is None:
            return ret
        return ret, PSFNorm.allAbsMax( self.kerMask[ novelMask], self.dim)

    def test4train(self, samples: int):
        assert not self.forSample
        ret = RuntimeInterpolate( self.data, self.trilinear,
                                  self.normType, samples)
        if self.kerMask is not None:
            ret.registerKerMask( self.kerMask)
        return ret
