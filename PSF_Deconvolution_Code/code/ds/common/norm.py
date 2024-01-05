#%%

__all__ = ['PSFNorm']


#%%

import torch


#%%

class PSFNorm:

    @staticmethod
    def none( psf: torch.Tensor):
        return psf

    @staticmethod
    def max( psf: torch.Tensor):
        return psf / psf.max()

    @staticmethod
    def std( psf: torch.Tensor):
        return psf / psf.std( unbiased=False)

    @staticmethod
    def absmax( psf: torch.Tensor):
        return psf / psf.abs().max()

    @staticmethod
    def allNone( psfs: torch.Tensor, dims: int):
        return psfs

    @staticmethod
    def allMax( psfs: torch.Tensor, dims: int):
        cache = DimCache._dimCache( dims)
        return psfs / psfs.amax( dim=cache, keepdim=True)

    @staticmethod
    def allStd( psfs: torch.Tensor, dims: int):
        cache = DimCache._dimCache( dims)
        return psfs / psfs.std( dim=cache, unbiased=False, keepdim=True)

    @staticmethod
    def allAbsMax( psfs: torch.Tensor, dims: int):
        cache = DimCache._dimCache( dims)
        return psfs / psfs.abs().amax( dim=cache, keepdim=True)

    @classmethod
    def norm(cls, normType: str):
        if normType == 'none':
            return cls.none
        elif normType == 'max':
            return cls.max
        elif normType == 'std':
            return cls.std
        elif normType == 'absmax':
            return cls.absmax
        else:
            raise KeyError( normType)

    @classmethod
    def allNorm(cls, normType: str):
        if normType == 'none':
            return cls.allNone
        elif normType == 'max':
            return cls.allMax
        elif normType == 'std':
            return cls.allStd
        elif normType == 'absmax':
            return cls.allAbsMax
        else:
            raise KeyError( normType)


#%%

class DimCache:

    _cache = {}

    @classmethod
    def _dimCache(cls, dims: int):
        ret = cls._cache.get( dims, None)
        if ret is None:
            cls._cache[ dims] = ret = tuple( range( -dims, 0))
        return ret
