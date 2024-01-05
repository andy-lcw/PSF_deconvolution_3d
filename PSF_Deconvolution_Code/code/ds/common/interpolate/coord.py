#%%

__all__ = ['CoordInterpolate']


#%%

from .runtime import RuntimeInterpolate, MaskedInterpolate

import typing as tp

import torch


#%%

class CoordInterpolate:

    def __init__(self, ip: tp.Union[ RuntimeInterpolate, MaskedInterpolate],
                 leftTop: tp.Sequence[ int], grids: tp.Sequence[ int]):
        self.ip = ip
        self.lt = tuple( leftTop)
        self.gs = tuple( grids)

        self.intDims = self.ip.intDims
        assert self.intDims == len( self.lt) == len( self.gs)

        self.br = tuple( self.lt[ i] + self.ip.ranges[ i] * self.gs[ i]
                         for i in range( self.intDims))
        self.br1 = tuple( br + 1 for br in self.br)

        self.masked = isinstance( self.ip, MaskedInterpolate)

    def getAvailablePos(self, pos: tp.Sequence[ int], onlyCheck=False):

        assert len( pos) == self.intDims
        pos = [ p - l for p, l in zip( pos, self.lt)]
        # assert all( p >= 0 for p in pos)

        possible = []
        offset = []

        for i in range( self.intDims):
            q, r = divmod( pos[ i], self.gs[ i])
            qmax = self.ip.ranges[ i]
            if q < 0 or q > qmax or ( q == qmax and r > 0):
                raise IndexError
            if q == 0 or r > 0:
                possible.append( slice( q, q + 1))
                offset.append( q)
            # q > 0 and r == 0
            elif q == qmax:
                possible.append( slice( q - 1, q))
                offset.append( q - 1)
            else:
                possible.append( slice( q - 1, q + 1))
                offset.append( q - 1)

        if self.masked:
            possible = tuple( possible)
            sliced = self.ip.mask[ possible]
            if not sliced.any():
                if onlyCheck:
                    return False
                return None
            elif onlyCheck:
                return True
            coord = MaskedInterpolate.flat2Index( sliced.shape,
                                                  sliced.to( torch.uint8).argmax())
            coord += torch.tensor( offset)
            coord = coord.tolist()
        elif onlyCheck:
            return True
        else:
            coord = offset
        coeff = [ ( p - c * g) / g for p, c, g in zip( pos, coord, self.gs)]
        return coord, coeff

    def interpolateAt(self, pos: tp.Sequence[ int], *, _debug=False):
        cc = self.getAvailablePos( pos, onlyCheck=False)
        if _debug or cc is None:
            return cc
        return self.ip._interpolate( *cc)
