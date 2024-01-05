#%%

__all__ = ['CoordBlock']


#%%

from .base import CoordBlockBase as CBB

from itertools import product
import typing as tp


#%%

SI = tp.Sequence[ int]
OI = tp.Optional[ int]
TI = tp.Union[ tp.List[ int], tp.Tuple[ int, int], tp.Tuple[ int, int, int]]
TOI = tp.Union[ None, tp.List[ OI], tp.Tuple[ OI, OI], tp.Tuple[ OI, OI, OI]]


class CoordBlock:

    def __init__(self, cbegin: TI, cstep: TI, cnum: TI,
                 fbegin: TOI, fend: TOI, fstep: TOI):

        self.dims = len( cbegin)
        assert 2 <= len( cstep) == len( cnum) == self.dims <= 3
        self.trilinear = self.dims == 3

        none = ( None,) * self.dims
        dup = lambda f: none if f is None else f
        fbegin = dup( fbegin)
        fend = dup( fend)
        fstep = dup( fstep)
        assert len( fbegin) == len( fend) == len( fstep) == self.dims

        self.bases: tp.Sequence[ CBB] = tuple(
            CBB( cbegin[ i], cstep[ i], cnum[ i],
                 fbegin[ i], fend[ i], fstep[ i])
            for i in range( self.dims))

        collect = lambda k: tuple( getattr( b, k) for b in self.bases)

        self.cbegin, self.cstep, self.cnum = cbegin, cstep, cnum
        self.cend = collect( 'cend')

        self.fbegin = collect( 'fbegin')
        self.fstep = collect( 'fstep')
        self.fnum = collect( 'fnum')
        self.fend = collect( 'fend')

    def blockCoord(self, blockIndices: SI):
        return tuple( self.bases[ i].coord[ blockIndices[ i]]
                      for i in range( self.dims))

    def iterInner(self, blockIndices: SI):
        poses, coeff = [], []
        for i in range( self.dims):
            poses.append( self.bases[ i].poses[ blockIndices[ i]])
            coeff.append( self.bases[ i].coeff[ blockIndices[ i]])
        poses = product( * poses)
        coeff = product( * coeff)
        yield from zip( poses, coeff)

    def iterBlockIndices(self):
        yield from product( * ( range( len( self.bases[ i].coord))
                                for i in range( self.dims)))

    def iter(self):
        for bi in self.iterBlockIndices():
            yield self.blockCoord( bi), self.iterInner( bi)
