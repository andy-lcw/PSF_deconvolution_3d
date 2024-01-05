#%%

__all__ = ['ImgBase']


#%%

from core.shape import takePart
from ds.common import CoordInterpolate as CI
from ds.common import RuntimeInterpolate as RI, MaskedInterpolate as MI

from itertools import product
import typing as tp

import torch


#%%

class ImgBase:

    def __init__(self,
                 name: str, dims: int, img: torch.Tensor,
                 psf: torch.Tensor, psfShape: tp.Sequence[ int],
                 mask: tp.Optional[ torch.Tensor],
                 rtint: tp.Union[ RI, MI], cint: CI):
        self.name = name
        self.dims = dims
        self.img = img
        self.psf = psf
        self.psfShape = tuple( psfShape)
        self.mask = mask
        self.int = self.rtint = rtint
        self.cint = cint

    def psfAt(self, pos: tp.Sequence[ int]):
        return self.cint.interpolateAt( pos)

    def imgAt(self, pos: tp.Sequence[ int],
              shape: tp.Optional[ tp.Sequence[ int]] = None):
        if shape is None:
            shape = self.psfShape
        # pos = tuple( p + d for p, d in zip( pos, self.imgPosDelta))
        return takePart( self.img, shape, pos, True)

    def iterItem(self, printDim: tp.Optional[ int],
                 begin: tp.Optional[ tp.Sequence[ int]] = None,
                 end: tp.Optional[ tp.Sequence[ int]] = None,
                 step: tp.Optional[ tp.Sequence[ int]] = None,
                 getPsf=True, getImg=True, allowNonePsf=True,
                 imgShape: tp.Sequence[ int] = None):
        """ pos, psf, img """

        if printDim is None:
            printDim = -1
        else:
            assert -1 <= printDim < self.dims

        if begin is None:
            begin = self.cint.lt
        else:
            assert len( begin) == self.dims

        if end is None:
            end = self.cint.br1
        else:
            assert len( end) == self.dims

        if step is None:
            step = ( 1,) * self.dims
        else:
            assert len( step) == self.dims

        sepDim = printDim + 1
        lines = product( * ( range( bg, ed, st) for bg, ed, st in
                             zip( begin[ :sepDim], end[ :sepDim], step[ :sepDim])))
        cols = lambda: product( * ( range( bg, ed, st) for bg, ed, st in
                                    zip( begin[ sepDim:], end[ sepDim:], step[ sepDim:])))

        for ln in lines:
            if len( ln) > 0:
                print( ' '.join( map( str, ln)))

            for cl in cols():
                pos = ln + cl
                output = [ pos]

                if getPsf:
                    psf = self.psfAt( pos)
                    if psf is None:
                        continue
                    output.append( psf)
                elif not allowNonePsf and not self.cint.getAvailablePos( pos):
                    continue

                if getImg:
                    output.append( self.imgAt( pos, imgShape))

                yield output

    def iterBatch(self, batchSize: int, printDim: tp.Optional[ int],
                  begin: tp.Optional[ tp.Sequence[ int]] = None,
                  end: tp.Optional[ tp.Sequence[ int]] = None,
                  step: tp.Optional[ tp.Sequence[ int]] = None,
                  getPsf=True, getImg=True, allowNonePsf=True,
                  imgShape: tp.Sequence[ int] = None):

        poses = []
        psfs = []
        imgs = []

        def getOutput():
            output = [ poses.copy()]
            if getPsf:
                output.append( psfs.copy())
            if getImg:
                output.append( imgs.copy())
            return output

        for item in self.iterItem( printDim, begin, end, step,
                                   getPsf, getImg, allowNonePsf, imgShape):
            poses.append( item.pop( 0))
            if getPsf:
                psfs.append( item.pop( 0))
            if getImg:
                imgs.append( item.pop( 0))
            if len( poses) >= batchSize:
                yield getOutput()
                poses.clear()
                psfs.clear()
                imgs.clear()

        if len( poses) > 0:
            yield getOutput()

    def validIndices(self, begin: tp.Optional[ tp.Sequence[ int]] = None,
                     end: tp.Optional[ tp.Sequence[ int]] = None,
                     step: tp.Optional[ tp.Sequence[ int]] = None, *,
                     pointList: tp.Iterable[ tp.Sequence[ int]] = None):

        if pointList is not None:
            assert begin is None and end is None and step is None

        else:
            if begin is None:
                begin = self.cint.lt
            else:
                assert len( begin) == self.dims
            if end is None:
                end = self.cint.br1
            else:
                assert len( end) == self.dims
            if step is None:
                step = ( 1,) * self.dims
            else:
                assert len( step) == self.dims

            pointList = product( * ( range( bg, ed, st) for bg, ed, st in
                                     zip( begin, end, step)))

        poses = []
        coord = []
        coeff = []
        for pos in pointList:
            ret = self.cint.getAvailablePos( pos)
            if ret is not None:
                poses.append( pos)
                coord.append( ret[ 0])
                coeff.append( ret[ 1])

        return torch.as_tensor( poses, dtype=torch.long), \
               torch.as_tensor( coord, dtype=torch.long), \
               torch.as_tensor( coeff, dtype=torch.float)

    @classmethod
    def iterCache(cls, printDim: tp.Optional[ int],
                  poses: tp.Sequence[ tp.Tuple[ int, ...]],
                  psfs: tp.Optional[ tp.Sequence[ torch.Tensor]],
                  imgs: tp.Optional[ tp.Sequence[ torch.Tensor]]):

        slc = None
        if printDim is not None and printDim != -1:
            assert 0 <= printDim < len( poses[ 0])
            slc = slice( None, printDim + 1)

        zips = [ poses]
        if psfs is not None:
            zips.append( psfs)
        if imgs is not None:
            zips.append( imgs)

        last = None

        for item in zip( * zips):
            if slc is not None:
                now = item[ 0][ slc]
                if now != last:
                    last = now
                    print( ' '.join( map( str, now)))
            yield item
