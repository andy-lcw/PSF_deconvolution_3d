#%%

__all__ = ['ImgCoordBlock', 'ImgCoordBlockDataset']


#%%

from ..common.block.nested import CoordBlock, TOI
from .base import ImgBase

from itertools import product
import typing as tp

import torch
import torch.utils.data as tud


#%%

class ImgCoordBlock:

    def __init__(self, img: ImgBase,
                 fbegin: TOI = None, fend: TOI = None, fstep: TOI = None, *,
                 coordRange=None):

        self.img = img

        if coordRange is not None:
            assert fbegin is None and fend is None and fstep is None
            if isinstance( coordRange, torch.Tensor):
                coordRange = coordRange.tolist()
            else:
                coordRange = tuple( coordRange)
            fbegin, fend = coordRange[ :2]
            if len( coordRange) > 2:
                fstep, = coordRange[ 2:]

        self.cb = cb = CoordBlock(
            img.cint.lt, img.cint.gs, img.psf.shape[ : img.dims],
            fbegin, fend, fstep)
        assert cb.dims == img.dims
        assert tuple( cb.cnum) == img.psf.shape[ : img.dims]

    def getBases(self, blockIndices: tp.Sequence[ int]):
        cd = self.cb.blockCoord( blockIndices)
        cd1 = tuple( c + 1 for c in cd)
        prod = product( * zip( cd, cd1))
        return torch.stack( tuple( self.img.psf[ p] for p in prod))

    def getCoeff(self, coeff: tp.Sequence[ float]):
        vv, uu = coeff[ -2:]
        v, u = 1 - vv, 1 - uu
        ret = v * u, v * uu, vv * u, vv * uu

        if self.img.dims == 3:
            ww, = coeff[ :-2]
            w = 1 - ww
            ret = tuple( w * r for r in ret) + tuple( ww * r for r in ret)
        return torch.tensor( ret, dtype=torch.float)


#%%

class ImgCoordBlockDataset( tud.IterableDataset):

    def __init__(self, icb: ImgCoordBlock,
                 reqImage: bool, imageShape: tp.Sequence[ int] = None):
        self.icb = icb
        self.reqImage = reqImage
        self.imageShape = imageShape

        self.img, self.cb = icb.img, icb.cb

    def getLoader(self, batchSize: int, numWorkers: tp.Optional[ int],
                  pinMemory: bool):
        self.batchSize = batchSize
        return tud.DataLoader(
            self, None, num_workers=numWorkers,
            pin_memory=pinMemory, collate_fn=self._dummyCollate)

    def __iter__(self):
        """ bases, poses, coeff, image/None """
        winfo = tud.get_worker_info()
        biter = self.icb.cb.iterBlockIndices()
        if winfo is not None:
            wnum, wid = winfo.num_workers, winfo.id
            biter = ( bi for i, bi in enumerate( biter) if i % wnum == wid)

        for bi in biter:
            bases = self.icb.getBases( bi)
            bpc = self.bpcIter( bi)
            while True:
                try:
                    yield bases, * next( bpc)
                except StopIteration:
                    break

    def __getitem__(self, _):
        raise NotImplementedError

    def __len__(self):
        return torch.tensor( self.cb.fnum, dtype=torch.long).prod().item()

    def bpcIter(self, blockIndices: tp.Sequence[ int]):
        """ block: poses coeff imgs/None """
        if not hasattr( self, 'batchSize') or self.batchSize is None:
            self.batchSize = 1
        iterator = self.cb.iterInner( blockIndices)

        while True:
            poses, coeff, imgs = [], [], None
            if self.reqImage:
                imgs = []

            stat = -1
            try:
                for _ in range( self.batchSize):
                    p, c = next( iterator)
                    poses.append( p)
                    coeff.append( self.icb.getCoeff( c))
                    if self.reqImage:
                        imgs.append( self.img.imgAt( p, self.imageShape))
                    stat = 1
            except StopIteration:
                if stat == 1:
                    stat = 0

            if stat >= 0:
                poses = torch.tensor( poses, dtype=torch.long)
                coeff = torch.stack( coeff)
                if self.reqImage:
                    imgs = torch.stack( imgs)
                yield poses, coeff, imgs

            if stat <= 0:
                return

    @staticmethod
    def _dummyCollate( contents):
        return contents
