#%%

__all__ = ['ImgDatasetBase', 'ImgDataset',
           'ImgIterDatasetBase', 'ImgIterDataset']


#%%

from .base import ImgBase
from ..common import RuntimeInterpolate as RI, CoordInterpolate as CI

from itertools import product
import typing as tp

import torch
import torch.utils.data as tud


#%%

class ImgDatasetBase( tud.Dataset):

    def __init__(self, poses: torch.Tensor, coord: torch.Tensor,
                 coeff: torch.Tensor,
                 getImage: tp.Optional[ tp.Callable[
                     [ tp.Sequence[ int]], torch.Tensor]],
                 * rtints: RI):

        assert poses.ndim == 2 and poses.shape == coord.shape == coeff.shape
        self.poses = poses
        self.coord, self.coeff = coord, coeff

        self.getImage = None
        if getImage is not None:
            self.getImage = staticmethod( getImage)

        for rt in rtints:
            assert rt.intDims == self.poses.shape[ 1]
        self.rtints = rtints

    def __len__(self):
        return self.poses.shape[ 0]

    def __getitem__(self, index: int):
        """ pos, [img,] others """
        pos = self.poses[ index]
        ret = pos,
        if self.getImage is not None:
            ret += self.getImage( pos.tolist()),

        return ret + tuple( rt._interpolate( self.coord[ index].tolist(),
                                             self.coeff[ index].tolist())
                            for rt in self.rtints)

    def getLoader(self, batchSize, numWorkers, pinMemory=True):
        return tud.DataLoader(
            self, batch_size=batchSize,
            num_workers=numWorkers, pin_memory=pinMemory)


#%%

class ImgDataset( ImgDatasetBase):

    def __init__(self, img: ImgBase, poses: torch.Tensor,
                 coord: torch.Tensor, coeff: torch.Tensor,
                 imgShape: tp.Sequence[ int]):

        self.img = img
        self.imgShape = tuple( imgShape)
        super().__init__( poses, coord, coeff,
                          lambda ps: self.img.imgAt( ps, self.imgShape),
                          self.img.int)

    def __getitem__(self, index: int):
        """ pos, psf, img """
        ret = super()[ index]
        return ret[ 0], ret[ 2], ret[ 1]


#%%

class ImgIterDatasetBase( tud.IterableDataset):

    def __init__(self, coordRange: torch.Tensor,
                 getImage: tp.Optional[ tp.Callable[
                     [ tp.Sequence[ int]], torch.Tensor]],
                 * cints: CI, skipNone: bool):

        assert coordRange.ndim == 2 and \
               coordRange.shape[ 0] in [ 2, 3] and \
               coordRange.dtype == torch.long
        self.coordRange = coordRange

        self.getImage = None
        if getImage is not None:
            self.getImage = staticmethod( getImage)

        for c in cints:
            assert c.intDims == self.coordRange.shape[ 1]
        self.cints = cints

        self.skipNone = skipNone

    def __len__(self):
        num = self.coordRange[ 1].sub( self.coordRange[ 0])
        if self.coordRange.shape[ 0] == 3:
            num.sub_( 1).div_( self.coordRange[ 2],
                               rounding_mode='trunc').add_( 1)
        return num.prod().item()

    def __iter__(self):
        """ pos, [img,] others """

        workerInfo = tud.get_worker_info()
        numWorkers = workerID = None
        if workerInfo is not None:
            numWorkers, workerID = workerInfo.num_workers, workerInfo.id

        for index, pos in enumerate( product( * (
                range( * ir) for ir in self.coordRange.T.tolist()))):
            if numWorkers is not None and index % numWorkers != workerID:
                continue

            others = tuple( c.interpolateAt( pos) for c in self.cints)
            if self.skipNone and None in others:
                continue
            ret = torch.tensor( pos, dtype=torch.long),
            if self.getImage is not None:
                ret += self.getImage( pos),
            yield ret + others

    def getLoader(self, batchSize, numWorkers, pinMemory=True):
        return tud.DataLoader(
            self, batch_size=batchSize,
            num_workers=numWorkers, pin_memory=pinMemory)

    def __getitem__(self, _):
        raise NotImplementedError


#%%

class ImgIterDataset( ImgIterDatasetBase):

    def __init__(self, img: ImgBase, coordRange: torch.Tensor,
                 imgShape: tp.Sequence[ int]):

        self.img = img
        self.imgShape = tuple( imgShape)
        super().__init__( coordRange,
                          lambda ps: self.img.imgAt( ps, self.imgShape),
                          self.img.cint, skipNone=True)

    def __iter__(self):
        """ pos, psf, img """
        for ret in super().__iter__():
            yield ret[ 0], ret[ 2], ret[ 1]

    def __getitem__(self, _):
        raise NotImplementedError
