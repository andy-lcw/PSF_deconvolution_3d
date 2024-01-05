#%%

__all__ = ['pretrain']


#%%

import core.infer, core.loss, core.shape
from core.misc import Steps, Epochs
import ds.img

import math
import typing as tp

import torch
import torch.nn as nn


#%%

def pretrain( ids: ds.img.ImgBase,
              freq: bool, invShape: tp.Sequence[ int],
              optimFunc, Hyper, him, hilm,
              device, batchSize: int, batches: int,
              recordStat=False,
              order=1.):

    lastDims = tuple( range( - ids.dims, 0))
    inv = core.infer.invIdentity( ids.psfShape, invShape)
    if freq:
        inv = torch.view_as_real(
            torch.fft.rfftn( inv, dim=lastDims, norm='forward'))
    scaleFactor = inv.std( False) * math.sqrt( inv.numel())
    inv = nn.Parameter( inv.div_( scaleFactor).to( device))
    optim = optimFunc( [ inv])

    statItems = statEpoch = None
    if recordStat:
        statItems = 'ls l2 rs ct'.split()
        statEpoch = Epochs( * statItems)

    for step in range( batches):
        data = torch.stack( [ ids.int[ None] for _ in range( batchSize)])
        data = data.to( device, non_blocking=True, copy=True)

        adata = data.abs()
        dataScales = adata.amax( dim=lastDims, keepdim=True)
        data.div_( dataScales)
        if 'abs' in [ him, hilm]:
            adata.div_( dataScales)

        masks = him
        if masks == 'abs':
            masks = adata
        lossMasks = hilm
        if lossMasks == 'abs':
            lossMasks = 1 - adata

        now = inv.mul( scaleFactor)
        if freq:
            now = torch.fft.irfftn( torch.view_as_complex( now),
                                    invShape, norm='forward')
        if masks is not None:
            now = now.mul( masks)

        rsts = core.infer.batchedFullConv( data, now)
        if lossMasks is not None:
            now = now * lossMasks
        lss = core.loss.lossFunc( rsts, now, Hyper)

        optim.zero_grad()
        lss.pow( order).mean().backward()
        optim.step()

        if recordStat:
            statStep = Steps( * statItems)
            with torch.inference_mode():
                l2s = core.loss.rawL2LossFunc( rsts)
                rss = core.loss.rawRescaledLossFunc( rsts)
                cts = core.shape.CenterCache.getCenter( rsts)
                for si, ls in zip( statItems, [ lss, l2s, rss, cts]):
                    statStep.update( si, ls)
            statEpoch.update( statStep)

        del data, adata, dataScales, masks, lossMasks, now, rsts, lss

    with torch.inference_mode():
        inv.detach_().mul_( scaleFactor)
        if freq:
            inv = torch.fft.irfftn( torch.view_as_complex( inv),
                                    invShape, norm='forward')
        inv = inv.cpu()

    if recordStat:
        return inv, statEpoch
    return inv
