#%%

__all__ = ['AllDsName', 'dsShortName',
           'buildHyper', 'HyperInvMask', 'HyperInvLossMask']


#%%

import core.loss.common
from core.misc import DictObjType

import typing as tp


#%%

_AllDs = dict(
    Marmousi2='m2', Marmousi2_dis='m2d',
    Marmousi4='m4', Marmousi4_dis='m4d', MarmousiK='mk',
    Qikou='q', Layer6='l6', Sigsbee2='s2', SigsbeeK='sk', SigsbeeK2='sk2',
    Qikou3d='q3d', Qikou3d2='q3d2', Qikou3d3='q3d3',
    Qikou3d4='q3d4', Qikou3dK='q3dk',
    Over3d='o3d', Over3d2='o3d2',
    Over3d3_raw='o3d3r', Over3d3_half='o3d3h', Over3d3_quater='o3d3q',
    Over3dK='o3dk', Over3dK2='o3dk2')


AllDsName = list( _AllDs.keys())


def dsShortName( dsName: str):
    return _AllDs[ dsName]


#%%

def buildHyper( lossType: int,
                psfShape: tp.Sequence[ int],
                invShape: tp.Sequence[ int]):

    dims = len( psfShape)
    assert len( invShape) == dims

    class Hyper( metaclass=DictObjType):

        if lossType == 0:
            loss = 'plain'
        elif lossType in [ 1, 2]:
            loss = 'rescaled'
        elif lossType in [ 3, 4, 5, 6, 7, 8, 9]:
            loss = 'posRescaled'
        else:
            raise NotImplementedError

        if lossType >= 1:
            cClampMin = 0.1
            cClampMax = None
            cCenterRate = None

        if lossType >= 2:
            cCenterRate = 1e-4
            cCenterThres = 2
            cCenterBeta = None
            cCenterDelta = 3

        if lossType >= 3:
            cCenterNegRate = 10
            cCenterNegThres = 0.1
            cCenterNegBeta = None
            cCenterNegDelta = None

        cBeta = cDelta = None
        maskType = 'mean'
        plainRate = 1
        diffRate = diff2Rate = None

        invMask = invLossMask = None
        invHalf = invLossHalf = None
        invL1Rate = invL2Rate = None

        if lossType in [ 4, 5]:
            assert tuple( psfShape) == tuple( invShape)

            invMask = 'abs'
            if lossType == 5:
                invLossMask = 'abs'
                invL1Rate = 0.01

        elif lossType in [ 6, 7]:
            invMask = 'recip'
            invHalf = ( 2.5,) * dims
            if lossType == 7:
                invLossMask = 'recip'
                invLossHalf = ( 2.5,) * dims
                invL1Rate = 0.01

        elif lossType in [ 8, 9]:
            invMask = 'gaussian'
            invHalf = ( 2.5,) * dims
            if lossType == 7:
                invLossMask = 'gaussian'
                invLossHalf = ( 2.5,) * dims
                invL1Rate = 0.01

        # if tuple( invShape) != ( 50, 50) and lossType in [ 6, 7, 8, 9]:
        #     if len( invShape) == 2:
        #         wn.warn( 'masks in masked losses are not adjusted for '
        #                  'inverse kernels of shape other than (50, 50).')
        #     else:
        #         raise NotImplementedError

    return Hyper


def _getMask( mask, half, inverse, invShape, device):
    if mask is None or mask == 'abs':
        return mask
    return core.loss.common.MaskCache.getMask(
        mask, half, invShape, 0, False, device, inverse)


def HyperInvMask( Hyper, invShape, device):
    return _getMask( Hyper.invMask, Hyper.invHalf, False, invShape, device)


def HyperInvLossMask( Hyper, invShape, device):
    return _getMask( Hyper.invLossMask, Hyper.invLossHalf, True,
                     invShape, device)
