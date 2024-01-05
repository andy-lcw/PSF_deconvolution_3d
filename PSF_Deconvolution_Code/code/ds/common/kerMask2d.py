#%%

__all__ = ['absKerMask', 'absRangeKerMask']


#%%

from core.shape import centerPadding
import typing as tp

import torch
import torch.nn.functional as F

try:
    from torchvision.transforms.functional import affine, InterpolationMode as IM
except:
    pass


#%%

def centerPad( psf: torch.Tensor, invShape: tp.Sequence[ int]):
    return F.pad( psf, centerPadding( psf.shape[ -2:], invShape))


def absKerMask( psf: torch.Tensor, invShape: tp.Sequence[ int],
                scale: tp.Optional[ float]):

    assert len( invShape) == 2
    ret = centerPad( psf, invShape).abs()
    if scale is not None and scale != 1:
        ret = affine( ret, 0, [ 0, 0], scale, [ 0, 0], IM.BILINEAR)
    return ret


def absRangeKerMask( psf: torch.Tensor, invShape: tp.Sequence[ int],
                     smin: float, smax: float, steps=21):

    assert len( invShape) == 2
    ret = centerPad( psf, invShape).abs()

    scales = []
    for scale in torch.linspace( smin, smax, steps).tolist():
        scales.append( affine( ret, 0, [ 0, 0], scale, [ 0, 0], IM.BILINEAR))
    return torch.stack( scales).amax( dim=0)
