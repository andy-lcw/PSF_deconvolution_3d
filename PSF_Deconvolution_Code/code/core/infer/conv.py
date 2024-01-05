#%%

__all__ = ['fullConv2d', 'fullConv3d', 'fullConv',
           'batchedFullConv2d', 'batchedFullConv3d', 'batchedFullConv',
           'crossFullConv2d', 'crossFullConv3d', 'crossFullConv']


#%%

import torch
import torch.nn.functional as F


#%%

def fullConv2d( img: torch.Tensor, ker: torch.Tensor):
    assert img.ndim == 2
    H, W = ker.shape
    if H == W:
        padding = H - 1
    else:
        img = F.pad( img, [ W-1, W-1, H-1, H-1])
        padding = 0
    return F.conv2d( img[ None, None], ker[ None, None], padding=padding)[ 0, 0]


def fullConv3d( img: torch.Tensor, ker: torch.Tensor):
    assert img.ndim == 3
    D, H, W = ker.shape
    if D == H == W:
        padding = D - 1
    else:
        img = F.pad( img, [ W-1, W-1, H-1, H-1, D-1, D-1])
        padding = 0
    return F.conv3d( img[ None, None], ker[ None, None], padding=padding)[ 0, 0]


def fullConv( img: torch.Tensor, ker: torch.Tensor):
    if img.ndim == 2:
        return fullConv2d( img, ker)
    elif img.ndim == 3:
        return fullConv3d( img, ker)
    else:
        raise NotImplementedError


#%%

def batchedFullConv2d( img: torch.Tensor, ker: torch.Tensor):
    b, h, w = img.shape
    b_, ( H, W) = 1, ker.shape[ -2:]
    if ker.ndim > 2:
        b_, = ker.shape[ :-2]
        assert b == b_

    if H == W:
        padding = H - 1
    else:
        img = F.pad( img, [ W-1, W-1, H-1, H-1])
        padding = 0

    sd = int( b_ <= 1)
    imgR = img.unsqueeze( sd)
    kerR = ker.view( b_, 1, H, W)
    return F.conv2d( imgR, kerR, padding=padding, groups=b_).squeeze( sd)


def batchedFullConv3d( img: torch.Tensor, ker: torch.Tensor):
    b, d, h, w = img.shape
    b_, ( D, H, W) = 1, ker.shape[ -3:]
    if ker.ndim > 3:
        b_, = ker.shape[ :-3]
        assert b == b_

    if D == H == W:
        padding = D - 1
    else:
        img = F.pad( img, [ W-1, W-1, H-1, H-1, D-1, D-1])
        padding = 0

    sd = int( b_ <= 1)
    imgR = img.unsqueeze( sd)
    kerR = ker.view( b_, 1, D, H, W)
    return F.conv3d( imgR, kerR, padding=padding, groups=b_).squeeze( sd)


def batchedFullConv( img: torch.Tensor, ker: torch.Tensor):
    if img.ndim == 3:
        return batchedFullConv2d( img, ker)
    elif img.ndim == 4:
        return batchedFullConv3d( img, ker)
    else:
        raise NotImplementedError


#%%

def crossFullConv2d( img: torch.Tensor, ker: torch.Tensor):
    assert img.ndim == 3
    _, H, W = ker.shape
    if H == W:
        padding = H - 1
    else:
        img = F.pad( img, [ W-1, W-1, H-1, H-1])
        padding = 0

    imgR = img.unsqueeze( 1)
    kerR = ker.unsqueeze( 1)
    return F.conv2d( imgR, kerR, padding=padding)


def crossFullConv3d( img: torch.Tensor, ker: torch.Tensor):
    assert img.ndim == 4
    _, D, H, W = ker.shape
    if D == H == W:
        padding = D - 1
    else:
        img = F.pad( img, [ W-1, W-1, H-1, H-1, D-1, D-1])
        padding = 0

    imgR = img.unsqueeze( 1)
    kerR = ker.unsqueeze( 1)
    return F.conv3d( imgR, kerR, padding=padding)


def crossFullConv( img: torch.Tensor, ker: torch.Tensor):
    if img.ndim == 3:
        return crossFullConv2d( img, ker)
    elif img.ndim == 4:
        return crossFullConv3d( img, ker)
    else:
        raise NotImplementedError
