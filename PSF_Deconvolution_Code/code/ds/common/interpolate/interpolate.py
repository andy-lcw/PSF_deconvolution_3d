#%%

__all__ = ['interpolate2d', 'interpolate3d',
           'interpolate2d_slow', 'interpolate3d_slow']


#%%

import torch


#%%

def interpolate2d( data: torch.Tensor,
                   y: int, x: int, s: float, r: float):

    yy, xx = y + 1, x + 1
    ss, rr = 1 - s, 1 - r

    ret  = data[ y, x] * ( ss * rr)
    ret += data[ y, xx] * ( ss * r)
    ret += data[ yy, x] * ( s * rr)
    ret += data[ yy, xx] * ( s * r)
    return ret


def interpolate3d( data: torch.Tensor,
                   z: int, y: int, x: int,
                   t: float, s: float, r: float):

    zz, yy, xx = z + 1, y + 1, x + 1
    tt, ss, rr = 1 - t, 1 - s, 1 - r

    ret  = data[ z, y, x] * ( tt * ss * rr)
    ret += data[ z, y, xx] * ( tt * ss * r)
    ret += data[ z, yy, x] * ( tt * s * rr)
    ret += data[ z, yy, xx] * ( tt * s * r)
    ret += data[ zz, y, x] * ( t * ss * rr)
    ret += data[ zz, y, xx] * ( t * ss * r)
    ret += data[ zz, yy, x] * ( t * s * rr)
    ret += data[ zz, yy, xx] * ( t * s * r)
    return ret


#%%

def interpolate2d_slow( data: torch.Tensor,
                        y: int, x: int, s: float, r: float):
    s = torch.tensor( [ 1-s, s]).view( 2, 1)
    r = torch.tensor( [ 1-r, r])
    return torch.tensordot( s*r, data[ y:y+2, x:x+2], 2)


def interpolate3d_slow( data: torch.Tensor,
                        z: int, y: int, x: int,
                        t: float, s: float, r: float):
    t = torch.tensor( [ 1-t, t]).view( 2, 1, 1)
    s = torch.tensor( [ 1-s, s]).view( 2, 1)
    r = torch.tensor( [ 1-r, r])
    return torch.tensordot( t*s*r, data[ z:z+2, y:y+2, x:x+2], 3)
