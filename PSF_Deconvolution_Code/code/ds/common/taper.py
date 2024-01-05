#%%

__all__ = ['taper']


#%%

import typing as tp

import torch


#%%

def taper( shape: tp.Sequence[ int],
           edge: tp.Sequence[ int],
           dtype=None, eps=1e-4):

    assert len( shape) == len( edge)

    tshape = torch.tensor( shape, dtype=torch.long)
    tedge = torch.tensor( edge, dtype=torch.long)
    assert tshape.ge( tedge.mul( 2).sub( 1)).all()

    if dtype is None:
        dtype = torch.get_default_dtype()
    else:
        dtype = torch.dtype( dtype)

    ret = torch.ones( * shape, dtype=dtype)

    for sp, eg, pad in zip( shape, edge, reversed( range( len( shape)))):
        curve = torch.arange( eg).to( dtype).mul( torch.pi / ( eg - 1))
        curve = curve.cos().neg().add( 1).div( 2)
        curve[ 0] = 0
        curve[ -1] = 1

        if sp >= eg * 2:
            taped = [ curve, torch.ones( sp - eg * 2, dtype=dtype), curve.flip( 0)]
        else:
            taped = [ curve, curve[ :-1].flip( 0)]
        taped = torch.cat( taped)
        d = ( sp,) + ( 1,) * pad
        ret *= taped.view( d)

    for i, sp in enumerate( shape):
        slcs = ( slice( None, None),) * i + ( slice( None, None, sp - 1),)
        ret[ slcs] = eps

    return ret
