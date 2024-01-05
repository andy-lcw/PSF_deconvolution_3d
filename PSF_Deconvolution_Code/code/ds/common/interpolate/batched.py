#%%

__all__ = ['batchedInterpolate']


#%%

from .interpolate import interpolate2d, interpolate3d

from functools import reduce
from itertools import product, chain
import typing as tp
import torch


#%%

def batchedInterpolate( data: torch.Tensor,
                        grids: tp.Sequence[ int], *, _debug=False):

    intDims = len( grids)
    intShapes = data.shape[ : intDims]
    assert all( sp > 1 for sp in intShapes)
    dataShapes = data.shape[ intDims :]

    def genCoeff():
        shape = [ 1] * intDims
        for ig, g in enumerate( grids):
            shape[ ig] = g + 1
            co = torch.arange( g+1).double().div( g).to( device=data.device)
            yield torch.stack( [ co.flip( 0), co]).view( 2, * shape)
            shape[ ig] = 1

    coeffs = list( genCoeff())

    dv = data.view( * data.shape, * [ 1] * intDims)
    ret = 0
    for ds in product( * [ range( 2) for _ in range( intDims)]):
        slcs = tuple( slice( None, -1) if not d else slice( 1, None) for d in ds)
        coeff = reduce( lambda x, y: x * y, ( co[ d] for co, d in zip( coeffs, ds)))
        ret += dv[ slcs] * coeff
    ret = ret.to( data.dtype).movedim(
        list( range( - intDims, 0)), list( range( 1, 2*intDims, 2)))

    if _debug:
        gs = ds = target = None
        try:
            if intDims == 2:
                interpolate = interpolate2d
            elif intDims == 3:
                interpolate = interpolate3d
            else:
                raise NotImplementedError

            tgrids = torch.tensor( grids).double()
            for gs in product( * [ range( g + 1) for g in grids]):
                t = torch.tensor( gs).div( tgrids).tolist()
                for ds in product( * [ range( sp - 1) for sp in intShapes]):
                    target = interpolate( data, * ds, * t)
                    indices = tuple( chain.from_iterable( zip( ds, gs)))
                    assert target.sub( ret[ indices]).abs().le( 1e-6).all()

        except AssertionError:
            print( 'failed at:', ds, gs)
            return ret, target
        print( 'passed')
        return ret, None

    flatShapes = tuple( ( sp - 1) * g + 1 for sp, g in zip( intShapes, grids))
    result = torch.empty( * flatShapes, * dataShapes,
                          dtype=data.dtype, device=data.device)

    for ds in product( * [ range( 2) for _ in range( intDims)]):
        rstIndices = tuple( slice( None, -1) if not d else slice( -1, None)
                            for d in ds)
        retIndices = tuple( chain.from_iterable(
            ( slice( None, None), slice( None, -1)) if not d else ( -1, -1) for d in ds ))
        rsp = tuple( sp - 1 if not d else 1 for d, sp in zip( ds, flatShapes))
        result[ rstIndices] = ret[ retIndices].reshape( * rsp, * dataShapes)

    gridIndices = tuple( slice( None, None, g) for g in grids)
    result[ gridIndices] = data
    return result
