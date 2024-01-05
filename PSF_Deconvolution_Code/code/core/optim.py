#%%

__all__ = ['Func', 'Constant', 'Chain', 'FuncFunc',
           'LinearFunc', 'CosineFunc', 'CosPowFunc',
           'AdamScheduler']


#%%

import abc
import math
import typing as tp

import torch.optim


#%%

class Func( abc.ABC):
    @abc.abstractmethod
    def __call__(self, rate: float): ...


class Constant( Func):
    def __init__(self, c: float):
        self.c = c

    def __call__(self, rate: float):
        return self.c


class Chain( Func):
    def __init__(self, funcs: tp.Sequence[ Func],
                 percentages: tp.Sequence[ float]):
        self.funcs = tuple( funcs)
        self.percentages = tuple( percentages)
        assert len( self.funcs) == len( self.percentages) + 1 > 1
        assert 0 < percentages[ 0] and percentages[ -1] < 1
        for p in range( 1, len( self.percentages)):
            assert self.percentages[ p - 1] < self.percentages[ p]

    def __call__(self, rate: float):
        index = 0
        length = len( self.percentages)
        while index < length and self.percentages[ index] <= rate:
            index += 1
        left = 0 if index == 0 else self.percentages[ index - 1]
        right = 1 if index == length else self.percentages[ index]
        return self.funcs[ index]( ( rate - left) / ( right - left))


class FuncFunc( Func):
    def __init__(self, a: float, b: float, func: tp.Callable[ [ float], float]):
        self.a, self.b = a, b
        self.func = func

    def __call__(self, rate: float):
        return ( self.b - self.a) * self.func( rate) + self.a


class LinearFunc( FuncFunc):
    def __init__(self, a: float, b: float):
        super().__init__( a, b, lambda p: p)


class CosineFunc( FuncFunc):
    def __init__(self, a: float, b: float):
        super().__init__( b, a, lambda p: ( math.cos( math.pi * p) + 1) / 2)


class CosPowFunc( FuncFunc):
    def __init__(self, a: float, b: float, alpha: float):
        self.left = alpha >= 1
        self.alpha = alpha if self.left else 1 / alpha

        def func( p: float):
            p = p if self.left else 1 - p
            p = p ** self.alpha
            p = p if self.left else 1 + p
            return ( math.cos( math.pi * p) + 1) / 2

        super().__init__( b, a, func)


#%%

class AdamScheduler:

    _FuncType = tp.Union[ Func, tp.Sequence[ Func]]

    def __init__(self, optim: torch.optim.Adam,
                 steps: int, lrFunc: _FuncType,
                 betaFunc: _FuncType, beta2Func: _FuncType,
                 last=-1, fakeOptim=True):

        def _wrap( _f):
            _len = len( self.optim.param_groups)
            if isinstance( _f, Func):
                return ( _f,) * _len
            _f = list( _f)
            assert len( _f) == _len
            return _f

        self.optim = optim

        self.steps = steps
        self.lrFunc = _wrap( lrFunc)
        self.betaFunc = _wrap( betaFunc)
        self.beta2Func = _wrap( beta2Func)

        self.last = last
        self.step( fake=False)

        self.fakeOptim = fakeOptim

    def zero_grad(self):
        if not self.fakeOptim:
            raise RuntimeError
        self.optim.zero_grad()

    def step(self, now: int = None, fake: bool = None):
        if fake is None:
            fake = self.fakeOptim
        if fake:
            self.optim.step()

        if now is None:
            now = self.last + 1
        assert 0 <= now <= self.steps
        rate = now / self.steps

        for i, pg in enumerate( self.optim.param_groups):
            lr = self.lrFunc[ i]( rate)
            beta = self.betaFunc[ i]( rate)
            beta2 = self.beta2Func[ i]( rate)
            pg['lr'] = lr
            pg['betas'] = beta, beta2

        self.last = now
