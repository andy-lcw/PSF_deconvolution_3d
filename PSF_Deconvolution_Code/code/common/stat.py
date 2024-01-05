#%%

__all__ = ['Dots', 'Timer']


#%%

import datetime as dt
import math
import sys
import time
from contextlib import AbstractContextManager

import torch


#%%

class Dots:

    def __init__(self, total: int, dots=10):
        self.total = total
        self.dots = dots
        self.last = 0

    def step(self, now: int, flush=True):
        dots = min( math.floor( now * self.dots / self.total), self.dots)
        delta = dots - self.last
        if delta > 0:
            print( '.' * delta, end='')
            self.last = dots
            if flush:
                sys.stdout.flush()


#%%

class Timer( AbstractContextManager):

    def __init__(self, device):
        self.device = torch.device( device)

        self.cuda = self.device.type == 'cuda'
        self.entered = False

        # cpu
        self.total = 0
        self.last = None

        if self.cuda:
            self.totalGPU = 0
            self.stream = torch.cuda.current_stream( self.device)
            self.begin = torch.cuda.Event( enable_timing=True)
            self.end = torch.cuda.Event( enable_timing=True)

    def __enter__(self):
        assert not self.entered
        self.entered = True
        self.last = time.perf_counter()
        if self.cuda:
            self.begin.record( self.stream)
        return self

    enter = __enter__

    def __exit__(self, eType, eValue, eTrace):
        assert self.entered
        if self.cuda:
            self.end.record( self.stream)
            self.end.synchronize()
            self.totalGPU += self.begin.elapsed_time( self.end) / 1000
        self.total += time.perf_counter() - self.last
        self.last = None
        self.entered = False

    exit = __exit__

    def dt(self):
        ret = dt.timedelta( seconds=round( self.total))
        if self.cuda:
            ret = ret, dt.timedelta( seconds=round( self.totalGPU))
        return ret

    def __str__(self):
        ret = self.dt()
        if self.cuda:
            return f'{ret[ 0]} (GPU: {ret[ 1]})'
        return str( ret[ 0])

    @staticmethod
    def _parse( cost: float, now: float, total: float):
        if not ( 0 < now <= total):
            full = remains = '---'
        else:
            full = cost * total / now
            remains = str( dt.timedelta( seconds=round( full - cost)))
            full = str( dt.timedelta( seconds=round( full)))
        cost = str( dt.timedelta( seconds=round( cost)))
        return ' | '.join( [ cost, remains, full])

    def fullStr(self, now: float, total: float):
        ret = self._parse( self.total, now, total)
        if self.cuda:
            retGPU = self._parse( self.totalGPU, now, total)
            ret = f'{ret} (GPU: {retGPU})'
        return ret
