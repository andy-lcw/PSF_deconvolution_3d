#%%

__all__ = ['detach', 'findAndRedirect', 'gimshow']


#%%

from core.misc import DictObj

import os
import pathlib as pl
import sys
import typing as tp

import torch


#%%

def detach():
    if os.fork() > 0:
        exit()
    os.setsid()
    if os.fork() > 0:
        exit()


def findAndRedirect( savePath: str, saveName: str,
                     runInBackground: bool):

    savePath = pl.Path( savePath)
    savePath.mkdir( parents=True, exist_ok=True)

    index = -1
    while True:
        index += 1
        sn = f'{saveName}.{index}' if index else saveName
        saveTarget = savePath / sn
        if not saveTarget.exists():
            saveTarget.mkdir( parents=True)
            break

    ret = DictObj( _customName='Redirect',
                   index=( str( index) if index else ''),
                   root=str( saveTarget), out=None)

    if runInBackground:
        ret.out = str( saveTarget / 'output.txt')
        print( 'output is redirected to:', ret.out)
        detach()
        redirect = open( ret.out, mode='w', buffering=1)
        os.dup2( redirect.fileno(), sys.stdout.fileno())
        os.dup2( redirect.fileno(), sys.stderr.fileno())

    return ret


#%%

def gimshow( img: torch.Tensor, figSize: tp.Sequence[ int] = None,
             vrange=( None, None), colorBar=False,
             fileName=None, pureSave=False):

    if fileName is None:
        assert not pureSave

    import matplotlib.pyplot as plt

    plt.figure( figsize=figSize)
    plt.axis( False)
    plt.imshow( img.detach().cpu(), cmap='gray',
                vmin=vrange[ 0], vmax=vrange[ 1])
    if colorBar:
        plt.colorbar()
    plt.tight_layout()

    if fileName is not None:
        plt.savefig( fileName)
    if pureSave:
        plt.close()
    else:
        plt.show()
        if not plt.isinteractive():
            plt.close()
