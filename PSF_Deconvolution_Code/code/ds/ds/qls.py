#%%

__all__ = ['Qikou', 'Layer6', 'Sigsbee2', 'SigsbeeK', 'SigsbeeK2']


#%%

from .rtds import RuntimeInterpolateDataset as RTDS

import torch


#%%

class Qikou( RTDS):

    def __init__(self, forSample: bool, normType: str):

        self.name = 'qikou'
        self.path = 'data/qikou/psf_qikou_{}.pth'.format(
            'smp' if forSample else 'whl')

        data = torch.load( self.path)
        sp = ( 17, 15) if forSample else ( 18, 16)
        sp += ( 50, 50)
        assert data.shape == sp

        super().__init__( data, False, forSample, normType)


class Layer6( RTDS):

    def __init__(self, forSample: bool, normType: str):

        self.name = 'layer6'
        self.path = 'data/layer6/psf_layer6.pth'

        data = torch.load( self.path)
        assert data.shape == ( 19, 19, 50, 50)
        super().__init__( data, False, forSample, normType)


#%%

class Sigsbee2( RTDS):

    def __init__(self, forSample: bool, normType: str):

        self.name = 'sigsbee2'
        self.path = 'data/sigsbee2/psf_sigsbee2_{}.pth'.format(
            'smp' if forSample else 'whl')

        data = torch.load( self.path)
        sp = ( 45, 125) if forSample else ( 46, 126)
        sp += ( 50, 50)
        assert data.shape == sp

        super().__init__( data, False, forSample, normType)


#%%

class SigsbeeK12( RTDS):

    def __init__(self, version: int, forSample: bool, normType: str):

        assert version in [ 1, 2]
        self.name = 'sigsbee_kir' + ['', '2'][ version - 1]
        self.path = 'data/{name}/psf_{name}.pth'.format( name=self.name)

        data = torch.load( self.path)
        assert data.shape == ( 19, 63, 50, 50)

        super().__init__( data, False, forSample, normType)


class SigsbeeK( SigsbeeK12):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 1, forSample, normType)


class SigsbeeK2( SigsbeeK12):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 2, forSample, normType)
