#%%

__all__ = ['Qikou3d', 'Qikou3d2', 'Qikou3d3', 'Qikou3d4', 'Qikou3dK',
           'Over3d', 'Over3d2',
           'Over3d3_raw', 'Over3d3_half', 'Over3d3_quater',
           'Over3dK', 'Over3dK2']


#%%

from .rtds import RuntimeInterpolateDataset as RTDS

import torch


#%%

class Qikou3d( RTDS):

    def __init__(self, forSample: bool, normType: str):

        self.name = 'qikou3d'
        self.path = 'data/qikou3d/psf_qikou3d.pth'

        data = torch.load( self.path)
        assert data.shape == ( 7, 7, 7, 21, 21, 21)
        super().__init__( data, True, forSample, normType)


#%%

class Qikou3d234( RTDS):

    def __init__(self, version: int, forSample: bool, normType: str):

        assert version in [ 2, 3, 4]
        self.name = f'qikou3d{version}'
        self.path = 'data/{name}/psf_{name}_{type}.pth'.format(
            name=self.name, type=( 'smp' if forSample else 'whl'))

        data = torch.load( self.path)
        if forSample:
            sp = [ ( 35, 21, 29), ( 35, 21, 29), ( 39, 21, 21)]
        else:
            sp = [ ( 35, 22, 30), ( 35, 22, 30), ( 40, 22, 22)]
        sp = sp[ version - 2] + ( 21, 21, 21)
        assert data.shape == sp

        super().__init__( data, True, forSample, normType)


class Qikou3d2( Qikou3d234):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 2, forSample, normType)


class Qikou3d3( Qikou3d234):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 3, forSample, normType)


class Qikou3d4( Qikou3d234):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 4, forSample, normType)


#%%

class Qikou3dK( RTDS):

    def __init__(self, forSample: bool, normType: str):

        self.name = 'qikou3d_kir'
        self.path = 'data/qikou3d_kir/psf_qikou3d_kir_{}.pth'.format(
            'smp' if forSample else 'whl')

        data = torch.load( self.path)
        sp = ( 33, 21, 29) if forSample else ( 33, 22, 30)
        sp += ( 21, 21, 21)
        assert data.shape == sp

        super().__init__( data, True, forSample, normType)


#%%

class Over3d12( RTDS):

    def __init__(self, version: int, forSample: bool, normType: str):

        assert version in [ 1, 2]
        self.name = 'over3d' + ['', '2'][ version - 1]
        self.path = 'data/{name}/psf_{name}_{type}.pth'.format(
            name=self.name, type=( 'smp' if forSample else 'whl'))

        data = torch.load( self.path)
        sp = ( 7, 13, 13) if forSample else ( 7, 14, 14)
        sp += ( [ 80, 20][ version - 1], 20, 20)
        assert data.shape == sp

        super().__init__( data, True, forSample, normType)


class Over3d( Over3d12):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 1, forSample, normType)


class Over3d2( Over3d12):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 2, forSample, normType)


#%%

class Over3d3s( RTDS):

    def __init__(self, name: str, forSample: bool, normType: str):

        assert name in 'raw half quater'.split()
        self.name = 'over3d3_' + name
        self.path = 'data/over3d3/psf_{name}_{type}.pth'.format(
            name=self.name, type=( 'smp' if forSample else 'whl'))

        data = torch.load( self.path)
        sp = ( 9, 13, 13) if forSample else ( 9, 14, 14)
        sp += ( 20, 20, 20)
        assert data.shape == sp

        super().__init__( data, True, forSample, normType)


class Over3d3_raw( Over3d3s):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 'raw', forSample, normType)


class Over3d3_half( Over3d3s):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 'half', forSample, normType)


class Over3d3_quater( Over3d3s):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 'quater', forSample, normType)


#%%

class Over3dK12( RTDS):

    def __init__(self, version: int, forSample: bool, normType: str):

        assert version in [ 1, 2]
        self.name = 'over3d_kir' + ['', '2'][ version - 1]
        self.path = 'data/{name}/psf_{name}_{type}.pth'.format(
            name=self.name, type=( 'smp' if forSample else 'whl'))

        data = torch.load( self.path)
        if forSample:
            sp = [ 5, 7][ version - 1], 13, 13
        else:
            sp = [ 6, 7][ version - 1], 14, 14
        sp += ( 20, 20, 20)
        assert data.shape == sp

        super().__init__( data, True, forSample, normType)


class Over3dK( Over3dK12):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 1, forSample, normType)


class Over3dK2( Over3dK12):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 2, forSample, normType)
