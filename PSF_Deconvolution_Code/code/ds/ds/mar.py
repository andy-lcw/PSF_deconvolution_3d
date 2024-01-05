#%%

__all__ = ['Marmousi2', 'Marmousi2_dis',
           'Marmousi4', 'Marmousi2_dis', 'MarmousiK']


#%%

from .rtds import RuntimeInterpolateDataset as RTDS

import torch


#%%

class Marmousi24( RTDS):

    def __init__(self, version: int, original: bool,
                 forSample: bool, normType: str):

        assert version in [ 2, 4]
        name = f'marmousi{version}'
        nameFull = name + ( '_ori' if original else '_dis')
        self.name = name if original else name + '_dis'
        self.original = original

        self.path = 'data/{name}/psf_{nameFull}_{type}.pth'.format(
            name=name, nameFull=nameFull, type=( 'smp' if forSample else 'whl'))

        data = torch.load( self.path)
        if version == 2:
            sp = 13, ( 27 if forSample else 28)
        else:
            sp = 7, ( 13 if forSample else 14)
        sp += 50, 50
        assert data.shape == sp

        super().__init__( data, False, forSample, normType)


class Marmousi2( Marmousi24):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 2, True, forSample, normType)


class Marmousi2_dis( Marmousi24):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 2, False, forSample, normType)


class Marmousi4( Marmousi24):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 4, True, forSample, normType)


class Marmousi4_dis( Marmousi24):
    def __init__(self, forSample: bool, normType: str):
        super().__init__( 4, False, forSample, normType)


#%%

class MarmousiK( RTDS):
    def __init__(self, forSample: bool, normType: str):

        self.name = 'marmousi_kir'
        self.path = 'data/marmousi_kir/psf_marmousi_kir_{}.pth'.format(
            'smp' if forSample else 'whl')

        data = torch.load( self.path)
        sp = 19, ( 35 if forSample else 36), 20, 20
        assert data.shape == sp

        super().__init__( data, False, forSample, normType)
