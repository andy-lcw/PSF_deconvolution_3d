#%%

__all__ = ['Marmousi2', 'Marmousi2_dis',
           'Marmousi4', 'Marmousi4_dis', 'MarmousiK']


#%%

from .base import ImgBase
from ..common import loadBin
from ..common import RuntimeInterpolate as RI, CoordInterpolate as CI


#%%

class Marmousi24( ImgBase):

    def __init__(self, version: int, original: bool):

        assert version in [ 2, 4]
        name = 'Marmousi{ver}{dis}'.format(
            ver=version, dis=( '' if original else '_dis'))

        img = loadBin( 'data/marmousi2/raw/{}_owe_image.rsf@'.format(
            'ori' if original else 'disturb'))
        img /= img.abs().amax()
        img = img.view( 737, 375).T

        if version == 2:
            from ..build.marmousi2 import compose
            topLeft, stride = [ 23, 24], [ 25, 25]
        else:
            from ..build.marmousi4 import compose
            topLeft, stride = [ 48, 49], [ 50, 50]

        psf = compose( original)
        psf /= psf.abs().amax()

        rtint = RI( psf, False, 'none', None)
        cint = CI( rtint, topLeft, stride)

        super().__init__( name, 2, img, psf, ( 50, 50), None, rtint, cint)


class Marmousi2( Marmousi24):
    def __init__(self):
        super().__init__( 2, True)


class Marmousi2_dis( Marmousi24):
    def __init__(self):
        super().__init__( 2, False)


class Marmousi4( Marmousi24):
    def __init__(self):
        super().__init__( 4, True)


class Marmousi4_dis( Marmousi24):
    def __init__(self):
        super().__init__( 4, False)


#%%

class MarmousiK( ImgBase):

    def __init__(self):
        img = loadBin( 'data/marmousi_kir/raw/mar_ori_owe_img_393_737.rsf@')
        img /= img.abs().amax()
        img = img.view( 737, 393).T

        from ..build.marmousi_kir import compose
        psf = compose()
        psf /= psf.abs().amax()

        rtint = RI( psf, False, 'none', None)
        cint = CI( rtint, [ 10, 10], [ 20, 20])

        super().__init__( 'MarmousiK', 2, img, psf, ( 20, 20),
                          None, rtint, cint)
