#%%

__all__ = ['Qikou', 'Layer6', 'Sigsbee2', 'SigsbeeK', 'SigsbeeK2']


#%%

from .base import ImgBase
from ..common import loadBin
from ..common import RuntimeInterpolate as RI, CoordInterpolate as CI


#%%

class Qikou( ImgBase):

    def __init__(self):
        img = loadBin( 'data/qikou/raw/owe_485_new.bin')
        img /= img.abs().amax()
        img = img.view( 437, 501).T

        from ..build.qikou import compose
        psf = compose()
        psf /= psf.abs().amax()

        rtint = RI( psf, False, 'none', None)
        cint = CI( rtint, [ 23, 24], [ 25, 25])

        super().__init__( 'Qikou', 2, img, psf, ( 50, 50),
                          None, rtint, cint)


class Layer6( ImgBase):

    def __init__(self):
        img = loadBin( 'data/layer6/raw/6layers_2d_demig_lightowe_IMG1124.rsf@')
        img.neg_()
        img /= img.abs().amax()
        img = img.view( 1001, 1601).T

        from ..build.layer6 import compose
        psf = compose()
        psf /= psf.abs().amax()

        rtint = RI( psf, False, 'none', None)
        cint = CI( rtint, [ 79, 50], [ 80, 50])

        super().__init__( 'Layer6', 2, img, psf, ( 50, 50),
                          None, rtint, cint)


#%%

class Sigsbee2( ImgBase):

    def __init__(self):
        img = loadBin( 'data/sigsbee2/raw/sigsbee2a_img_owe_velsm.rsf@')
        img /= img.abs().amax()
        img = img.view( 3201, 1201).T

        from ..build.sigsbee2 import compose
        psf = compose()
        psf /= psf.abs().amax()

        rtint = RI( psf, False, 'none', None)
        cint = CI( rtint, [ 45, 45], [ 25, 25])

        super().__init__( 'Sigsbee2', 2, img, psf, ( 50, 50),
                          None, rtint, cint)


#%%

class SigsbeeK12( ImgBase):

    def __init__(self, version: int):
        assert version in [ 1, 2]
        vstr = ['', '2'][ version - 1]

        imgPath = ['_mu_1201_3201', '']
        imgPath = 'data/sigsbee_kir{}/raw/sigsbee_img_owe{}.rsf@'.format(
            vstr, imgPath[ version - 1])

        imgShape = 3201, 1201

        exec( f'from ..build.sigsbee_kir{vstr} import compose', globals())
        topLeft = 270, 60
        name = f'SigsbeeK{vstr}'

        img = loadBin( imgPath)
        img /= img.abs().amax()
        img = img.view( imgShape).T

        psf = compose()
        psf /= psf.abs().amax()

        rtint = RI( psf, False, 'none', None)
        cint = CI( rtint, topLeft, [ 50, 50])

        super().__init__( name, 2, img, psf, ( 50, 50),
                          None, rtint, cint)


class SigsbeeK( SigsbeeK12):
    def __init__(self):
        super().__init__( 1)


class SigsbeeK2( SigsbeeK12):
    def __init__(self):
        super().__init__( 2)
