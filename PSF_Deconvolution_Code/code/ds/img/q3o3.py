#%%

__all__ = ['Qikou3d', 'Qikou3d2', 'Qikou3d3', 'Qikou3d4', 'Qikou3dK',
           'Over3d', 'Over3d2',
           'Over3d3_raw', 'Over3d3_half', 'Over3d3_quater',
           'Over3dK', 'Over3dK2']


#%%

from .base import ImgBase
from ..common import loadBin
from ..common import RuntimeInterpolate as RI, CoordInterpolate as CI

from scipy.io import loadmat

import torch


#%%

class Qikou3d( ImgBase):

    def __init__(self):
        img = loadmat( 'data/qikou3d/raw/loc_mig.mat')['loc_mig']
        img = torch.as_tensor( img)
        img /= img.abs().amax()
        img = img.float()

        from ..build.qikou3d import compose
        psf = compose()
        psf /= psf.abs().amax()

        rtint = RI( psf, True, 'none', None)
        cint = CI( rtint, [ 12, 10, 10], [ 20, 20, 20])

        super().__init__( 'Qikou3d', 3, img, psf, ( 21, 21, 21),
                          None, rtint, cint)


#%%

class Qikou3d234( ImgBase):

    def __init__(self, version: int):
        assert version in [ 2, 3, 4]

        imgPath = 'img  owe_706_601_437  owe'.split()
        imgPath = 'data/qikou3d{}/raw/qikou_{}.rsf@'.format(
            version, imgPath[ version - 2])

        imgShape = [ ( 601, 437, 701), ( 601, 437, 706), ( 437, 437, 801)]
        imgShape = imgShape[ version - 2]

        exec( f'from ..build.qikou3d{version} import compose', globals())
        topLeft = [ 5, 10, 10][ version - 2], 10, 10
        name = f'Qikou3d{version}'

        img = loadBin( imgPath)
        img.neg_()
        img /= img.abs().amax()
        img = img.view( imgShape).permute( 2, 1, 0)

        psf = compose()
        psf /= psf.abs().amax()

        rtint = RI( psf, True, 'none', None)
        cint = CI( rtint, topLeft, [ 20, 20, 20])

        super().__init__( name, 3, img, psf, ( 21, 21, 21),
                          None, rtint, cint)


class Qikou3d2( Qikou3d234):
    def __init__(self):
        super().__init__( 2)


class Qikou3d3( Qikou3d234):
    def __init__(self):
        super().__init__( 3)


class Qikou3d4( Qikou3d234):
    def __init__(self):
        super().__init__( 4)


#%%

class Qikou3dK( ImgBase):

    def __init__(self):
        img = loadBin( 'data/qikou3d_kir/raw/Qikoued_img_owe_701_437_601.rsf@')
        img.neg_()
        img /= img.abs().amax()
        img = img.view( 601, 437, 701).permute( 2, 1, 0)

        from ..build.qikou3d_kir import compose
        psf = compose()
        psf /= psf.abs().amax()

        rtint = RI( psf, True, 'none', None)
        cint = CI( rtint, [ 45, 10, 10], [ 20, 20, 20])

        super().__init__( 'Qikou3dK', 3, img, psf, ( 21, 21, 21),
                          None, rtint, cint)


#%%

class Over3d12( ImgBase):

    def __init__(self, version: int):
        assert version in [ 1, 2]
        vstr = ['', '2'][ version - 1]

        exec( f'from ..build.over3d{vstr} import compose', globals())
        topLeft = [ 40, 10][ version - 1], 10, 10
        psfShape = grids = [ 80, 20][ version - 1], 20, 20
        name = f'Over3d{vstr}'

        img = loadBin( 'data/over3d/raw/over_3d_owe_image_601_281_281.rsf@')
        img = img.view( 281, 281, 601).permute( 2, 1, 0)
        if version == 2:
            img = img[ ::4]
        img /= img.abs().amax()

        psf = compose()
        psf /= psf.abs().amax()

        rtint = RI( psf, True, 'none', None)
        cint = CI( rtint, topLeft, grids)

        super().__init__( name, 3, img, psf, psfShape,
                          None, rtint, cint)


class Over3d( Over3d12):
    def __init__(self):
        super().__init__( 1)


class Over3d2( Over3d12):
    def __init__(self):
        super().__init__( 2)


#%%

class Over3d3s( ImgBase):

    def __init__(self, name: str):
        assert name in 'raw half quater'.split()

        from ..build.over3d3 import compose
        topLeft = 30, 10, 10
        psfShape = grids = 20, 20, 20

        img = loadBin( f'data/over3d3/raw/over3d_mig_{name}_121_281_281.bin')
        img.neg_()
        img = img.view( 281, 281, 216).permute( 2, 1, 0)
        img /= img.abs().amax()

        psf = compose( name)
        psf /= psf.abs().amax()

        rtint = RI( psf, True, 'none', None)
        cint = CI( rtint, topLeft, grids)

        super().__init__( f'Over3d3_{name}', 3, img, psf, psfShape,
                          None, rtint, cint)


class Over3d3_raw( Over3d3s):
    def __init__(self):
        super().__init__( 'raw')


class Over3d3_half( Over3d3s):
    def __init__(self):
        super().__init__( 'half')


class Over3d3_quater( Over3d3s):
    def __init__(self):
        super().__init__( 'quater')


#%%

class Over3dK12( ImgBase):

    def __init__(self, version: int):
        assert version in [ 1, 2]
        vstr = ['', '2'][ version - 1]

        imgPath = ['3d_img_owe_151_281_281', '_new_owe_img']
        imgPath = 'data/over3d_kir{}/raw/over{}.rsf@'.format(
            vstr, imgPath[ version - 1])

        imgShape = 281, 281, 151

        exec( f'from ..build.over3d_kir{vstr} import compose', globals())
        topLeft = [ 30, 10][ version - 1], 10, 10
        name = f'Over3dK{vstr}'

        img = loadBin( imgPath)
        img /= img.abs().amax()
        img = img.view( imgShape).permute( 2, 1, 0)

        psf = compose()
        psf /= psf.abs().amax()

        rtint = RI( psf, True, 'none', None)
        cint = CI( rtint, topLeft, [ 20, 20, 20])

        super().__init__( name, 3, img, psf, ( 20, 20, 20),
                          None, rtint, cint)


class Over3dK( Over3dK12):
    def __init__(self):
        super().__init__( 1)


class Over3dK2( Over3dK12):
    def __init__(self):
        super().__init__( 2)
