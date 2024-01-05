#%%

__all__ = ['CoordBlockBase']


#%%

import typing as tp


#%%

OI = tp.Optional[ int]


class CoordBlockBase:

    def __init__(self, cbegin: int, cstep: int, cnum: int,
                 fbegin: OI, fend: OI, fstep: OI):

        # coarse
        assert cstep > 0 and cnum > 1
        self.cbegin, self.cstep, self.cnum = cbegin, cstep, cnum
        self.cend = cend = cbegin + cstep * ( cnum - 1) + 1

        # fine begin
        if fbegin is None:
            fbegin = cbegin
        else:
            assert cbegin <= fbegin < cend
        self.fbegin = fbegin

        # fine step
        if fstep is None:
            fstep = 1
        else:
            assert fstep > 0
        self.fstep = fstep

        # fine num & end
        if fend is None:
            fend = cend
        else:
            assert fbegin < fend <= cend
        self.fnum = fnum = ( fend - 1 - fbegin) // fstep + 1
        self.fend = fend = fbegin + fstep * ( fnum - 1) + 1

        # generate
        self.coord = []
        self.poses = []
        self.coeff = []

        for p in range( fbegin, fend, fstep):
            q, r = divmod( p - cbegin, cstep)
            if q == cnum - 1:
                assert r == 0
                q -= 1
                r = cstep
            r /= cstep
            if len( self.coord) == 0 or self.coord[ -1] != q:
                self.coord.append( q)
                self.poses.append( [ p])
                self.coeff.append( [ r])
            else:
                self.poses[ -1].append( p)
                self.coeff[ -1].append( r)


#%%

if __name__ == '__main__':

    import random
    cs = random.randint( 2, 6)
    cn = random.randint( 2, 6)
    fb = random.randint( 0, cs * ( cn - 1))
    fe = random.randint( fb + 1, cs * ( cn - 1) + 1)
    fs = random.randint( 2, 6)

    nn = ''
    n = ''
    cl, fl = '', ''
    ci = fi = 0
    for i in range( cs * ( cn - 1) + 1):
        n += str( i % 10)
        nn += str( i // 10)
        if i % cs == 0:
            cl += str( ci % 10)
            ci += 1
        else:
            cl += ' '
        if fb <= i < fe and ( i - fb) % fs == 0:
            fl += str( fi % 10)
            fi += 1
        else:
            fl += ' '
    print( cs, cn, fb, fe, fs)
    print( nn)
    print( n)
    print( cl)
    print( fl)

    c = CoordBlockBase( 0, cs, cn,
                        fb, fe, fs)
    print( c.coord)
    print( c.poses)
    print( c.remain)
