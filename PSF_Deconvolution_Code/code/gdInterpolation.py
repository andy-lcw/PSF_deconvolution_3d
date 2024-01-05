#%%

from common import *
import core.infer, core.loss, core.optim, core.shape
from core.misc import DictObj, DictObjType, Steps, Epochs
import ds.img

import enum
import math
import pathlib as pl
import sys
from contextlib import ExitStack

import torch
import torch.nn as nn


#%% overall configurations

debug = False
runInBackground = not debug


class Data( metaclass=DictObjType):
    # see `common.AllDsName` for the available list
    dsName = 'Over3d3_quater'  # can be `None`

    device = None           # can be `None`
    numWorkers = 0 if debug else 4

    batchSize = 4096 // 8
    totalIterations = 100

    pretrain = False
    pretrainBatchSize = 128
    pretrainBatches = 250

    """
    'mode' determines the sampling policy.  Only PSFs at the chosen points will
    be solved.  The point coordinate starts from (0,0) (top left). However,
    PSFs don't exist near the boundary as they lack information to interpolate
    from the provided original PSFs.  These boundary points are thus invalid.

    predefined modes:
     - 'whole': all the valid points.  no extra options.  alias of 'area'.
     - 'original': only take the points corresponding to the raw PSFs from the
                   original data.  none of the PSFs are generated from
                   interpolation.  no extra options.  alias of 'grids'.
     - 'points': using a list of points `pointList`.  **not implemented now.**

    regular modes:
     - 'area': an area from `topLeft` to `botRight`, of a specific `size`.  
               `topLeft` is `None` means the area starting from most top-left
               point.  if both `botRight` and `size` is `None`, it ends to most 
               bottom-right point.  if both are specified, they must be
               consistent with the actual bottom-right position.
     - 'grids': a table of a specific grid `size`, from `topLeft` to `botRight`.
                `size` cannot be `None`.  `topLeft` is `None` means most top-
                left.  it is similar for `botRight`.

    modes that will be visualized: all except 'points'.
    modes that will save PSFs, corresponding deconvolution kernels, and 
        deconvolution results: 'original', 'points', and 'grids'.
    """
    mode = 'grids'
    # extra options here
    topLeft = None
    botRight = None
    size = 10, 10, 10
    # size = 5, 5, 5
    # size = 3, 3, 3


class Save( metaclass=DictObjType):
    savePath = 'save-pointwise-cb1/'

    """ data will be saved in separate files if False """
    saveInSingleFile = False

    saveTimes = []
    # saveTimes = [ 1, 2, 4, 8, 16, 32, 50]
    # saveTimes = [ 1, 2, 4, 8, 16, 32, 50, 100, 200]

    """
    'area': always visualize; no PSFs, kernels, and results saving.  
    'original' and 'grids': always save kernels; visualization and results
        saving are optional (controlled by the following options).
    'points': the same as 'grids', while no visualization is applied. 
    """

    # savePSFs = False
    saveRsts = False

    visualize = False
    saveOriImg = False      # only if visualization is enabled

    # save loss curve for every kernel; only valid in non-area modes
    saveIndividualLoss = False
    # loss, raw-l2, raw-rescaled, center value
    statItems = 'ls l2 rs ct'.split()


class Train( metaclass=DictObjType):

    outputShape = None              # None for the PSF shape

    """ False for expressing the deconvolution kernel in spatial domain """
    freqExpression = True

    """
    0: plain L2 loss;
    1: rescaled loss, with only minimum tricks for stabilizing training;
    2: rescaled loss, also avoids exploding center values;
    3: pos-rescaled loss, with center values that are better controlled;
    4: pos-rescaled loss, also uses PSF-based mask;
    5: pos-rescaled loss, also applies mask loss on deconvolution kernels;
    6: pos-rescaled loss, use reciprocal mask instead, basing on (4);
    7: pos-rescaled loss, use both reciprocal mask and mask loss, like (5).

    for now, only losses based on plain L2 are available.  any form of kernel
        mask, or kernel loss, are not implemented.
    """
    lossType = 0            # can be `None`
    override = dict()

    """
    0: 0.8 -1/2->
    1: 0.8 -1->
    2: 0.9 -1/2->
    3: 0.9 -1->
    <0: plain Adam optimizer, no learning rate scheduler
    -1: lr 1e-3
    -2: lr 1e-2
    """
    optimType = 0           # can be `None`


#%%

# input
if Data.dsName is None:
    Data.dsName = input( 'dsName: ')
if Data.device is None:
    Data.device = input( 'device: ')
    if Data.device != 'cpu':
        Data.device = int( Data.device)
if Train.lossType is None:
    Train.lossType = input( 'lossType: ')
if Train.optimType is None:
    Train.optimType = int( input( 'optimType: '))

# Save
Save.saveTimes = sorted( list( Save.saveTimes))
for st in range( 1, len( Save.saveTimes)):
    assert Save.saveTimes[ st-1] < Save.saveTimes[ st]
if len( Save.saveTimes) == 0 or Save.saveTimes[ -1] < Data.totalIterations:
    Save.saveTimes.append( Data.totalIterations)
else:
    assert Save.saveTimes[ -1] == Data.totalIterations
assert Save.saveTimes[ 0] >= 0

saveName = '{ds}-{mode}{freq}-ls_{loss}-opt_{optim}'.format(
    ds=dsShortName( Data.dsName), mode=Data.mode,
    freq=( '' if Train.freqExpression else '-spatial'),
    loss=Train.lossType, optim=Train.optimType,
)
# this function only works on unix-like systems
root = findAndRedirect( Save.savePath, saveName, runInBackground)
Save.saveRoot = root.toDict()
root = root.root

# Data
ids: ds.img.ImgBase = getattr( ds.img, Data.dsName)()
lastDims = tuple( range( - ids.dims, 0))

if torch.device( Data.device).type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device( Data.device)

if Data.numWorkers > 0:
    import torch.multiprocessing as tmp
    tmp.set_start_method( 'fork')

# for Data.mode
Data.modeAlias = Data.mode

if Data.mode in ['whole', 'original']:
    Data.begin = ids.cint.lt
    Data.end = ids.cint.br1
    if Data.mode == 'whole':
        Data.mode = 'area'
        Data.step = ( 1,) * ids.dims
    else:
        Data.mode = 'grids'
        Data.step = ids.cint.gs
    # Data.pointList = None

elif Data.mode == 'points':
    raise NotImplementedError
    # Data.begin = Data.end = Data.step = None

elif Data.mode in ['area', 'grids']:
    if Data.topLeft is None:
        Data.begin = ids.cint.lt
    else:
        Data.begin = tuple( Data.topLeft)
        assert len( Data.begin) == ids.dims and \
               all( lt <= b < br1 for lt, b, br1 in
                    zip( ids.cint.lt, Data.begin, ids.cint.br1))

    if Data.size is not None:
        assert len( Data.size) == ids.dims and all( s > 0 for s in Data.size)

    if Data.botRight is None and ( Data.mode == 'grids' or Data.size is None):
        Data.end = ids.cint.br1
    else:
        if Data.mode == 'area' and Data.size is not None:
            Data.end = tuple( b + s for b, s in zip( Data.begin, Data.size))
            if Data.botRight is not None:
                assert Data.end == tuple( Data.botRight)
        else:
            Data.end = tuple( Data.botRight)
        assert len( Data.end) == ids.dims and \
               all( e <= br1 for e, br1 in zip( Data.end, ids.cint.br1))

    Data.step = ( 1,) * ids.dims
    if Data.mode == 'grids':
        assert Data.size is not None
        Data.step = tuple( Data.size)
    # Data.pointList = None

else:
    raise KeyError( Data.mode)

# Save
if Data.mode == 'points':
    raise NotImplementedError
    # Save.visualize = Save.saveOriImg = False
elif Data.mode == 'area':
    # Save.savePSFs = False
    Save.saveRsts = Save.saveIndividualLoss = False
    Save.visualize = True
elif not Save.visualize:        # 'grids'
    Save.saveOriImg = False

# Train & Hyper
Train.invShape = ids.psfShape
if Train.outputShape is not None:
    Train.invShape = tuple( Train.outputShape)

try:
    Train.realLossType = Train.lossType = int( Train.lossType)
    Train.realOverride = {}
except ValueError:
    Train.realLossType, Train.realOverride = Train.override[ Train.lossType]
Hyper = buildHyper( Train.realLossType, ids.psfShape, Train.invShape)
Hyper.update( ** Train.realOverride)
assert Hyper.loss == 'plain'
assert Hyper.invL1Rate is None and Hyper.invL2Rate is None

him = HyperInvMask( Hyper, Train.invShape, Data.device)
hilm = HyperInvLossMask( Hyper, Train.invShape, Data.device)
assert him is None and hilm is None


class Train2( metaclass=DictObjType):
    palpha = 1
    plr = 0.001
    pbeta = pbeta2 = 0.8, 0.9
    porder = 1

    if Train.optimType < 0:
        alpha = 1
        lr = [ ( 0.001, 0.001), ( 0.01, 0.01)][ - Train.optimType - 1]
        beta = 0.9, 0.9
        beta2 = 0.99, 0.99
    else:
        alpha = [ 1/2, 1][ Train.optimType % 2]
        lr = 0.001, 0.01
        beta = [ 0.8, 0.9][ Train.optimType // 2], 0.95
        beta2 = beta[ 0], 0.99


poptimFunc = lambda p: core.optim.AdamScheduler(
    torch.optim.Adam( p), Data.pretrainBatches,
    core.optim.Constant( Train2.plr),
    core.optim.CosPowFunc( * Train2.pbeta, Train2.palpha),
    core.optim.CosPowFunc( * Train2.pbeta2, Train2.palpha))
optimFunc = lambda p: core.optim.AdamScheduler(
    torch.optim.Adam( p), Data.totalIterations,
    core.optim.CosPowFunc( * Train2.lr, Train2.alpha),
    core.optim.CosPowFunc( * Train2.beta, Train2.alpha),
    core.optim.CosPowFunc( * Train2.beta2, Train2.alpha))
Train.update( **Train2)

# save
save = DictObj(
    Data=Data.toDict(), Save=Save.toDict(),
    Train=Train.toDict(), Hyper=Hyper.toDict(),
)


#%%

print( 'loading...')

# coordinate-based loader       base pos coeff img/None
coordRange = None
# validIndices = None
if Data.mode != 'points':
    coordRange = torch.as_tensor( [ Data.begin, Data.end, Data.step],
                                  dtype=torch.long)
    icbloader = ds.img.ImgCoordBlock( ids, coordRange=coordRange)
    icbloader = ds.img.ImgCoordBlockDataset( icbloader,
                                             Save.visualize, Train.invShape)
else:
    raise NotImplementedError
    # validIndices = ids.validIndices( pointList=Data.pointList)
    # if len( validIndices[ 0]) == 0:
    #     raise ValueError( 'no valid data')
    # iloader = ds.img.ImgDataset( ids, * validIndices, Train.invShape)

icbloader = icbloader.getLoader( Data.batchSize,
                                 numWorkers=Data.numWorkers, pinMemory=True)

# save
TC, SI, ST = TensorCollector, Save.saveInSingleFile, Save.saveTimes

save.poses = save.invs = save.rsts = 'not applicable'
# save.psfs = 'not applicable'
if Data.mode != 'area':
    save.poses = TC.single( SI, root, 'poses')
    # if Save.savePSFs:
    #     save.psfs = TC.single( SI, root, 'psfs')
    save.invs = TC.list( SI, root, 'invs', ST)
    if Save.saveRsts:
        save.rsts = TC.list( SI, root, 'rsts', ST)
    if Save.saveIndividualLoss:
        save.losses = TC.list( SI, root, 'losses', Save.statItems)

oriImg = None
save.oriImg = save.res = 'not applicable'
# save.resn = 'not applicable'
if Data.mode != 'points':
    oriImg = ids.img[ tuple( slice( * ir) for ir in coordRange.T.tolist())]
    if Save.visualize:
        if Save.saveOriImg:
            save.oriImg = TC.wrap( oriImg, SI, root, 'oriImg')
        build = lambda: [ torch.zeros_like( oriImg) for _ in Save.saveTimes]

        save.res = DictObj()
        save.res.vis = 'not applicable'
        # save.res.abs = save.res.sqr = 'not applicable'
        if Hyper.loss == 'plain':
            save.res.vis = tuple( build())
            # save.res.abs = tuple( build())
            # save.res.sqr = tuple( build())

        # save.resn = DictObj()
        # save.resn.vis = tuple( build())
        # save.resn.abs = tuple( build())
        # save.resn.sqr = tuple( build())

# param
if Data.pretrain:
    initParam = pretrain(
        ids, Train.freqExpression, Train.invShape, poptimFunc, Hyper, him, hilm,
        Data.device, Data.pretrainBatchSize, Data.pretrainBatches,
        order=Train.porder)
else:
    initParam = core.infer.invIdentity( ids.psfShape, Train.invShape)
save.init = initParam.clone()

if Train.freqExpression:
    initParam = torch.view_as_real(
        torch.fft.rfftn( initParam, dim=lastDims, norm='forward'))
scaleFactor = initParam.std( False) * math.sqrt( initParam.numel())
initParam /= scaleFactor


#%%

print( 'warn: this timer only covers the training time. '
       'other time (e.g., for visualization) is not included. '
       'it will be shorter than the wall clock time.')
timer = Timer( Data.device)


class LossObj:
    def __init__(self):
        self.loss = tuple( [] for _ in Save.statItems)

    def update(self, _losses):
        for _i, _ls in enumerate( _losses):
            self.loss[ _i].append( _ls.cpu())

    def appendSelf(self):
        for _i, _ls in enumerate( self.loss):
            # b, it
            _ls = torch.stack( _ls, dim=1)
            save.losses[ _i].append( _ls)


class Infer( enum.IntFlag):
    inv = 0b000     # get deconv kernel
    rst = 0b001     # get deconv result using real kernel
    upd = 0b011     # update statistic data
    opt = 0b111     # step optimizer


def inference( _oinvs, _bases, _coeff,
               _statStep, _optim, _infer: Infer):
    with ExitStack() as _exitStack:
        def _ap( _sls):
            return _sls.view_as( _coeff).mul( _coeff).sum( 0)

        if Infer.opt in _infer:
            _optim.zero_grad()
            _exitStack.enter_context( timer)

        _invs = _oinvs * scaleFactor
        if Train.freqExpression:
            _invs = torch.fft.irfftn( torch.view_as_complex( _invs),
                                      Train.invShape, norm='forward')

        _rsts = None
        if Infer.rst in _infer:
            # coeff: (nb, b); bases: (nb, ...); rsts: (nb*b, ...)
            _rsts = core.infer.crossFullConv( _bases, _invs).flatten( 0, 1)

            if Infer.upd in _infer:
                _lss = _ap( core.loss.lossFunc( _rsts, None, override=Hyper))

                if Infer.opt in _infer:
                    _lss.sum().backward()
                    _optim.step()
                    _exitStack.close()
                    _exitStack.enter_context( torch.inference_mode())

                _l2s = _ap( core.loss.rawL2LossFunc( _rsts))
                _rss = _ap( core.loss.rawRescaledLossFunc( _rsts))
                _cts = _ap( core.shape.CenterCache.getCenter( _rsts))
                _losses = tuple( _l.detach() for _l in
                                 ( _lss, _l2s, _rss, _cts))

                for _it, _dt in zip( Save.statItems, _losses):
                    _statStep.update( _it, _dt)

        return _invs, _rsts, _losses


def writeInSave( _poses, _imgs, _coeff, _invs, _rsts, _posit):
    _invs = _invs.detach()
    _rsts = _rsts.detach()

    with torch.inference_mode():
        if Data.mode != 'area':
            save.invs[ _posit].append( _invs.cpu())
            if Save.saveRsts:
                _rsts = _rsts.view( _coeff.shape + _rsts.shape[ 1:])
                _coeff = _coeff.view( _coeff.shape + ( 1,) * ( _rsts.ndim - 2))
                # (nb, b, 1...), (nb, b, ...) -> (b, ...)
                _rsts = _rsts.mul( _coeff).sum( 0)
                save.rsts[ _posit].append( _rsts.cpu())
        if Data.mode == 'points' or not Save.visualize:
            return

        _values = _imgs.mul( _invs).sum( dim=lastDims)
        save.res.vis[ _posit][ _poses] = _values.cpu()


#%%

stepList = [ Steps( * Save.statItems) for _ in range( Data.totalIterations + 1)]

totalPSFs = 0
# if validIndices is not None:
#     allPSFs = len( validIndices[ 0])
# else:
allPSFs = oriImg.numel()

for bases, poses, coeff, imgs in icbloader:
    totalPSFs += len( poses)
    print( f'{totalPSFs}/{allPSFs}: ', end='')
    sys.stdout.flush()

    lossObj = None
    if Data.mode != 'area':
        save.poses.append( poses)
        # if Save.savePSFs:
        #     save.psfs.append( psfs)
        if Save.saveIndividualLoss:
            lossObj = LossObj()

    if Data.mode != 'points':
        poses = poses.sub( coordRange[ 0]) \
            .div_( coordRange[ 2], rounding_mode='trunc').T.unbind()
        if Save.visualize:
            imgs = imgs.to( Data.device, non_blocking=True)

    bases = bases.to( Data.device, non_blocking=True)
    coeff = coeff.T.to( Data.device, non_blocking=True)

    oinvs = nn.Parameter( initParam.to( Data.device)
                          .repeat( coeff.shape[ 1], * ( 1,) * initParam.ndim))
    optim = optimFunc( [ oinvs])

    dots = Dots( Data.totalIterations)
    for iteration in range( Data.totalIterations + 1):
        dots.step( iteration)
        if iteration < Data.totalIterations:
            ret = inference( oinvs, bases, coeff,
                             stepList[ iteration], optim, Infer.opt)
        else:
            with torch.inference_mode():
                ret = inference( oinvs, bases, coeff,
                                 stepList[ iteration], None, Infer.upd)
        invs, rsts, losses = ret

        if Data.mode != 'area' and Save.saveIndividualLoss:
            lossObj.update( losses)

        if iteration in Save.saveTimes:
            position = Save.saveTimes.index( iteration)
            writeInSave( poses, imgs, coeff, invs, rsts, position)
        del ret, invs, rsts, losses

    if Data.mode != 'area' and Save.saveIndividualLoss:
        lossObj.appendSelf()

    del bases, poses, coeff, imgs, lossObj, oinvs, optim
    print( '', timer.fullStr( totalPSFs, allPSFs))
print( 'finished', timer)


#%%

print( 'postprocessing...')

epochs = Epochs( * Save.statItems)
for st in stepList:
    epochs.update( st)
save.stat = epochs.stateDict()

if Data.mode != 'area':
    save.poses = save.poses.bake( -1)
    # if Save.savePSFs:
    #     save.psfs = save.psfs.bake( -1)
    save.invs = tuple( s.bake( -1) for s in save.invs)
    if Save.saveRsts:
        save.rsts = tuple( s.bake( -1) for s in save.rsts)

    if Save.saveIndividualLoss:
        save.losses = { si: ls.bake( -1) for si, ls in
                        zip( Save.statItems, save.losses)}

if Data.mode != 'points' and Save.visualize:
    save.res.vis = tuple(
        TC.wrapList( save.res.vis, SI, root, f'res.vis', ST))
    # if Hyper.loss == 'plain':
    #     for item in 'vis abs sqr'.split():
    #         save.res[ item] = tuple(
    #             TC.wrapList( save.res[ item], SI, root, f'res.{item}', ST))
    save.res = save.res.toDict()

    # for item in 'vis abs sqr'.split():
    #     save.resn[ item] = tuple(
    #         TC.wrapList( save.resn[ item], SI, root, f'resn.{item}', ST))
    # save.resn = save.resn.toDict()

torch.save( save.toDict(), str( pl.Path( root) / 'save.pth'))

print( 'done')
