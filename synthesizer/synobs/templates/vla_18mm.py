# Setup based on VLA observations from Zamponi et al. (submitted)

import time
Simobserve = True
Clean = True

if Simobserve:
    simobserve(
        project = 'synobs_data',
        skymodel = 'radmc3d_I.fits',
        incenter = '18GHz',
        inwidth = '6.144GHz', 
        setpointings = True,
        integration = '2s',
        totaltime = '32.5min',
        indirection = 'J2000 16h32m22.62 -24d28m32.5',
        refdate = '2021/01/02', 
        hourangle = 'transit',
        obsmode = 'int',
        antennalist = 'vla.a.cfg',
        thermalnoise = 'tsys-atm',
        graphics = 'file',
        overwrite = True,
        verbose = False,
    )

if Clean:
    tclean(
        vis = 'synobs_data/synobs_data.vla.a.noisy.ms',
        imagename = 'synobs_data/clean_I',
        imsize = 400,
        cell = '0.01125arcsec',
        reffreq = '18GHz', 
        specmode = 'mfs',
        gridder = 'standard',
        deconvolver = 'multiscale',
        scales = [1, 8, 20], 
        weighting = 'briggs',
        robust = 0.0,
        niter = 10000,
        threshold = '2e-5Jy',
        pbcor = True, 
        mask = 'synobs_data/synobs_data.vla.a.skymodel', 
        interactive = False,
        verbose = False,
    )
    imregrid(
        'synobs_data/clean_I.image', 
        template = 'synobs_data/synobs_data.vla.a.skymodel', 
        output = 'synobs_data/clean_I.image_modelsize', 
        overwrite = True,
    )
    exportfits(
        'synobs_data/clean_I.image_modelsize', 
        fitsimage = 'synobs_I.fits', 
        dropstokes = True, 
        dropdeg = True, 
        overwrite = True,
    )
