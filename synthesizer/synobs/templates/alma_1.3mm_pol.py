# Setup based on Sadavoy et al. (2018b)

import random

Simobserve = True
Clean = True

if Simobserve:
    simobserve(
        project = 'synobs_data',
        skymodel = 'radmc3d_I.fits',
        incenter = '233GHz',
        inwidth = '7.5GHz', 
        setpointings = True,
        integration = '2s',
        totaltime = '7.2min',
        indirection = 'J2000 16h32m22.63 -24d28m31.8',
        refdate = '2017/05/20', 
        hourangle = 'transit',
        obsmode = 'int',
        antennalist = 'alma.cycle3.5.cfg',
        thermalnoise = 'tsys-atm',
        seed = int(random.random() * 100),
        graphics = 'file',
        overwrite = True,
        verbose = True
    )

if Clean:
    tclean(
        vis = 'synobs_data/synobs_data.alma.cycle3.5.noisy.ms',
        imagename = 'synobs_data/clean_I',
        imsize = 400,
        cell = '0.029arcsec',
        reffreq = '233GHz', 
        specmode = 'mfs',
        gridder = 'standard',
        deconvolver = 'multiscale',
        scales = [0, 8, 20, 40], 
        weighting = 'briggs',
        robust = 0.5,
        uvtaper = '0.1arcsec',
        mask = 'synobs_data/synobs_data.alma.cycle3.5.skymodel',
        niter = 10000,
        threshold = '280uJy',
        pbcor = True, 
        interactive = False,
        verbose = True
    )

    imregrid(
        'synobs_data/clean_I.image', 
        template = 'synobs_data/synobs_data.alma.cycle3.5.skymodel', 
        output = 'synobs_data/clean_I.image_modelsize', 
        overwrite = True
    )

    exportfits(
        'synobs_data/clean_I.image_modelsize', 
        fitsimage = 'synobs_I.fits', 
        dropstokes = True, 
        dropdeg = True, 
        overwrite = True
    )
