# Setup based on Maureira et al. (2020) and used in Zamponi et al. (2021)

Simobserve = True
Clean = True

if Simobserve:
    simobserve(
        project = 'synobs_data',
        skymodel = 'radmc3d_I.fits',
        inbright = '',
        incell = '',
        mapsize = '',
        incenter = '99988140037.24495Hz',
        inwidth = '2GHz',
        setpointings = True,
        integration = '2s',
        totaltime = '1.25h',
        refdate = '2017/10/08',
        indirection = 'J2000 16h32m22.63 -24d28m31.8',
        hourangle = 'transit',
        obsmode = 'int',
        antennalist = 'alma.cycle5.10.cfg',
        thermalnoise = 'tsys-manual',
        graphics = 'file',
        overwrite = True,
        verbose = False
    )

if Clean:
    tclean(
        vis = 'synobs_data/synobs_data.alma.cycle5.10.noisy.ms',
        imagename = 'synobs_data/clean_I',
        imsize = 400,
        cell = '0.0058arcsec',
        reffreq = '99988140037.24495Hz',
        specmode = 'mfs',
        gridder = 'standard',
        deconvolver = 'multiscale',
        scales = [1, 8, 24], 
        weighting = 'briggs',
        robust = 0.5,
        niter = 10000,
        threshold = '5e-5Jy',
        mask = 'synobs_data/synobs_data.alma.cycle5.10.skymodel', 
        interactive = False,
        verbose = False
    )
    
    imregrid(
        'synobs_data/clean_I.image', 
        template = 'synobs_data/synobs_data.alma.cycle5.10.skymodel', 
        output = 'synobs_data/clean_I.image_modelsize', 
        overwrite = True
    )

    exportfits(
        'synobs_data/clean_I.image_modelsize', 
        fitsimage = 'synobs_I.fits', 
        dropstokes = True, 
        overwrite = True
    )
