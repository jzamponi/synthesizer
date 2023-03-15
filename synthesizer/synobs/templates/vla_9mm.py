# Setup based on Zamponi et al. (submitted)

Simobserve = True
Clean = True

if Simobserve:
    simobserve(
        project = 'synobs_data',
        skymodel = 'radmc3d_I.fits',
        incenter = '33GHz',
        inwidth = '7GHz', 
        setpointings = True,
        integration = '2s',
        totaltime = '0.8h',
        indirection = 'J2000 16h32m22.62 -24d28m32.5',
        refdate = '2022/03/07', 
        hourangle = 'transit',
        obsmode = 'int',
        antennalist = 'vla.a.cfg',
        thermalnoise = 'tsys-atm',
        graphics = 'file',
        overwrite = True,
        verbose = False
    )

if Clean:
    tclean(
        vis = 'synobs_data/synobs_data.vla.a.noisy.ms',
        imagename = 'synobs_data/clean_I',
        imsize = 300,
        cell = '0.0097arcsec',
        reffreq = '33GHz', 
        specmode = 'mfs',
        gridder = 'standard',
        deconvolver = 'multiscale',
        scales = [1, 8, 20], 
        weighting = 'briggs',
        robust = 0.5,
        niter = 50,
        threshold = '2.27e-5Jy',
        uvrange = ">215.3klambda", 
        mask = 'synobs_data/synobs_data.vla.a.skymodel', 
        pbcor = True, 
        interactive = False,
        verbose = False
    )

    imregrid(
        'synobs_data/clean_I.image', 
        template='synobs_data/synobs_data.vla.a.skymodel', 
        output='synobs_data/clean_I.image_modelsize', 
        overwrite=True
    )
    exportfits(
        'synobs_data/clean_I.image_modelsize', 
        fitsimage='synobs_I.fits', 
        dropstokes=True, 
        dropdeg=True, 
        overwrite=True
    )

