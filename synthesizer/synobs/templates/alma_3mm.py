# Setup based on Maureira et al. (2020) 

Simobserve = False
Clean = False

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
        indirection = 'J2000 16h32m22.63 -24d28m31.8',
        hourangle = 'transit',
        obsmode = 'int',
        antennalist = 'alma.cycle5.10.cfg',
        thermalnoise = 'tsys-manual',
        graphics = 'both',
        overwrite = True,
        verbose = False
    )

if Clean:
    tclean(
        vis = 'synobs_data/synobs_data.alma.cycle5.10.noisy.ms',
        imagename = 'synobs_data/clean',
        imsize = 300,
        cell = '0.0058arcsec',
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
    'synobs_data/clean.image', 
    template='synobs_data/synobs_data.alma.cycle5.10.skymodel.flat', 
    output='synobs_data/clean.image.modelsize', 
    overwrite=True
)

exportfits(
    'synobs_data/clean.image.modelsize', 
    fitsimage='synobs_I.fits', 
    dropstokes=True, 
    overwrite=True
)

# Smooth to match the 1.3mm resolution. Meant for the spectral index map.
imsmooth(
    'synobs_I.fits', 
    major='0.081855237483972arcsec', 
    minor='0.06690255552529199arcsec', 
    pa='79.39564514160deg', 
    targetres=True, 
    outfile='smoothed'
)

exportfits(
    'smoothed', 
    fitsimage='synobs_I_smoothed1.3mm.fits', 
    dropstokes=True, 
    overwrite=True
)
os.system('rm -r smoothed')
