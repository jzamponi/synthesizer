# Setup based on Maureira et al. (2020) and used in Zamponi et al. (2020)

Simobserve = True
Clean = True

if Simobserve:
	simobserve(
		project = 'synobs_data',
		skymodel = 'radmc3d_I.fits',
		inbright = '',
		incell = '',
		mapsize = '',
		incenter = '223GHz',
		inwidth = '0.12GHz',
		setpointings = True,
		integration = '2s',
		totaltime = '1.25h',
        refdate = '2017/08/21', 
		indirection = 'J2000 16h32m22.63 -24d28m31.8',
		hourangle = 'transit',
		obsmode = 'int',
		antennalist = 'alma.cycle4.7.cfg',
		thermalnoise = 'tsys-manual',
		graphics = 'file',
		overwrite = True,
		verbose = True
	)

if Clean:
	tclean(
		vis = 'synobs_data/synobs_data.alma.cycle4.7.noisy.ms',
		imagename = 'synobs_data/clean_I',
		imsize = 400,
		cell = '0.008arcsec',
        reffreq = '223GHz', 
		specmode = 'mfs',
		gridder = 'standard',
		deconvolver = 'multiscale',
        scales = [1, 8, 24], 
		weighting = 'briggs',
		uvrange = '120~2670klambda',
		robust = 0.0,
		niter = 10000,
		threshold = '2.5e-4Jy',
        mask = 'synobs_data/synobs_data.alma.cycle4.7.skymodel', 
		interactive = False,
		verbose = True
	)

    imregrid(
        'synobs_data/clean_I.image', 
        template = 'synobs_data/synobs_data.alma.cycle4.7.skymodel',
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
