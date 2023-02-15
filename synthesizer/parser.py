#!/usr/bin/env python3
"""
    Main command-line argument parser.

    Example:

    $ synthesizer --sphfile snap_001.dat --ncells 100 --bbox 50
        --show-grid-2d --show-grid-3d --raytrace --lam 3000 --amax 10 
        --polarization --opacity --material s --amin 0.1 --amax 10 --na 100 
        --show-rt --synobs --show-synobs

    For details, run:
    $ synthesizer --help

    Requisites:
        Software:   python3, CASA, RADMC3D, 
                    Mayavi (optional), ParaView (optional)

        Modules:    python3-aplpy, python3-scipy, python3-numpy, python3-h5py
                    python3-matplotlib, python3-astropy, python3-mayavi,

"""


import sys
import argparse

from synthesizer import utils
from synthesizer.pipeline import Pipeline


def synthesizer():

    # Initialize the argument parser
    parser = argparse.ArgumentParser(prog='Synthesizer',
        description="")

    # Define command line options
    parser.add_argument('-g', '--grid', action='store_true', default=False, 
        help='Create an input grid for the radiative transfer')

    exc_grid = parser.add_mutually_exclusive_group() 
    exc_grid.add_argument('--model', action='store', default=None, 
        choices=['constant', 'plaw', 'ppdisk', 'pcore', 'gidisk', 'spiral-disk',
            'filament'], 
        help='Keyword for a predefined density model.')

    exc_grid.add_argument('--sphfile', action='store', default=None, 
        help='Name of the input SPH file (snapshot from a particle-based code')

    exc_grid.add_argument('--amrfile', action='store', default=None, 
        help='Name of the input AMR grid file (snapshot from a grid-based code)')

    parser.add_argument('--source', action='store', default='sphng-bin', 
        choices=['sphng-bin', 'gizmo', 'gadget', 'arepo', 'gandalf', 'gradsph',
            'athena', 'zeustw', 'flash', 'enzo', 'ramses'], 
        help='Name of the code used to generate the inputfile.')

    parser.add_argument('--ncells', action='store', type=int, default=100,
        help='Number of cells in every direction')

    exc_trim = parser.add_mutually_exclusive_group() 
    exc_trim.add_argument('--bbox', action='store', type=float, default=None, 
        help='Size of the half-lenght of a bounding box in au.')

    exc_trim.add_argument('--rout', action='store', type=float, default=None, 
        help='Size of the outer radial boundary in au (i.e., zoom in)')

    parser.add_argument('--g2d', action='store', type=float, default=100, 
        help='Set the gas-to-dust mass ratio.')

    parser.add_argument('--vector-field', action='store', type=str, default=None, 
        choices=['x', 'y', 'z', 'toroidal', 'radial', 'hourglass', 
        'helicoidal', 'dipole', 'quadrupole'], 
        help='Create a vectory field for alignment of elongated grains.')

    parser.add_argument('--temperature', action='store_true', default=False,
        help='Write the dust temperature from the model.')

    parser.add_argument('--show-grid-2d', action='store_true', default=False,
        help='Plot the midplane of the newly created grid')

    parser.add_argument('--show-grid-3d', action='store_true', default=False,
        help='Render the new cartesian grid in 3D')

    parser.add_argument('--vtk', action='store_true', default=False,
        help='Call RADCM3D to create a VTK file of the newly created grid')

    parser.add_argument('--render', action='store_true', default=False,
        help='Visualize the VTK file using ParaView')

    parser.add_argument('-op', '--opacity', action='store_true', default=False,
        help='Call dustmixer to generate a dust opacity table')

    parser.add_argument('--material', action='store', default='s',
        help='Dust optical constants. Can be a predefined key, a path or a url')

    parser.add_argument('--amin', action='store', type=float, default=0.1,
        help='Minimum value for the grain size distribution')

    parser.add_argument('--amax', action='store', type=float, default=10,
        help='Maximum value for the grain size distribution')

    parser.add_argument('--na', action='store', type=int, default=100,
        help='Number of size bins for the logarithmic grain size distribution')

    parser.add_argument('--q', action='store', type=float, default=-3.5,
        help='Slope of the grain size distribution in logspace')

    parser.add_argument('--nang', action='store', type=int, default=3,
        help='Number of scattering angles used to sample the dust efficiencies')

    parser.add_argument('--show-opacity', action='store_true', default=False,
        help='Plot the resulting dust opacities.')

    parser.add_argument('--show-nk', action='store_true', default=False,
        help='Plot the input dust optical constants.')

    parser.add_argument('-mc', '--monte-carlo', action='store_true', default=False,
        help='Call RADMC3D to raytrace the new grid and plot an image')

    parser.add_argument('--nphot', action='store', type=float, default=1e5,
        help='Set the number of photons for scattering and thermal Monte Carlo')

    parser.add_argument('--nthreads', action='store', default=1, 
        help='Number of threads used for the Monte-Carlo runs')

    parser.add_argument('-rt', '--raytrace', action='store_true', default=False,
        help='Call RADMC3D to raytrace the new grid and plot an image')

    parser.add_argument('--lam', action='store', type=float, default=1300,
        help='Wavelength used to generate an image in units of micron')

    parser.add_argument('--lmin', action='store', type=float, default=0.1,
        help='Lower end of the wavelength grid in microns.')

    parser.add_argument('--lmax', action='store', type=float, default=1e5,
        help='Upper end of the wavelength grid in microns.')

    parser.add_argument('--nlam', action='store', type=int, default=200,
        help='Number of wavelengths to build a logarithmically spaced grid.')

    parser.add_argument('--npix', action='store', type=int, default=300,
        help='Number of pixels per side of new image')

    parser.add_argument('--incl', action='store', type=float, default=0,
        help='Inclination angle of the grid in degrees')

    parser.add_argument('--sizeau', action='store', type=int, default=100,
        help='Physical size of the image in AU')

    parser.add_argument('--distance', action='store', type=float, default=141,
        help='Physical distance in pc, used to convert fluxes into Jy/pixel.')

    parser.add_argument('--star', action='store', default=None, nargs=6, 
        metavar=('x', 'y', 'z', 'Rstar', 'Mstar', 'Teff'), 
        help='6 parameters used to define a radiating star (values should be given in cgs)')
        
    parser.add_argument('--tau', action='store_true', default=False,  
        help='Generate a 2D optical depth map.')

    parser.add_argument('--tau-surf', default=None, type=float, metavar=('tau'), 
        help='Generate a 3D surface at optical depth = %(metavar)s.')

    parser.add_argument('--show-tau-surf', action='store_true', default=False, 
        help='Render the 3D the optical depth = tau surface.')

    parser.add_argument('--polarization', action='store_true', default=False,
        help='Enable polarized RT and full scattering matrix opacity tables')

    parser.add_argument('--alignment', action='store_true', default=False,
        help='Enable polarized RT of thermal emission from aligned grains.')

    parser.add_argument('--noscat', action='store_true', default=False,
        help='Turn off the addition of scattered flux to the thermal flux')

    parser.add_argument('--sublimation', action='store', type=float, default=0,
        help='Percentage of refractory carbon that evaporates from the dust')

    parser.add_argument('--soot-line', action='store', type=float, default=300,
        help='Temperature at which carbon is supposed to sublimate')

    parser.add_argument('--dust-growth', action='store_true', default=False,  
        help='Enable dust growth within the soot-line')

    parser.add_argument('--show-rt', action='store_true', default=False,
        help='Plot the intensity map generated by ray-tracing')

    parser.add_argument('--radmc3d', action='store', type=str, default='',
        help='Additional commands to be passed to RADMC3D.', nargs="*")

    parser.add_argument('--synobs', action='store_true', default=False,
        help='Call CASA to run a synthetic observation from the new image')

    parser.add_argument('--dont-observe', action='store_true', default=False,
        help="Skips CASA's Simobserve task.")

    parser.add_argument('--dont-clean', action='store_true', default=False,
        help="Skips CASA's tclean task to clean and image simobserve's output.")

    parser.add_argument('--dont-export', action='store_true', default=False,
        help="Don't export tclean's output as a fits image.")

    parser.add_argument('--obs-time', action='store', type=float, default=None,
        help="Set the observing time in hours.")

    parser.add_argument('--resolution', action='store', type=float, default=None,
        help="Set a desired angular resolution in arcseconds.")

    parser.add_argument('--obsmode', action='store', type=str, default='int',
        choices=['int', 'sd'], 
        help="Wheter to observe with an radio interferometer or a single-dish.")
    
    parser.add_argument('--telescope', action='store', type=str, default='alma',
        choices=['alma', 'aca', 'vla'], 
        help="Radio telescope to use. Default: ALMA")

    parser.add_argument('--script', action='store', default=None,
        help='Name, path or url to a CASA script for the synthetic observation.')

    parser.add_argument('--show-synobs', action='store_true', default=False,
        help='Plot the ALMA/JVLA synthetic image generated by CASA')

    parser.add_argument('--overwrite', action='store_true', default=False,
        help='Overwrite radmc3d input files.')

    parser.add_argument('--quiet', action='store_true', default=False,
        help='Disable verbosity.')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0')



    # Store the command-line given arguments
    cli = parser.parse_args()

    # Initialize the pipeline
    pipeline = Pipeline(lam=cli.lam, amax=cli.amax, nphot=cli.nphot, 
        lmin=cli.lmin, lmax=cli.lmax, nlam=cli.nlam, nthreads=cli.nthreads, 
        csubl=cli.sublimation, sootline=cli.soot_line, dgrowth=cli.dust_growth,
        polarization=cli.polarization, alignment=cli.alignment, star=cli.star, 
        material=cli.material, overwrite=cli.overwrite, verbose=not cli.quiet)

    # Generate the input grid for RADMC3D
    if cli.grid:
        pipeline.create_grid(sphfile=cli.sphfile, amrfile=cli.amrfile, 
            source=cli.source, model=cli.model, ncells=cli.ncells, g2d=cli.g2d, 
            bbox=cli.bbox, rout=cli.rout, temperature=cli.temperature, 
            render=cli.render, vtk=cli.vtk, show_2d=cli.show_grid_2d, 
            show_3d=cli.show_grid_3d, vector_field=cli.vector_field)

    # Generate the dust opacity tables
    if cli.opacity:
        pipeline.dust_opacity(cli.amin, cli.amax, cli.na, cli.q, cli.nang,
            show_nk=cli.show_nk, show_opac=cli.show_opacity)

    # Run a thermal Monte-Carlo
    if cli.monte_carlo:
        pipeline.monte_carlo(nphot=cli.nphot, radmc3d_cmds=cli.radmc3d)

    # Run a ray-tracing on the new grid and generate an image
    if cli.raytrace:
        pipeline.raytrace(incl=cli.incl, npix=cli.npix, distance=cli.distance,
            sizeau=cli.sizeau, show=cli.show_rt, noscat=cli.noscat, tau=cli.tau,
            tau_surf=cli.tau_surf, show_tau_surf=cli.show_tau_surf, 
            radmc3d_cmds=cli.radmc3d)

    # Run a synthetic observation of the new image by calling CASA
    if cli.synobs:
        pipeline.synthetic_observation(show=cli.show_synobs, script=cli.script, 
            simobserve=not cli.dont_observe, clean=not cli.dont_clean, 
            exportfits=not cli.dont_export, obstime=cli.obs_time,  
            resolution=cli.resolution, obsmode=cli.obsmode, 
            telescope=cli.telescope, verbose=not cli.quiet)

    # If none of the main options is given, do nothing
    if not cli.grid and not cli.opacity and not cli.monte_carlo and \
        not cli.raytrace and not cli.synobs and not cli.quiet:

        utils.print_('Nothing to do. Main options: ' +\
            '--grid, --opacity, --monte-carlo, --raytrace or --synobs')
        utils.print_('Run synthesizer --help for details.')