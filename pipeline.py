#!/usr/bin/env python3

import os, sys
import requests
import subprocess
import numpy as np
from pathlib import Path
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from glob import glob

from synthesizer import utils
from synthesizer import dustmixer
from synthesizer import gridder
from synthesizer import synobs


class Pipeline:
    
    def __init__(self, lam=1300, amax=10, nphot=1e5, nthreads=1, sootline=300,
            lmin=0.1, lmax=1e6, nlam=200, star=None, dgrowth=False, csubl=0, 
            material='sg', polarization=False, alignment=False, overwrite=False, verbose=True):
        self.steps = []
        self.lam = int(lam)
        self.lmin = lmin
        self.lmax = lmax
        self.nlam = nlam
        self.lgrid = np.logspace(np.log10(lmin), np.log10(lmax), nlam)
        self.amax = str(int(amax))
        self.material = material
        self.nphot = int(nphot)
        self.nthreads = int(nthreads)
        self.polarization = polarization
        self.alignment = alignment
        if polarization:
            self.scatmode = 5
            self.inputstyle = 10
        else:
            self.scatmode = 2
            self.inputstyle = 1

        if alignment:
            self.scatmode = 4
            self.inputstyle = 20
            self.polarization = True
    
        self.csubl = csubl
        self.nspec = 1 if self.csubl == 0 else 2
        self.dcomp = [material]*2 if self.csubl == 0 else [material, material+'o']
        self.sootline = sootline
        self.dgrowth = dgrowth
        if star is None:
            self.xstar = 0
            self.ystar = 0
            self.zstar = 0
            self.rstar = 2e11
            self.mstar = 3e22
            self.tstar = 4000        
        else:
            self.xstar = star[0]
            self.ystar = star[1]
            self.zstar = star[2]
            self.rstar = star[3]
            self.mstar = star[4]
            self.tstar = star[5]

        self.npix = None
        self.incl = None
        self.sizeau = None
        self.overwrite = overwrite
        self.verbose = verbose

    @utils.elapsed_time
    def create_grid(self, sphfile, source='sphng', ncells=None, bbox=None,  
            rout=None, vector_field=None, show_2d=False, show_3d=False, 
            vtk=False, render=False, g2d=100):
        """ Initial step in the pipeline: creates an input grid for RADMC3D """

        self.sphfile = sphfile
        self.ncells = ncells
        self.bbox = bbox
        self.rout = rout
        self.vector_field = vector_field
        self.g2d = g2d

        # Register the pipeline step 
        self.steps.append('create_grid')

        # Create a grid instance
        print('')
        utils.print_('Creating model grid ...\n', bold=True)
        self.grid = gridder.CartesianGrid(
            ncells=self.ncells, 
            bbox=self.bbox, 
            rout=self.rout,
            csubl=self.csubl, 
            nspec=self.nspec, 
            sootline=self.sootline, 
            g2d=self.g2d, 
        )

        # Read the SPH data
        self.grid.read_sph(self.sphfile, source=source.lower())

        # Set a bounding box to trim the new grid
        if self.bbox is not None:
            self.grid.trim_box(bbox=self.bbox * u.au.to(u.cm))

        # Set a radius at which to trim the new grid
        if self.rout is not None:
            self.grid.trim_box(rout=self.rout * u.au.to(u.cm))

        # Interpolate the SPH points onto a regular cartesian grid
        self.grid.interpolate_points(field='dens', show_2d=show_2d, show_3d=show_3d)
        self.grid.interpolate_points(field='temp', show_2d=show_2d, show_3d=show_3d)

        # Write the new cartesian grid to radmc3d file format
        self.grid.write_grid_file()

        # Write the dust density distribution to radmc3d file format
        self.grid.write_density_file()
        
        # Write the dust temperature distribution to radmc3d file format
        self.grid.write_temperature_file()

        if self.vector_field is not None:
            self.grid.write_vector_field(morphology=self.vector_field)
        
        # Call RADMC3D to read the grid file and generate a VTK representation
        if vtk:
            self.grid.create_vtk(dust_density=False, dust_temperature=True, rename=True)
        
        # Visualize the VTK grid file using ParaView
        if render:
            self.grid.render()

    @utils.elapsed_time
    def dust_opacity(self, amin, amax, na, q=-3.5, nang=3, material=None, 
            show_nk=False, show_opac=False, savefig=None):
        """
            Call dustmixer to generate dust opacity tables. 
            New dust materials can be manually defined here if desired.
        """

        print('')
        utils.print_("Calculating dust opacities ...\n", bold=True)

        if material is not None: self.material = material
        self.amin = amin
        self.amax = amax
        self.na = na
        self.q = q 
        self.a_dist = np.logspace(np.log10(amin), np.log10(amax), na)
        if self.polarization and nang < 181:
            self.nang = 181
        else:
            self.nang = nang
        #nth = self.nthreads
        # use 1 until the parallelization of polarization is properly implemented
        nth = 1
        
        if self.material == 's':
            mix = dustmixer.Dust(name='Silicate')
            mix.set_lgrid(self.lmin, self.lmax, self.nlam)
            mix.set_nk('astrosil-Draine2003.lnk', skip=1, get_density=True)
            mix.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
        
        elif self.material == 'g':
            mix = dustmixer.Dust(name='Graphite')
            mix.set_lgrid(self.lmin, self.lmax, self.nlam)
            mix.set_nk('c-gra-Draine2003.lnk', skip=1, get_density=True)
            mix.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)

        elif self.material == 'p':
            mix = dustmixer.Dust(name='Pyroxene')
            mix.set_lgrid(self.lmin, self.lmax, self.nlam)
            mix.set_nk('pyrmg70.lnk', get_density=False)
            mix.set_density(3.01, cgs=True)
            mix.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
        
        elif self.material == 'o':
            mix = dustmixer.Dust(name='Organics')
            mix.set_lgrid(self.lmin, self.lmax, self.nlam)
            mix.set_nk('organics.nk', get_density=False)
            mix.set_density(1.50, cgs=True)
            mix.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
        
        elif self.material == 'sg':
            sil = dustmixer.Dust(name='Silicate')
            gra = dustmixer.Dust(name='Graphite')
            sil.set_lgrid(self.lmin, self.lmax, self.nlam)
            gra.set_lgrid(self.lmin, self.lmax, self.nlam)
            sil.set_nk('astrosil-Draine2003.lnk', skip=1, get_density=True)
            gra.set_nk('c-gra-Draine2003.lnk', skip=1, get_density=True)
            sil.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
            gra.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)

            # Sum the opacities weighted by their mass fractions
            mix = sil * 0.625 + gra * 0.375

        else:
            try:
                mix = dustmixer.Dust(self.material.split('/')[-1].split('.')[0])
                mix.set_lgrid(self.lmin, self.lmax, self.nlam)
                mix.set_nk(path=self.material, skip=1, get_density=True)
                mix.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
                self.material = mix.name

            except Exception as e:
                utils.print_(e, red=True)
                raise ValueError(f'Material = {material} not found.')

        if show_nk or savefig is not None:
            mix.plot_nk(show=show_nk, savefig=savefig)

        if show_opac or savefig is not None:
            mix.plot_opacities(show=show_opac, savefig=savefig)

        # Write the opacity table
        mix.write_opacity_file(scatmat=self.polarization, 
            name=f'{self.material}-a{int(self.amax)}um')

        # Write the alignment efficiencies
        if self.alignment:
            mix.write_align_factor(f'{self.material}-a{int(self.amax)}um')

        # Register the pipeline step 
        self.steps.append('dustmixer')
    
    def radmc3d_banner(self):
        utils.print_(f'{"="*21}  <RADMC3D>  {"="*21}', bold=True)

    def generate_input_files(self, inpfile=False, wavelength=False, stars=False, 
            dustopac=False, dustkappa=False, dustkapalignfact=False,
            grainalign=False):
        """ Generate the necessary input files for radmc3d """
        if inpfile:
            # Create a RADMC3D input file
            with open('radmc3d.inp', 'w+') as f:
                f.write(f'incl_dust = 1\n')
                f.write(f'istar_sphere = 0\n')
                f.write(f'modified_random_walk = 1\n')
                f.write(f'setthreads = {self.nthreads}\n')
                f.write(f'nphot_scat = {self.nphot}\n')
                f.write(f'scattering_mode = {self.scatmode}\n')
                if self.alignment: f.write(f'alignment_mode = 1\n')

        if wavelength: 
            # Create a wavelength grid in micron
            with open('wavelength_micron.inp', 'w+') as f:
                f.write(f'{self.lgrid.size}\n')
                for wav in self.lgrid:
                    f.write(f'{wav:13.6}\n')

        if stars:
            # Create a stellar spectrum file
            with open('stars.inp', 'w+') as f:
                f.write('2\n')
                f.write(f'1 {self.lgrid.size}\n')
                f.write(f'{self.rstar} {self.mstar} ')
                f.write(f'{self.xstar} {self.ystar} {self.zstar}\n')
                for wav in self.lgrid:
                    f.write(f'{wav:13.6}\n')
                f.write(f'{-self.tstar}\n')

        if dustopac:
            # Create a dust opacity file
            self.amax = int(self.amax)
            with open('dustopac.inp', 'w+') as f:
                f.write('2\n')
                f.write(f'{self.nspec}\n')
                f.write('---------\n')
                f.write(f'{self.inputstyle}\n')
                f.write('0\n')
                if self.csubl > 0:
                    f.write(f'{self.dcomp[1]}-a{self.amax}um-'+ \
                        '{int(self.csubl)}org\n')
                else:
                    f.write(f'{self.material}-a{self.amax}um\n')

                if self.nspec > 1:
                    # Define a second species 
                    f.write('---------\n')
                    f.write(f'{self.inputstyle}\n')
                    f.write('0\n')
                    if self.dgrowth:
                        f.write(f'{self.dcomp[0]}-a1000um\n')
                    else:
                        f.write(f'{self.dcomp[0]}-a{self.amax}um\n')
                f.write('---------\n')

        if dustkappa:

            # Fetch the corresponding opacity table from a public repo
            table = 'https://raw.githubusercontent.com/jzamponi/utils/main/' +\
                f'opacity_tables/dustkappa_{self.dcomp[0]}-a{self.amax}um.inp'

            utils.download_file(table)

            if self.csubl > 0:
                # Download also the table for a second dust composition
                table = table.replace(f'{self.dcomp[0]}', f'{self.dcomp[1]}')
                if 'sgo' in table:
                    table = table.replace('um.inp', f'um-{int(self.csubl)}org.inp')

                if self.dgrowth:
                    # Download also the table for grown dust
                    table = table.replace(f'{self.amax}', '1000') 

                utils.download_file(table)

        if dustkapalignfact:
            # To do: convert the graphite_oblate.dat and silicate_oblate.dat 
            # from the polaris repo, into a radmc3d format. Then upload the
            # radmc3d table to my github repo and download it from here
            
            raise ImportError(f'{utils.color.red}There is no ' +\
                f'dustkapalignfact_*.inp file. Run the pipeline again with ' +\
                f'the option --opacity --alignment.{utils.color.none}')

        if grainalign:
            raise ImportError(f'{utils.color.red}There is no ' +\
                f'grainalign_dir.inp file. Run the pipeline again adding ' +\
                f'--vector-field to --grid to create the alignment field ' +\
                f'from the input model.{utils.color.none}')

    @utils.elapsed_time
    def monte_carlo(self, nphot, radmc3d_cmds=''):
        """ 
            Call radmc3d to calculate the radiative temperature distribution 
        """

        print('')
        utils.print_("Running a thermal Monte Carlo ...", bold=True)
        self.nphot = nphot
        self.radmc3d_banner()
        subprocess.run(
            f'radmc3d mctherm nphot {self.nphot} {radmc3d_cmds}'.split())
        self.radmc3d_banner()

        # Register the pipeline step 
        self.steps.append('monte_carlo')

    @utils.elapsed_time
    def raytrace(self, lam=None, incl=None, npix=None, sizeau=None, distance=141,
            show=True, noscat=True, fitsfile='radmc3d_I.fits', radmc3d_cmds=''):
        """ 
            Call radmc3d to raytrace the newly created grid and plot an image 
        """

        print('')
        utils.print_("Ray-tracing the model density and temperature ...\n", 
            bold=True)

        self.distance = distance

        if lam is not None:
            self.lam = lam
        if npix is not None:
            self.npix = npix
        if sizeau is not None:
            self.sizeau = sizeau
        if incl is not None:
            self.incl = incl

        # To do: What's the diff. between passing noscat and setting scatmode=0
        if noscat: self.scatmode = 0

        # Generate only the input files that are not available in the directory
        if not os.path.exists('radmc3d.inp') or self.overwrite:
            self.generate_input_files(inpfile=True)

        if not os.path.exists('wavelength_micron.inp') or self.overwrite:
            self.generate_input_files(wavelength=True)

        if not os.path.exists('stars.inp') or self.overwrite:
            self.generate_input_files(stars=True)

        # Write a new dustopac file only if dustmixer was used or if unexistent
        if not os.path.exists('dustopac.inp') or \
            'dustmixer' in self.steps or self.overwrite:
            self.generate_input_files(dustopac=True)

        # If opacites were calculated within the pipeline, don't overwrite them
        if 'dustmixer' not in self.steps:
            # If not manually provided, download it from the repo
            if not self.polarization:
                if len(glob('dustkappa*')) == 0 or self.overwrite:
                    self.generate_input_files(dustkappa=True)

        # If align factors were calculated within the pipeline, don't overwrite
        if self.alignment:
            if 'dustmixer' not in self.steps:
                # If not manually provided, download it from the repo
                if len(glob('dustkapalignfact*')) == 0:
                    self.generate_input_files(dustkapalignfact=True)

            if not os.path.exists('grainalign_dir.inp'):
                self.generate_input_files(grainalign=True)

        # Now double check that all necessary input files are available 
        utils.file_exists('amr_grid.inp')
        utils.file_exists('dust_density.inp')
        utils.file_exists('dust_temperature.dat')
        utils.file_exists('radmc3d.inp')
        utils.file_exists('wavelength_micron.inp')
        utils.file_exists('stars.inp')
        utils.file_exists('dustopac.inp')
        utils.file_exists('dustkapscat*' if self.polarization else 'dustkappa*')
        if self.alignment: 
            utils.file_exists('dustkapalignfact*')
            utils.file_exists('grainalign_dir.inp')
 
        # Explicitly the model rotate by 180.
        # Only for the current model. This line should be later removed.
        self.incl = 180 - int(self.incl)

        # Set the RADMC3D command by concatenating options
        cmd = f'radmc3d image '
        cmd += f'lambda {self.lam} '
        cmd += f'incl {self.incl} ' if self.incl is not None else ' '
        cmd += f'npix {self.npix} ' if self.npix is not None else ' '
        cmd += f'sizeau {self.sizeau} ' if self.sizeau is not None else ' '
        cmd += f'stokes ' if self.polarization else ' '
        cmd += f'{" ".join(radmc3d_cmds)} '
        
        # Call RADMC3D and pipe the output also to radmc3d.out
        utils.print_(f'Executing command: {cmd}')
        self.radmc3d_banner()

        try:
            os.system(f'{cmd} 2>&1 | tee radmc3d.out')
        except KeyboardInterrupt:
            raise Exception('Received SIGKILL. Execution halted by user.')

        self.radmc3d_banner()
        
        # Read radmc3d.out and stop the pipeline if RADMC3D finished in error
        with open ('radmc3d.out', 'r') as out:
            for line in out.readlines():
                if 'error' in line.lower() or 'stop' in line.lower():
                    raise Exception(
                        f'{utils.color.red}[RADMC3D] {line}{utils.color.none}')
    
        # Generate a FITS file from the image.out
        if os.path.exists(fitsfile): os.remove(fitsfile)
        utils.radmc3d_casafits(fitsfile, stokes='I', dpc=distance)

        # Clean extra keywords from the header to avoid APLPy axis errors
        utils.fix_header_axes(fitsfile)

        # Also for Q and U Stokes components if considering polarization
        if self.polarization:
            for s in ['Q', 'U']:
                # Write FITS file for each component
                stokesfile = f'radmc3d_{s}.fits'
                if os.path.exists(stokesfile):
                        os.remove(stokesfile)
                utils.radmc3d_casafits(stokesfile, stokes=s, dpc=distance)

                # Clean extra keywords from the header to avoid APLPy errors 
                utils.fix_header_axes(stokesfile)

        # Plot the new image in Jy/pixel
        if show:
            utils.print_('Plotting image.out')

            if self.polarization:
                fig = utils.polarization_map(
                    source='radmc3d',
                    render='I', 
                    rotate=90 if self.alignment else 0, 
                    step=15, 
                    scale=10, 
                    min_pfrac=0, 
                    const_pfrac=True, 
                    vector_color='white',
                    vector_width=1, 
                    verbose=False,
                    block=True, 
                )
            else:
                fig = utils.plot_map(
                    filename='radmc3d_I.fits', 
                    bright_temp=False,
                    verbose=False,
                )

        # Register the pipeline step 
        self.steps.append('raytrace')

    @utils.elapsed_time
    def synthetic_observation(self, show=False, cleanup=True, 
            script=None, simobserve=True, clean=True, exportfits=True, 
            obstime=None, resolution=None, graphic=True, verbose=False):
        """ 
            Prepare the input for the CASA simulator from the RADMC3D output,
            and call CASA to run a synthetic observation.
        """

        print('')
        utils.print_('Running synthetic observation ...\n', bold=True)

        if script is None:
            # Create a minimal template CASA script
            script = synobs.CasaScript(lam=self.lam)
            script.polarization = self.polarization
            script.simobserve = simobserve
            script.clean = clean
            script.graphic = graphic
            script.verbose = verbose
            script.overwrite = self.overwrite
            script.resolution = resolution
            if self.npix is not None: script.npix = int(self.npix + 20)
            if obstime is not None: script.totaltime = f'{obstime}h'
            script.verbose = verbose
            script.write('casa_script.py')

        elif 'http' in script: 
            # Download the script if a URL is provided
            url = script
            utils.download_file(url)
            script = synobs.CasaScript()
            script.name = script.split('/')[-1]
            script.read(script.name)

        # Call CASA
        script.run()

        # Show the new synthetic image
        if show:
            utils.print_(f'Plotting the new synthetic image')

            try:
                utils.file_exists(script.fitsimage('I'))
                utils.fix_header_axes(script.fitsimage('I'))
                    
                if self.polarization:
                    utils.file_exists(script.fitsimage('Q'))
                    utils.fix_header_axes(script.fitsimage('Q'))
                    utils.fix_header_axes(script.fitsimage('U'))

                    fig = utils.polarization_map(
                        source='obs', 
                        render='I', 
                        stokes_I=script.fitsimage('I'), 
                        stokes_Q=script.fitsimage('I'), 
                        stokes_U=script.fitsimage('I'), 
                        rotate=0, 
                        step=15, 
                        scale=10, 
                        const_pfrac=True, 
                        vector_color='white',
                        vector_width=1, 
                        verbose=True,
                    )
                else:
                    fig = utils.plot_map(
                        filename=script.fitsimage('I'),
                        bright_temp=False,
                        verbose=False,
                    )
            except Exception as e:
                utils.print_(
                    f'Unable to plot {script.fitsimage("I")}:\n{e}', bold=True)

        # Clean-up and remove unnecessary files created by CASA
        if cleanup:
            script.cleanup()

        # Register the pipeline step 
        self.steps.append('synobs')

