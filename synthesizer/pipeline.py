#!/usr/bin/env python3

import os
import sys
import copy
import random
import requests
import subprocess
import numpy as np
from glob import glob
from pathlib import Path
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from scipy.interpolate import griddata

from synthesizer import utils
from synthesizer import synobs
from synthesizer import gridder
from synthesizer import dustmixer

# Store the source code directory
source_path = Path(__file__).resolve()
source_dir = source_path.parent

class Pipeline:
    
    def __init__(self, lam=1300, amin=0.1, amax=10, na=100, q=3.5, nang=181, 
            nphot=1e5, nthreads=1, lmin=0.1, lmax=1e5, nlam=200, star=None, 
            dgrowth=False, csubl=0, sootline=300, material='sg', 
            polarization=False, alignment=False, 
            overwrite=False, verbose=True):
        self.steps = []
        self.lam = int(lam)
        self.lmin = lmin
        self.lmax = lmax
        self.nlam = nlam
        self.lgrid = np.logspace(np.log10(lmin), np.log10(lmax), nlam)
        self.amax = amax
        self.amin = amin
        self.na = na
        self.q = q
        self.nang = nang
        self.material = material
        self.nphot = int(nphot)
        self.nthreads = int(nthreads)
        self.npix = None
        self.incl = None
        self.sizeau = None
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
        if self.csubl > 0:
            self.nspec = 2
            self.material2 = self.material[:-1]
        else:
            self.nspec = 1

        self.sootline = sootline
        self.dgrowth = dgrowth
        self.kappa = None

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

        self.overwrite = overwrite
        self.verbose = verbose

    @utils.elapsed_time
    def create_grid(self, model=None, sphfile=None, amrfile=None, 
            source='sphng', bbox=None, rout=None, ncells=None, 
            vector_field=None, show_2d=False, show_3d=False, vtk=False, 
            render=False, g2d=100, temperature=True):
        """ Initial step in the pipeline: creates an input grid for RADMC3D """

        self.model = model
        self.sphfile = sphfile
        self.amrfile = amrfile
        self.ncells = ncells
        self.bbox = bbox * u.au.to(u.cm) if bbox is not None else bbox
        self.rout = rout * u.au.to(u.cm) if rout is not None else rout
        self.g2d = g2d

        # Make sure the model temp is read when c-sublimation is enabled
        if self.csubl > 0 and not temperature:
            utils.print_('--sublimation was given but not --temperature.') 
            utils.print_("I will set --temperature to read in the model's "+\
                f'temperature and set the sootline at {self.sootline} K.') 

            temperature = True


        # Create a grid instance
        print('')
        utils.print_('Creating model grid ...\n', bold=True)
    
        # Create a grid using an analytical model
        if model is not None:
            self.grid = gridder.AnalyticalModel(
                model=self.model,
                bbox=self.bbox, 
                ncells=self.ncells, 
                g2d=self.g2d,
                nspec=self.nspec,
                temp=temperature, 
            )
            
            # Create a model density grid 
            self.grid.create_model()

        # Create a grid from SPH particles
        elif sphfile is not None:
            self.grid = gridder.CartesianGrid(
                ncells=self.ncells, 
                bbox=self.bbox, 
                rout=self.rout,
                csubl=self.csubl, 
                nspec=self.nspec, 
                sootline=self.sootline, 
                g2d=self.g2d, 
                temp=temperature,
            )

            # Read the SPH data
            self.grid.read_sph(self.sphfile, source=source)

            # Set a bounding box or radius to trim particles outside of it
            if self.bbox is not None or self.rout is not None:
                self.grid.trim_box()
    
            # Interpolate the SPH points onto a regular cartesian grid
            self.grid.interpolate_points('dens', 'linear', fill='min')

            if temperature:
                self.grid.interpolate_points('temp', 'linear', fill='min')

        # Create a grid from an AMR grid
        elif amrfile is not None:
            self.grid = gridder.CartesianGrid(
                ncells=self.ncells, 
                bbox=self.bbox, 
                rout=self.rout,
                csubl=self.csubl, 
                nspec=self.nspec, 
                sootline=self.sootline, 
                g2d=self.g2d, 
                temp=temperature,
            )
            
            # Read the AMR data
            self.grid.read_amr(self.amrfile, source=source)
        
        else:
            raise ValueError(f'{utils.color.red}When --grid is set, either --'+\
                f'model, --sphfile or --amrfile must be given{utils.color.none}')

        self.bbox = self.grid.bbox

        # Write the new cartesian grid to radmc3d file format
        self.grid.write_grid_file()

        # Write the dust density distribution to radmc3d file format
        self.grid.write_density_file()
        
        if temperature:
            self.grid.write_temperature_file()

        if vector_field is not None:
            self.grid.write_vector_field(morphology=vector_field)

        # Plot the density midplane
        if show_2d:
            self.grid.plot_dens_midplane()

        # Plot the temperature midplane
        if show_2d and temperature:
            self.grid.plot_temp_midplane()

        # Render the density volume in 3D using Mayavi
        if show_3d:
            self.grid.plot_3d(density=True)

        # Render the temperature volume in 3D using Mayavi
        if show_3d and temperature:
            self.grid.plot_3d(temperature=True)
        
        # Call RADMC3D to read the grid file and generate a VTK representation
        if vtk:
            self.grid.create_vtk(
                dust_density=False, dust_temperature=True, rename=True)
        
        # Visualize the VTK grid file using ParaView
        if render:
            self.grid.render()

        # Register the pipeline step 
        self.steps.append('create_grid')


    @utils.elapsed_time
    def dustmixer(self, show_nk=False, pb=True, show_opac=False, savefig=None):
        """
            Call dustmixer to generate dust opacity tables. 
            New dust materials can be manually defined here if desired.
        """

        print('')
        utils.print_("Calculating dust opacities ...\n", bold=True)

        self.a_dist = np.logspace(
            np.log10(self.amin), np.log10(self.amax), self.na)

        if self.polarization and self.nang < 181:
            self.nang = 181

        # Use 1 until the parallelization of polarization is properly implemented
        nth = self.nthreads
        nth = 1

        mix = dustmixer.Dust()
        mix.pb = pb
        mix.set_lgrid(self.lmin, self.lmax, self.nlam)
        mix.scatmatrix = self.polarization

        pathnk = Path(source_dir/'dustmixer/nk')

        # Make sure the source n k tables are accesible. If not, download
        if not utils.file_exists(
            f'{pathnk}/astrosil-Draine2003.lnk', raise_=False):

            pathnk = 'https://raw.githubusercontent.com/jzamponi/'+\
                'utils/main/opacity_tables'

        if self.material == 's':
            mix.name = 'Silicate'
            mix.set_nk(f'{pathnk}/astrosil-Draine2003.lnk', skip=1, get_dens=True)
            mix.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
        
        elif self.material == 'g':
            mix.name = 'Graphite'
            mix.set_nk(f'{pathnk}/c-gra-Draine2003.lnk', skip=1, get_dens=True)
            mix.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
        
        elif self.material == 'o':
            mix.name = 'Organics'
            mix.set_nk(f'{pathnk}/c-org-Henning1996.lnk', microns=True, skip=1, 
                get_dens=True)
            mix.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)

        elif self.material == 'p':
            mix.name = 'Pyroxene-Mg70'
            mix.set_nk(f'{pathnk}/pyr-mg70-Dorschner1995.lnk', get_dens=False)
            mix.set_density(3.01, cgs=True)
            mix.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)

        elif self.material == 'sg':
            sil = copy.deepcopy(mix)
            sil.name = 'Silicate'
            sil.set_nk(f'{pathnk}/astrosil-Draine2003.lnk', skip=1, get_dens=True)
            sil.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
            if show_nk: sil.plot_nk(show=show_nk, savefig=savefig)

            gra = copy.deepcopy(mix)
            gra.name = 'Graphite'
            gra.set_nk(f'{pathnk}/c-gra-Draine2003.lnk', skip=1, get_dens=True)
            gra.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
            if show_nk: gra.plot_nk(show=show_nk, savefig=savefig)
            show_nk = False

            # Sum the opacities weighted by their mass fractions
            mix = sil * 0.625 + gra * 0.375

        elif self.material == 'sgo':
            mix.name = 'Sil-Gra-Org'
            sil = copy.deepcopy(mix)
            sil.name = 'Silicate'
            sil.set_nk(f'{pathnk}/astrosil-Draine2003.lnk', skip=1, get_dens=True)
            sil.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
            if show_nk: sil.plot_nk(show=show_nk, savefig=savefig)

            gra = copy.deepcopy(mix)
            gra.name = 'Graphite'
            gra.set_nk(f'{pathnk}/c-gra-Draine2003.lnk', skip=1, get_dens=True)
            gra.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
            if show_nk: gra.plot_nk(show=show_nk, savefig=savefig)

            org = copy.deepcopy(mix)
            org.name = 'Organics'
            org.set_nk(f'{pathnk}/c-org-Henning1996.lnk', microns=True, skip=1,
                 get_dens=True)
            org.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
            if show_nk: org.plot_nk(show=show_nk, savefig=savefig)
            show_nk = False

            mf_sil = 0.625
            mf_gra = 0.375
            if self.csubl > 0:
                # Carbon sublimation: 
                # Replace a percentage of graphite "csubl" by refractory organics
                mf_org = (self.csubl / 100) * mf_gra
                mf_gra = mf_gra - mf_org
            else:
                mf_org = 0

            # Sum the opacities weighted by their mass fractions
            mix = (sil * mf_sil) + (gra * mf_gra) + (org * mf_org)

        elif self.material == 'dsharp':
            utils.not_implemented('Opacity: DSHARP')

        elif self.material == 'diana':
            utils.not_implemented('Opacity: DIANA')
    
        else:
            try:
                mix.name = self.material.split('/')[-1].split('.')[0]
                mix.set_nk(path=self.material, skip=1, get_dens=True)
                self.material = mix.name

            except Exception as e:
                utils.print_(e, red=True)
                raise ValueError(f'Material = {self.material} not found.')

        if show_nk or savefig is not None:
            mix.plot_nk(show=show_nk, savefig=savefig)

        if show_opac or savefig is not None:
            mix.plot_opacities(show=show_opac, savefig=savefig)

        mix.write_opacity_file(name=self._get_opac_name(self.csubl))

        if self.alignment:
            mix.write_align_factor(name=self._get_opac_name(self.csubl))

        # Store the current dust opacity in the pipeline instance
        self.kappa = mix._get_kappa_at_lam(self.lam) 

        # Register the pipeline step 
        self.steps.append('dustmixer')
    

    def generate_input_files(self, mc=False, inpfile=False, wavelength=False, 
            stars=False, dustopac=False, dustkappa=False, dustkapalignfact=False,
            grainalign=False):
        """ Generate the necessary input files for radmc3d """

        if inpfile:
            # Create a RADMC3D input file
            with open('radmc3d.inp', 'w+') as f:
                f.write(f'incl_dust = 1\n')
                f.write(f'istar_sphere = 0\n')
                f.write(f'modified_random_walk = 1\n')
                f.write(f'setthreads = {self.nthreads}\n')
                f.write(f'nphot = {int(self.nphot)}\n')
                f.write(f'nphot_scat = {int(self.nphot)}\n')
                f.write(f'iseed = {random.randint(-1e4, 1e4)}\n')
                f.write(f'countwrite = {int(self.nphot / 100)}\n')
                f.write(f'mc_scat_maxtauabs = {int(5)}\n')
                f.write(f'scattering_mode = {self.scatmode}\n')
                if self.alignment and not mc: 
                    f.write(f'alignment_mode = 1\n')

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
            with open('dustopac.inp', 'w+') as f:
                f.write('2\n')
                f.write(f'{self.nspec}\n')
                f.write('---------\n')
                f.write(f'{self.inputstyle}\n')
                f.write('0\n')
                f.write(f'{self._get_opac_name(self.csubl)}\n')

                if self.nspec > 1:
                    # Define a second species 
                    f.write('---------\n')
                    f.write(f'{self.inputstyle}\n')
                    f.write('0\n')
                    if self.dgrowth:
                        f.write(f'{self._get_opac_name(dgrowth=self.dgrowth)}\n')
                    else:
                        f.write(f'{self._get_opac_name(material2=True)}\n')
                f.write('---------\n')

        if dustkappa:

            amax = int(self.amax)

            # Fetch the corresponding opacity table from a public repo
            repo = 'https://raw.githubusercontent.com/jzamponi/utils/main/' +\
                f'opacity_tables'
    
            if self.csubl > 0:
                # Download opacity of the first species 
                utils.download_file(
                    f'{repo}/dustkappa_{self._get_opac_name(self.csubl)}.inp')

                # Download opacity of the second species. If dgrowth, use 1000um
                if self.dgrowth:
                    utils.download_file(
                        f'{repo}/dustkappa_' +\
                        f'{self._get_opac_name(dgrowth=self.dgrowth)}.inp')
                else:
                    utils.download_file(f'{repo}/' +\
                        f'dustkappa_{self._get_opac_name(material2=True)}.inp')

            else:
                # Download opacity for the single dust species 
                utils.download_file(
                    f'{repo}/dustkappa_{self._get_opac_name()}.inp')


        if dustkapalignfact:
            # To do: convert the graphite_oblate.dat and silicate_oblate.dat 
            # from the polaris repo, into a radmc3d format. Then upload the
            # radmc3d table to my github repo and download it from here
            
            raise ImportError(f'{utils.color.red}There is no ' +\
                f'dustkapalignfact_*.inp file. Run synthesizer again with ' +\
                f'the option --opacity --alignment.{utils.color.none}')

        if grainalign:
            raise ImportError(f'{utils.color.red}There is no ' +\
                f'grainalign_dir.inp file. Run synthesizer again adding ' +\
                f'--vector-field to --grid to create the alignment field ' +\
                f'from the input model.{utils.color.none}')

    @utils.elapsed_time
    def monte_carlo(self, nphot, radmc3d_cmds=''):
        """ 
            Call radmc3d to calculate the radiative temperature distribution 
        """

        print('')
        utils.print_("Running a thermal Monte Carlo ...", bold=True)

        # Make sure RADMC3D is installed and callable
        utils.which('radmc3d') 

        self.nphot = nphot

        # Generate only the input files that are not available in the directory
        self.generate_input_files(inpfile=True, mc=True)

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
                    try:
                        self.generate_input_files(dustkappa=True)
                    except Exception as e:
                        utils.print_(
                            f'Unable to download opacity table. I will call ' +\
                            'dustmixer, as in synthesizer --opacity using '+\
                            'default values.', blue=True)

                        self.dust = self.dustmixer()
            else:
                if len(glob('dustkapscat*')) == 0 or self.overwrite:
                    utils.print_(
                        f'Unable to download opacity table. I will call ' +\
                        'dustmixer, as in synthesizer --opacity using '+\
                        'defualt values.', blue=True)

                    self.dust = self.dustmixer()

        # Call RADMC3D and pipe the output also to radmc3d.out
        try:
            utils.print_(f'Executing command: radmc3d mctherm {radmc3d_cmds}')
            self._radmc3d_banner()
            os.system(
                f'radmc3d mctherm {radmc3d_cmds} 2>&1 | tee -a radmc3d.out')

        except KeyboardInterrupt:
            raise Exception('Received SIGKILL. Execution halted by user.')

        self._radmc3d_banner()
        
        self._catch_radmc3d_error()

        # Register the pipeline step 
        self.steps.append('monte_carlo')

    @utils.elapsed_time
    def raytrace(self, lam=None, incl=None, npix=None, sizeau=None, show=True, 
            distance=141, tau=False, tau_surf=None, show_tau_surf=False, 
            noscat=False, fitsfile='radmc3d_I.fits', radmc3d_cmds=''):
        """ 
            Call radmc3d to raytrace the newly created grid and plot an image 
        """

        print('')
        utils.print_("Ray-tracing the model density and temperature ...\n", 
            bold=True)

        # Make sure RADMC3D is installed and callable
        utils.which('radmc3d') 

        self.distance = distance
        self.tau = tau
        self.tau_surf = tau_surf

        if lam is not None:
            self.lam = lam
        if npix is not None:
            self.npix = npix
        if incl is not None:
            self.incl = incl
        if sizeau is not None:
            self.sizeau = sizeau 
        else:
            self.sizeau = int(2 * self._get_bbox() * u.cm.to(u.au))

        # Explicitly the model rotate by 180.
        # Only for the current model. This line should be later removed.
        self.incl = 180 - int(self.incl)

        # To do: What's the diff. between passing noscat and setting scatmode=0
        if noscat: self.scatmode = 0

        # Generate only the input files that are not available in the directory
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
                    try:
                        self.generate_input_files(dustkappa=True)
                    except Exception as e:
                        utils.print_(
                            f'Unable to download opacity table. I will call ' +\
                            'dustmixer, as in synthesizer --opacity using '+\
                            'default values.', blue=True)
                        
                        self.dust = self.dustmixer()
            else:
                if len(glob('dustkapscat*')) == 0 or self.overwrite:
                    utils.print_(
                        f'Unable to download opacity table. I will call ' +\
                        'dustmixer, as in synthesizer --opacity using '+\
                        'default values.', blue=True)
                    
                    self.dust = self.dustmixer()

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

        # Generate a 2D optical depth map
        if self.tau:
            self.plot_tau(show)
            
        # Set the RADMC3D command by concatenating options
        cmd = f'radmc3d image '
        cmd += f'lambda {self.lam} '
        cmd += f'sizeau {self.sizeau} '
        cmd += f'noscat ' if noscat else ' '
        cmd += f'stokes ' if self.polarization else ' '
        cmd += f'incl {self.incl} ' if self.incl is not None else ' '
        cmd += f'npix {self.npix} ' if self.npix is not None else ' '
        cmd += f'{" ".join(radmc3d_cmds)} '
        

        # Call RADMC3D and pipe the output also to radmc3d.out
        try:
            utils.print_(f'Executing command: {cmd}')
            self._radmc3d_banner()
            os.system(f'{cmd} 2>&1 | tee --append radmc3d.out')

        except KeyboardInterrupt:
            raise Exception('Received SIGKILL. Execution halted by user.')

        self._radmc3d_banner()
        
        self._catch_radmc3d_error()

        utils.print_(
            f'Dust opacity used: kappa({self.lam}um) = ' +\
            f'{self._get_opacity():.2} cm2/g', blue=True)

        # Generate FITS files from the image.out
        utils.radmc3d_casafits(fitsfile, stokes='I', dpc=distance)

        # Write pipeline's info to the FITS headers
        for st in ['I', 'Q', 'U'] if self.polarization else ['I']:
            stfile = fitsfile.replace('I', st)
            utils.radmc3d_casafits(stfile, stokes=st, dpc=distance)
            utils.edit_header(stfile, 'INCL', self.incl)
            utils.edit_header(stfile, 'NPHOT', self.nphot)
            utils.edit_header(stfile, 'OPACITY', self.kappa)
            utils.edit_header(stfile, 'MATERIAL', self.material)
            utils.edit_header(stfile, 'CSUBL', self.csubl)
            utils.edit_header(stfile, 'NSPEC', self.nspec)

        # Plot the new image in Jy/pixel
        if show:
            self.plot_rt()

        # Generate a 3D surface at tau = tau_surf
        if self.tau_surf is not None:
            try:
                utils.print_(
                    'Generating tau surface at tau = {self.tau_surf}')
                os.system(f'radmc3d tausurf {self.tau_surf} ' +\
                f'lambda {self.lam} noscat '
                f'npix {self.npix} ' if self.npix is not None else ' ' +\
                f'sizeau {self.sizeau} ' if self.sizeau is not None else ' '+\
                f'incl {self.incl} ' if self.incl is not None else ' ')

                os.rename('image.out', 'tauimage.out')
            except Exception as e:
                utils.print_(f'Unable to generate tau surface.\n{e}\n', red=True)

        # Render the 3D surface in 
        if show_tau_surf:
            utils.not_implemented()
            from mayavi import mlab
            utils.file_exists('tausurface_3d.out')

        # Register the pipeline step 
        self.steps.append('raytrace')


    @utils.elapsed_time
    def synthetic_observation(self, show=False, cleanup=True, 
            script=None, simobserve=True, clean=True, exportfits=True, 
            obstime=1, resolution=None, obsmode='int', graphic=True, 
            use_template=False, telescope=None, verbose=False, 
            ):
        """ 
            Prepare the input for the CASA simulator from the RADMC3D output,
            and call CASA to run a synthetic observation.
        """

        print('')
        utils.print_('Running synthetic observation ...\n', bold=True)

        # Make sure RADMC3D is installed and callable
        utils.which('casa') 

        # If the observing wavelength is outside the working range of CASA, 
        # simplify the synthetic obs. to a PSF convolution and thermal noise
        if self.lam < 400 or self.lam > 4e6:
    
            if resolution is not None:
                self.resolution = resolution
            else:
                utils.print_(
                    '--resolution has not been set. I will use 0.1"', blue=True)
                self.resolution = 0.1

            img = synobs.SynImage('radmc3d_I.fits')
            img.convolve(self.resolution)
            img.add_noise(obstime, bandwidth=8*u.GHz.to(u.Hz))
            img.write_fits('synobs_I.fits')

            if self.polarization:
                img_q = synobs.SynImage('radmc3d_Q.fits')
                img_u = synobs.SynImage('radmc3d_U.fits')
                img_q.convolve(self.resolution)
                img_u.convolve(self.resolution)
                img_q.add_noise(obstime, bandwidth=8*u.GHz.to(u.Hz))
                img_u.add_noise(obstime, bandwidth=8*u.GHz.to(u.Hz))
                img_q.write_fits('synobs_Q.fits')
                img_u.write_fits('synobs_U.fits')

        # Use casa simobserve/tclean
        else:
            if script is None:

                script = synobs.CasaScript(lam=self.lam)

                if use_template and self.lam not in [1300, 3000, 7000, 18000]: 
                    utils.print_(
                        f"There's no template available at {self.lam} um "+\
                        'I will create a simple one named casa_script.py. '+\
                        'You can modify it later and re-run with ' +\
                        'synthesizer --synobs --script casa_script.py',
                        bold=True)
                    use_template = False

                if use_template:
                    utils.print_(
                        f'Using template casa script at {self.lam} microns')

                    template = {
                        1300: 'alma_1.3mm.py',
                        3000: 'alma_3mm.py',
                        7000: 'vla_7mm.py',
                        18000: 'vla_18mm.py',
                    }[self.lam]

                    if self.polarization: template.replace('1.3mm', '1.3mm_pol')

                    script.name = str(source_dir/f'synobs/templates/{template}')
                    script.read(script.name)                    

                else:
                    # Create a minimal template CASA script
                    script.resolution = resolution
                    script.obsmode = obsmode
                    script.telescope = telescope
                    if self.npix is not None: script.imsize = int(self.npix + 100)
                    script.totaltime = f'{obstime}h'

                # Tailor the script to the pipeline setup
                script.polarization = self.polarization
                script.simobserve = simobserve
                script.clean = clean
                script.graphic = graphic
                script.overwrite = self.overwrite
                script.verbose = False
                script.write('casa_script.py')

            else:
                if 'http' in script: 
                    # Download the script if a URL is provided
                    utils.download_file(script)

                script_name = script.split('/')[-1]
                script = synobs.CasaScript(lam=self.lam)
                script.read(script_name)

            self.script = script

            # Call CASA
            script.run()

            # Add the missing fits frequency keyword
            for s in ['I', 'Q', 'U'] if self.polarization else ['I']:
                utils.edit_header(f'synobs_{s}.fits', 'RESTFRQ', script.reffreq)

            # Clean-up and remove unnecessary files created by CASA 
            if not simobserve or not clean or not exportfits:
                script.cleanup()

        # Show the new synthetic image
        if show:
            self.plot_synobs()

        # Register the pipeline step 
        self.steps.append('synobs')



    def plot_rt(self):
        utils.print_('Plotting image.out')

        try:
            if self.alignment:
                utils.print_(f'Rotating vectors by 90 deg.', blue=True)

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
            fig.axis_labels.hide()
            fig.tick_labels.hide()
        except Exception as e:
            utils.print_(
                f'Unable to plot radmc3d image.\n{e}', bold=True)
    
    def plot_synobs(self):
        utils.print_(f'Plotting the new synthetic image')

        try:
            utils.file_exists('synobs_I.fits')
            utils.fix_header_axes('synobs_I.fits')
                
            if self.polarization:
                utils.file_exists('synobs_Q.fits')
                utils.file_exists('synobs_U.fits')
                utils.fix_header_axes('synobs_Q.fits')
                utils.fix_header_axes('synobs_U.fits')

                fig = utils.polarization_map(
                    source='obs', 
                    render='I', 
                    stokes_I='synobs_I.fits', 
                    stokes_Q='synobs_Q.fits', 
                    stokes_U='synobs_U.fits', 
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
                    filename='synobs_I.fits',
                    bright_temp=True,
                    verbose=False,
                )
        except Exception as e:
            utils.print_(
                f'Unable to plot synobs_I.fits:\n{e}', bold=True)

    def plot_tau(self, show=False):
        utils.print_(f'Generating optical depth map at {self.lam} microns')
        utils.file_exists('dust_density.inp')
        utils.file_exists('amr_grid.inp')
        rho = np.loadtxt('dust_density.inp', skiprows=3)
        amr = np.loadtxt('amr_grid.inp', skiprows=6)

        # Integrate density weigthed by the dust opacity along the line-of-sight
        dl = np.diff(amr)[0]
        nx = int(np.cbrt(rho.size))
        rho = rho.reshape((nx, nx, nx))
        tau2d = np.sum(rho * self._get_opacity() * dl, axis=0).T

        if show:
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.top'] = True
            plt.rcParams['ytick.right'] = True
            plt.rcParams['xtick.minor.visible'] = True
            plt.rcParams['ytick.minor.visible'] = True
            plt.title(fr'Optical depth at $\lambda = ${self.lam}$\mu$m')
            plt.imshow(tau2d, origin='lower')
            plt.yticks([])
            plt.xticks([])
            plt.colorbar()
            plt.show()

        utils.write_fits('tau.fits', data=tau2d, overwrite=True)

    def _get_opac_name(self, csubl=0, dgrowth=False, material2=False):
        """ Get the name of the currently used opacity file dustk*.inp """

        amax = int(self.amax)
        csubl = int(csubl)

        if csubl > 0:
            opacname = f'{self.material}-a{amax}um-{csubl}org'

        elif material2:
            opacname = f'{self.material2}-a{amax}um'

        elif dgrowth:
            opacname = f'{self.material2}-a1000um'

        else:
            opacname = f'{self.material}-a{amax}um'

        prefix = 'dustkapscat_' if self.polarization else 'dustkappa_'
        self.opacfile = prefix + opacname + '.inp'

        return opacname 
    
    def _get_opacity(self):
        """ Read in an opacity file, interpolate and find the opacity at lambda """
        from scipy.interpolate import interp1d

        # Check if it was created by dustmixer
        if self.kappa is not None:
           return self.kappa 

        # Generate the opacfile string and make sure file exists
        self._get_opac_name(csubl=self.csubl)

        try:
            # Read in opacity file (dustkap*.inp) and interpolate to find k_ext
            utils.file_exists(self.opacfile) 
            header = np.loadtxt(self.opacfile, max_rows=2)
            iformat = int(header[0])
            nlam = int(header[1])
            skip = 3 if 'kapscat' in self.opacfile else 2
            d = ascii.read(self.opacfile, data_start=skip, data_end=nlam + skip)
            l = d['col1']
            self.k_abs = d['col2']
            self.k_sca = d['col3'] if iformat > 1 else np.zeros(l.shape)
            self.k_ext = self.k_abs + self.k_sca
            self.kappa = float(interp1d(l, self.k_ext)(self.lam))

            return self.kappa

        except Exception as e:
            utils.print_(
                f"{e} I couldn't obtain the opacity from {self.opacfile}. " +\
                "I will assume k_ext = 1 g/cm3.")

            return 1.0

    def _radmc3d_banner(self):
        print(
            f'{utils.color.blue}{"="*31}  RADMC3D  {"="*31}{utils.color.none}')

    def _catch_radmc3d_error(self):
        """ Raise an exception to halt synthesizer if RADMC3D ended in Error """

        # Read radmc3d.out and stop the pipeline if RADMC3D finished in error
        utils.file_exists('radmc3d.out')
        with open ('radmc3d.out', 'r') as out:
            for line in out.readlines():
                line = line.lower()
                if 'error' in line or 'stop' in line:

                    msg = lambda m: f'{utils.color.red}\r{m}{utils.color.none}'
                    errmsg = msg(f'[RADMC3D] {line}...')

                    if 'g=<cos(theta)>' in line or 'the scat' in line:
                        errmsg += f' Try increasing --na or --nang '
                        errmsg += f'(currently, na: {self.na} nang {self.nang})'

                    raise Exception(errmsg)

    def _get_bbox(self):
        """ Return the current value for bbox if existent, namely, 
            if --grid was given. Otherwise read from the grid file.
        """

        try:
            if self.bbox is not None:
                return self.bbox
            else:
                g = np.loadtxt('amr_grid.inp', skiprows=6)
                return (g[-1] - g[0]) / 2
        except AttributeError:
            g = np.loadtxt('amr_grid.inp', skiprows=6)
            return (g[-1] - g[0]) / 2

    def plot_grid_2d(self):
        """ Plot the grid's density and temperature midplanes from files,
            in case they are not currently available from pipeline.grid
        """
        utils.not_implemented('Plot grid midplane')
        
    def plot_grid_3d(self):
        """ Render the grid's density and temperature in 3D from files,
            in case they are not currently available from pipeline.grid
        """
        utils.not_implemented('Render grid 3D')
