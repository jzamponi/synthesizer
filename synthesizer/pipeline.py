#!/usr/bin/env python3

import os
import sys
import copy
import random
import requests
import warnings
import subprocess
import numpy as np
from glob import glob
from pathlib import Path
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.constants as const
from astropy.io import ascii, fits
from scipy.interpolate import griddata

from synthesizer import utils
from synthesizer import raytrace
from synthesizer import synobs
from synthesizer import gridder
from synthesizer import dustmixer

# Store the source code directory
source_path = Path(__file__).resolve()
source_dir = source_path.parent

class Pipeline:
    
    def __init__(self, lam=1300, amin=0.1, amax=10, na=100, q=-3.5, nang=181, 
            nphot=1e5, nthreads=1, lmin=0.1, lmax=1e5, nlam=200, star=None, 
            dgrowth=False, csubl=0, sootline=300, bbox=None,
            material='s', mfrac=1, porosity=0, mixing='bruggeman', 
            polarization=False, alignment=False, print_photons=False, 
            overwrite=False, verbose=True):

        self.steps = []
        self.lam = int(lam)
        self.freq = const.c.cgs.value / (self.lam * u.micron.to(u.cm))
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
        self.mfrac = mfrac
        self.porosity = porosity
        self.mixing = mixing
        self.nphot = int(nphot)
        self.nthreads = int(nthreads)
        self.npix = None
        self.incl = None
        self.phi = None
        self.sizeau = None
        self.polarization = polarization
        self.alignment = alignment
        self.distance = 140
        self.cmap = 'magma'
        self.stretch = 'linear'
        self.print_photons = print_photons

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

        self.xstar = 0
        self.ystar = 0
        self.zstar = 0
        self.rstar = 2e11
        self.mstar = 2e33
        self.tstar = 4000        

        self.bbox = bbox
        self.overwrite = overwrite
        self.verbose = verbose

    @utils.elapsed_time
    def create_grid(
            self, model=None, sphfile=None, amrfile=None, 
            source='sphng', bbox=None, ncells=100, tau=False, 
            vector_field=None, show_2d=False, show_3d=False, vtk=False, 
            render=False, g2d=100, temperature=False, show_particles=False, 
            alignment=False, cmap=None, rin=None, rout=None, rc=None, r0=None, 
            h0=None, alpha=None, flare=None, mdisk=None, r_rim=None, 
            r_gap=None, w_gap=None, dr_gap=None, rho0=None, 
        ):
        """ Initial step in the pipeline: creates an input grid for RADMC3D """

        self.model = model
        self.sphfile = sphfile
        self.amrfile = amrfile
        self.ncells = ncells
        self.g2d = g2d
        self.bbox = bbox * u.au.to(u.cm) if bbox is not None else bbox
        self.rin = rin * u.au.to(u.cm) if rin is not None else rin
        self.rout = rout * u.au.to(u.cm) if rout is not None else rout
        self.rc = rc * u.au.to(u.cm) if rc is not None else rc
        self.r0 = r0 * u.au.to(u.cm) if r0 is not None else r0
        self.h0 = h0 * u.au.to(u.cm) if h0 is not None else h0
        self.mdisk = mdisk * u.Msun.to(u.g) if mdisk is not None else mdisk
        self.r_rim = r_rim * u.au.to(u.cm) if r_rim is not None else r_rim
        self.r_gap = r_gap * u.au.to(u.cm) if r_gap is not None else r_gap
        self.w_gap = w_gap * u.au.to(u.cm) if w_gap is not None else w_gap
        self.dr_gap = dr_gap * u.au.to(u.cm) if dr_gap is not None else dr_gap
        self.alpha = alpha 
        self.flare = flare 
        self.rho0 = rho0

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
            regular = True

            self.grid = gridder.AnalyticalModel(
                model=self.model,
                bbox=self.bbox, 
                ncells=self.ncells, 
                g2d=self.g2d,
                nspec=self.nspec,
                temp=temperature, 
                rin=self.rin, 
                rout=self.rout,
                rc=self.rc, 
                r0=self.r0, 
                h0=self.h0,
                alpha=self.alpha,
                flare=self.flare,
                mdisk=self.mdisk,
                r_rim=self.r_rim,
                r_gap=self.r_gap,
                w_gap=self.w_gap,
                dr_gap=self.dr_gap,
                rho0=self.rho0, 
            )
            
            # Create a model density grid 
            self.grid.create_model()

        # Create a grid from SPH particles
        elif sphfile is not None:
            regular = True

            self.grid = gridder.CartesianGrid(
                ncells=self.ncells, 
                bbox=self.bbox, 
                rout=self.rout,
                csubl=self.csubl, 
                nspec=self.nspec, 
                sootline=self.sootline, 
                g2d=self.g2d, 
                temp=temperature,
                vfield=alignment,
            )

            # Read the SPH data
            self.grid.read_sph(self.sphfile, source=source)

            # Set a bounding box or radius to trim particles outside of it
            if self.bbox is not None or self.rout is not None:
                self.grid.trim_box()

            # Render SPH particles in 3D space before interpolation
            if show_particles:
                self.grid.plot_particles()
    
            # Interpolate the SPH points onto a regular cartesian grid
            self.grid.interpolate('dens', 'linear', fill='min')

            if temperature:
                self.grid.interpolate('temp', 'linear', fill='min')

            if alignment:
                self.grid.interpolate('vx', 'linear', fill=0)
                self.grid.interpolate('vy', 'linear', fill=0)
                self.grid.interpolate('vz', 'linear', fill=0)

        # Create a grid from an AMR grid
        elif amrfile is not None:
            regular = False    

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
            raise ValueError(
                f'{utils.color.red}When --grid is set, either --model, ' +\
                f'--sphfile or --amrfile must be given{utils.color.none}'
            )

        self.bbox = self.grid.bbox

        # Write the new cartesian grid to radmc3d file format
        self.grid.write_grid_file(regular=regular)

        # Write the dust density distribution to radmc3d file format
        self.grid.write_density_file()
        
        if temperature:
            self.grid.write_temperature_file()

        if vector_field is not None or alignment:
            self.grid.write_vector_field(morphology=vector_field)

        # Plot the density midplane
        if show_2d:
            self.grid.plot_2d('density', cmap=cmap)

        # Plot the temperature midplane
        if show_2d and temperature:
            self.grid.plot_2d('temperature', cmap=cmap)

        # Render the density volume in 3D using Mayavi
        if show_3d:
            self.grid.plot_3d('density', tau=tau, cmap=cmap)

        # Render the temperature volume in 3D using Mayavi
        if show_3d and temperature:
            self.grid.plot_3d('temperature')
        
        # Call RADMC3D to read the grid file and generate a VTK representation
        if vtk:
            self.grid.create_vtk(dust_density=True, rename=True)

            if temperature:
                self.grid.create_vtk(dust_temperature=True, rename=True)
        
        # Visualize the VTK grid file using ParaView
        if render:
            self.grid.render(dust_density=True)

            if temperature:
                self.grid.render(dust_temperature=True)

        # Register the pipeline step 
        self.steps.append('create_grid')


    @utils.elapsed_time
    def dustmixer(self, 
            material=None,
            mfrac=None, 
            porosity=None, 
            mixing=None, 
            amin=None,
            amax=None,
            q=None,
            na=None,
            nang=None,
            polarization=False,
            show_nk=False, 
            show_z12z11=False,
            show_dust_eff=False, 
            show_opac=False, 
            pb=True, 
            savefig=None
        ):
        """
            Call dustmixer to generate dust opacity tables. 
            New dust materials can be manually defined here if desired.
        """

        if mfrac is not None: self.mfrac = mfrac
        self.porosity = 0 if porosity is None else porosity
        if mixing is not None: self.mixing = mixing.lower()
        if amin is not None: self.amin = amin
        if amax is not None: self.amax = amax
        if na is not None: self.na = na
        if q is not None: self.q = q
        if nang is not None: self.nang = nang
        if polarization is not None: self.polarization = polarization
        if self.polarization and self.nang < 181: self.nang = 181

        # Parse whether material is a single key or a list of materials to mix
        if material is not None: 
            if isinstance(material, (list, np.ndarray)):
                if len(material) ==  1:
                    self.material = material[0]
                else:
                    self.material = material
            else:
                self.material = material

        # Make sure mfrac is also set when using providing multiple materials
        if isinstance(self.material, (list, np.ndarray)):
            if mfrac is None or len(self.material) != len(mfrac):
                raise ValueError(f'{utils.color.red}Make sure to set --mfrac'+\
                    f' when providing multiple materials{utils.color.none}')

            # Make sure all mass fractions add up to 1
            if np.sum(mfrac) != 1:
                raise ValueError(
                    f'mass fractions should add up to 1 but {mfrac = }')

        # Make sure porosity is betwen 0 and 1
        if self.porosity < 0 or self.porosity >= 1:
            raise ValueError(f'Porosity must be between 0 and 1. {porosity = }')

        # Use 1 until the parallelization with polarization is fully implemented
        nth = self.nthreads
        nth = 1

        print('')
        utils.print_("Calculating dust opacities ...\n", bold=True)

        # Initialize a Dust object and set its wavelenght grid to the pipeline's
        dust = dustmixer.Dust()
        dust.amin = self.amin
        dust.amax = self.amax
        dust.na = self.na
        dust.nang = self.nang
        dust.scatmatrix = self.polarization
        dust.pb = pb

        # Create a wavelength grid
        dust.set_lgrid(self.lmin, self.lmax, self.nlam)

        # Create a dust grain size grid
        self.a_dist = np.logspace(
            np.log10(self.amin), np.log10(self.amax), self.na)

        # Source code location where the optical constants .lnk are stored
        pathnk = Path(source_dir/'dustmixer/nk')

        # Make sure the source n k tables are accesible. If not, download
        if not utils.file_exists(f'{pathnk}/astrosil-Draine2003.lnk', 
            raise_=False):

            pathnk = 'https://raw.githubusercontent.com/jzamponi/'+\
                'utils/main/opacity_tables'

        # Set predefined dust materials
        if self.material in ['s', 'sil']:
            utils.print_('Using silicate as dust material')
            dust.name = 'Silicate'
            dust.set_nk(f'{pathnk}/astrosil-Draine2003.lnk')
            if show_nk: dust.plot_nk(savefig=savefig)
            dust.get_opacities(self.a_dist, self.q, self.nang, nth)
        
        elif self.material in ['g', 'gra']:
            utils.print_('Using graphite as dust material')
            dust.name = 'Graphite'
            dust.set_nk(f'{pathnk}/c-gra-Draine2003.lnk')
            if show_nk: dust.plot_nk(savefig=savefig)
            dust.get_opacities(self.a_dist, self.q, self.nang, nth)
        
        elif self.material in ['o', 'org']:
            utils.print_('Using refractory organics as dust material')
            dust.name = 'Organics'
            dust.set_nk(f'{pathnk}/c-org-Henning1996.lnk')
            if show_nk: dust.plot_nk(savefig=savefig)
            dust.get_opacities(self.a_dist, self.q, self.nang, nth)

        elif self.material in ['p', 'pyr']:
            utils.print_('Using pyroxene (Mg70) as dust material')
            dust.name = 'Pyroxene-Mg70'
            dust.set_nk(f'{pathnk}/pyr-mg70-Dorschner1995.lnk', get_dens=False)
            dust.set_density(3.01, cgs=True)
            if show_nk: dust.plot_nk(savefig=savefig)
            dust.get_opacities(self.a_dist, self.q, self.nang, nth)

        elif self.material in ['t', 'tro']:
            utils.print_('Using troilite as dust material')
            dust.name = 'Troilite'
            dust.set_nk(f'{pathnk}/fes-Henning1996.lnk', get_dens=False)
            dust.set_density(4.83, cgs=True)
            if show_nk: dust.plot_nk(savefig=savefig)
            dust.get_opacities(self.a_dist, self.q, self.nang, nth)

        elif self.material in ['i', 'ice']:
            utils.print_('Using water ice as dust material')
            dust.name = 'Ice'
            dust.set_nk(f'{pathnk}/h2o-w-Warren2008.lnk', get_dens=False)
            dust.set_density(0.92, cgs=True)
            if show_nk: dust.plot_nk(savefig=savefig)
            dust.get_opacities(self.a_dist, self.q, self.nang, nth)

        elif self.material in ['sg', 'silgra']:
            utils.print_('Creating a mix of silicate and graphite')
            sil = copy.deepcopy(dust)
            gra = copy.deepcopy(dust)
            sil.name = 'Silicate'
            gra.name = 'Graphite'

            sil.set_nk(f'{pathnk}/astrosil-Draine2003.lnk')
            gra.set_nk(f'{pathnk}/c-gra-Draine2003.lnk')
            [d.plot_nk(savefig=savefig) for d in [sil, gra] if show_nk]

            sil.get_opacities(self.a_dist, self.q, self.nang, nth)
            gra.get_opacities(self.a_dist, self.q, self.nang, nth)

            # Sum the opacities weighted by their mass fractions
            dust = sil * 0.625 + gra * 0.375

        elif self.material in ['sgo', 'silgraorg']:
            utils.print_('Creating a mix of silicate, graphite and organics.')
            dust.name = 'Sil-Gra-Org'
            sil = copy.deepcopy(dust)
            gra = copy.deepcopy(dust)
            org = copy.deepcopy(dust)
            sil.name = 'Silicate'
            gra.name = 'Graphite'
            org.name = 'Organics'

            sil.set_nk(f'{pathnk}/astrosil-Draine2003.lnk')
            gra.set_nk(f'{pathnk}/c-gra-Draine2003.lnk')
            org.set_nk(f'{pathnk}/c-org-Henning1996.lnk')
            [d.plot_nk(savefig=savefig) for d in [sil, gra, org] if show_nk]

            sil.get_opacities(self.a_dist, self.q, self.nang, nth)
            gra.get_opacities(self.a_dist, self.q, self.nang, nth)
            org.get_opacities(self.a_dist, self.q, self.nang, nth)

            # This type of mixing weight the opacities with their mass fractions
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
            dust = (sil * mf_sil) + (gra * mf_gra) + (org * mf_org)

        elif self.material == 'dsharp':
            utils.print_('Creating DSHARP mix: water ice (20%), '+\
                'silicate (33%), troilite (7.4%), organics (39.6%)')

            dust.name = 'DSHARP'
            ice = copy.deepcopy(dust)
            sil = copy.deepcopy(dust)
            tro = copy.deepcopy(dust)
            org = copy.deepcopy(dust)
            ice.name = 'WaterIce'
            sil.name = 'Silicate'
            tro.name = 'Troilite'
            org.name = 'Organics'

            ice.set_nk(f'{pathnk}/h2o-w-Warren2008.lnk')
            sil.set_nk(f'{pathnk}/astrosil-Draine2003.lnk')
            tro.set_nk(f'{pathnk}/fes-Henning1996.lnk')
            org.set_nk(f'{pathnk}/c-org-Henning1996.lnk')
            [d.plot_nk(savefig=savefig) for d in [ice,sil,tro,org] if show_nk]

            # Set the individual material densities
            ice.set_density(0.92)
            sil.set_density(3.30)
            tro.set_density(4.83)
            org.set_density(1.50)

            # Set the individual mass fractions
            ice.set_mfrac(0.2000)
            sil.set_mfrac(0.3291)
            tro.set_mfrac(0.0743)
            org.set_mfrac(0.3966)

            # Brugemman Mixing of the optical constants
            dust.mix(
                comps=[ice, sil, tro, org], 
                rule=self.mixing, 
                porosity=self.porosity,
            )

            if show_nk:
                dust.plot_nk(savefig=savefig)

            # Convert the new mixed material to opacities
            dust.get_opacities(self.a_dist, self.q, self.nang, nth)

        elif self.material == 'diana':
            utils.not_implemented('Opacity: DIANA')
    
        else:
            if isinstance(self.material, str):
                # Use a single user provided nk table
                try:
                    dust.name = utils.basename(self.material)
                    utils.print_(f'Creating opacities for {dust.name}')
                    dust.set_nk(path=self.material, skip=1, get_dens=True)
                    self.material = dust.name
                    if show_nk:
                        dust.plot_nk(savefig=savefig)

                except Exception as e:
                    utils.print_(e, red=True)
                    raise ValueError(f'Material = {self.material} not found.')

            else:
                # Set several user provided nk tables
                material = [utils.basename(i) for i in self.material]
                utils.print_(
                    f'Creating opacities for {" and ".join(material)}')
                comps = []

                for i, m in enumerate(self.material):
                    d = copy.deepcopy(dust)
                    d.name = material[i]
                    d.set_nk(path=m, skip=1, get_dens=True)
                    d.set_mfrac(self.mfrac[i])
                    comps.append(d)
                    if show_nk: 
                        d.plot_nk(savefig=savefig)

                # Overwrite list of materials with a new name
                dust.name = 'NewMixture'
                self.material = dust.name

                dust.mix(
                    comps=comps,
                    rule=self.mixing,
                    porosity=self.porosity,
                )
                if show_nk: 
                    dust.plot_nk(savefig=savefig)

            # Convert the new material to opacities
            dust.get_opacities(self.a_dist, self.q, self.nang, nth)

        if show_z12z11:
            dust.plot_z12z11(self.lam, self.a_dist, self.nang)

        if show_opac or savefig is not None:
            dust.plot_opacities(show=show_opac, savefig=savefig)

        if show_dust_eff:
            dust.plot_efficiencies()

        dust.write_opacity_file(name=self._get_opac_name(self.csubl))

        if self.alignment:
            dust.write_align_factor(name=self._get_opac_name(self.csubl))

        # Store the current dust opacity in the pipeline instance
        self.kappa = dust._get_kappa_at_lam(self.lam) 

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
                f.write(f'mc_scat_maxtauabs = {int(5)}\n')
                f.write(f'scattering_mode = {self.scatmode}\n')
                if self.alignment and not mc: 
                    f.write(f'alignment_mode = 1\n')
                if not self.print_photons:
                    f.write(f'countwrite = {int(self.nphot)}\n')

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
    
            ofile = 'dustkappa' if not self.polarization else 'dustkapscatmat'

            try:
                # Fetch the corresponding opacity table from a public repo
                repo = 'https://raw.githubusercontent.com/jzamponi/'+\
                    f'utils/main/opacity_tables'
        
                if self.csubl > 0:
                    # Download opacity of the first species 
                    utils.download_file(
                        f'{repo}/{ofile}_{self._get_opac_name(self.csubl)}.inp')

                    # Download opacity of the second species. If dgrowth use 1mm
                    if self.dgrowth:
                        utils.download_file(
                            f'{repo}/{ofile}_' +\
                            f'{self._get_opac_name(dgrowth=self.dgrowth)}.inp')
                    else:
                        utils.download_file(f'{repo}/' +\
                            f'{ofile}_{self._get_opac_name(material2=True)}.inp')

                else:
                    # Download opacity for the single dust species 
                    utils.download_file(
                        f'{repo}/{ofile}_{self._get_opac_name()}.inp')

            # If unable to download from the repo, calculate it using dustmixer
            except Exception as e:
                utils.print_(
                    f'Unable to download opacity table. I will call ' +\
                    'dustmixer, as in synthesizer --opacity using '+\
                    'default values.', blue=True)
                
                self.dust = self.dustmixer()


        if dustkapalignfact:
            raise ImportError(f'{utils.color.red}There is no ' +\
                f'dustkapalignfact_*.inp file. Run synthesizer again with ' +\
                f'the option --opacity --alignment.{utils.color.none}')

        if grainalign:
            raise ImportError(f'{utils.color.red}There is no ' +\
                f'grainalign_dir.inp file. Run synthesizer again adding ' +\
                f'--vector-field to --grid to create the alignment field ' +\
                f'from the input model.{utils.color.none}')

        # Register the pipeline step 
        self.steps.append('generate_input_files')

    @utils.elapsed_time
    def monte_carlo(self, nphot, star=None, radmc3d_cmds=''):
        """ 
            Call radmc3d to calculate the radiative temperature distribution 
        """

        print('')
        utils.print_("Running a thermal Monte Carlo ...", bold=True)

        # Make sure RADMC3D is installed and callable
        utils.which('radmc3d', 
            msg="""You can easily install it with the following commands:
                    - git clone https://github.com/dullemond/radmc3d-2.0.git
                    - cd radmc3d-2.0/src
                    - make
                    - export PATH=$PWD:$PATH    
                    - cd ../../
                    - synthesizer --raytrace
                """) 

        self.nphot = nphot

        if star is not None:
            self.xstar = star[0] * u.au.to(u.cm)
            self.ystar = star[1] * u.au.to(u.cm)
            self.zstar = star[2] * u.au.to(u.cm)
            self.rstar = star[3] * u.Rsun.to(u.cm)
            self.mstar = star[4] * u.Msun.to(u.g)
            self.tstar = star[5]

        # Make sure there's at least a grid, density and temp. distribution
        utils.file_exists('amr_grid.inp', 
            msg='You must create a model grid first. Use synthesizer --grid')

        utils.file_exists('dust_density.inp', 
            msg='You must create a density model first. Use synthesizer --grid')

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
        if not utils.file_exists('dustka*.inp', raise_=False) or self.overwrite:
            self.generate_input_files(dustkappa=True)

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
        
        # Read in the new temperature 
        temp_mc = np.loadtxt('dust_temperature.dat', skiprows=3)
        nx = int(np.cbrt(temp_mc.size))
        temp_mc = temp_mc.reshape((nx, nx, nx))

        # Write the new temperature field to FITS file
        utils.write_fits(
            f'temperature-mc_midplane.fits', 
            data=temp_mc,
            header=fits.Header({
                'BTYPE': 'Dust Temperature',
                'BUNIT': 'K',
                'CDELT1': 2 * self.bbox / self.ncells,
                'CRPIX1': self.ncells // 2,
                'CRVAL1': 0,
                'CUNIT1': 'AU',
                'CDELT2': 2 * self.bbox / self.ncells,
                'CRPIX2': self.ncells // 2,
                'CRVAL2': 0,
                'CUNIT2': 'AU',
            }),
            overwrite=True,
            verbose=True,
        )
        
        # Free up memory
        del temp_mc

        # Register the pipeline step 
        self.steps.append('monte_carlo')

    @utils.elapsed_time
    def raytrace(self, lam=None, incl=None, phi=None, npix=None, sizeau=None, 
            distance=None, tau=False, tau_surf=None, show_tau_surf=False, 
            noscat=False, radmc3d_cmds='', cmap=None, stretch=None, show=False):
        """ 
            Call radmc3d to raytrace the newly created grid and plot an image 
        """

        print('')
        utils.print_("Ray-tracing the model density and temperature ...\n", 
            bold=True)

        # Make sure RADMC3D is installed and callable
        utils.which('radmc3d', 
            msg="""\n\nYou can easily install it with the following commands:
                    - git clone https://github.com/dullemond/radmc3d-2.0.git
                    - cd radmc3d-2.0/src
                    - make
                    - export PATH=$PWD:$PATH    
                    - cd ../../
                    - synthesizer --raytrace
                """) 

        # Make sure there's at least a grid, density and temp. distribution
        utils.file_exists('amr_grid.inp', 
            msg='You must create a model grid first. Use synthesizer --grid')

        utils.file_exists('dust_density.inp', 
            msg='You must create a density model first. Use synthesizer --grid')

        utils.file_exists('dust_temperature.dat',
            msg='You must create a temperature model first. '+\
                'Use synthesizer -g --temperature or synthesizer -mc')

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
        if not utils.file_exists('dustka*.inp', raise_=False) or self.overwrite:
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
        utils.file_exists('radmc3d.inp')
        utils.file_exists('wavelength_micron.inp')
        utils.file_exists('stars.inp')
        utils.file_exists('dustopac.inp')
        utils.file_exists('dustkapscat*' if self.polarization else 'dustkappa*')
        if self.alignment: 
            utils.file_exists('dustkapalignfact*')
            utils.file_exists('grainalign_dir.inp')

        self.distance = distance
        self.tau = tau
        self.tau_surf = tau_surf

        if lam is not None:
            self.lam = lam
        if npix is not None:
            self.npix = npix
        if incl is not None:
            self.incl = incl
        if phi is not None:
            self.phi = phi
        if sizeau is not None:
            self.sizeau = sizeau 
        else:
            self.sizeau = int(2 * self._get_bbox() * u.cm.to(u.au))

        # To do: Double check that this is correct. noscat does include 
        # k_sca in the extincion opacity but maybe scatmode=0 doesn't 
        if noscat: self.scatmode = 0

        # Set the RADMC3D command by concatenating options
        cmd = f'radmc3d image '
        cmd += f'lambda {self.lam} '
        cmd += f'sizeau {self.sizeau} '
        cmd += f'noscat ' if noscat else ' '
        cmd += f'stokes ' if self.polarization else ' '
        cmd += f'incl {self.incl} ' if self.incl is not None else ' '
        cmd += f'phi {self.phi} ' if self.phi is not None else ' '
        cmd += f'npix {self.npix} ' if self.npix is not None else ' '
        cmd += f'{" ".join(radmc3d_cmds)} '
        

        # Call RADMC3D and pipe the output also to radmc3d.out
        try:
            utils.print_(f'Executing command: {cmd}')
            self._radmc3d_banner()
            os.system(f'{cmd} 2>&1 | tee -a radmc3d.out')
            self._radmc3d_banner()

        except KeyboardInterrupt:
            raise Exception('Received SIGKILL. Execution halted by user.')

        self._catch_radmc3d_error()

        utils.print_(
            f'Dust opacity used: kappa({self.lam}um) = ' +\
            f'{self._get_opacity():.2} cm2/g', blue=True)

        # Generate FITS files from the image.out
        fitsfile = 'radmc3d_I.fits'
        utils.radmc3d_casafits(fitsfile, stokes='I', dpc=self.distance)

        # Write pipeline's info to the FITS headers
        for st in ['I', 'Q', 'U'] if self.polarization else ['I']:
            stfile = fitsfile.replace('I', st)
            utils.radmc3d_casafits(stfile, stokes=st, dpc=self.distance)
            utils.edit_header(stfile, 'NPHOT', f'{self.nphot:e}')
            utils.edit_header(stfile, 'OPACITY', self.kappa)
            utils.edit_header(stfile, 'MATERIAL', self.material)
            utils.edit_header(stfile, 'INCL', self.incl)
            utils.edit_header(stfile, 'PHI', self.phi)
            utils.edit_header(stfile, 'CSUBL', self.csubl)
            utils.edit_header(stfile, 'NSPEC', self.nspec)

        # Plot the new image in Jy/pixel
        if show:
            self.plot_rt()

        # Generate a 2D optical depth map using RADMC3D
        if self.tau:
            self.plot_tau(show)

        # Generate a 3D surface at tau = tau_surf
        if self.tau_surf is not None:
            try:
                utils.print_('Generating tau surface at tau = {self.tau_surf}')
                cmd.replace('image', f'tausurf {self.tau_surf}')
                cmd.replace('stokes', '')
                cmd.replace('noscat', '')
                os.system(f'{cmd}')

                os.rename('image.out', 'tauimage.out')
            except Exception as e:
                utils.print_(f'Unable to generate tau surface.\n{e}\n', red=True)

        # Render the 3D surface 
        if show_tau_surf:
            utils.not_implemented()
            from mayavi import mlab
            utils.file_exists('tausurface_3d.out')

        # Register the pipeline step 
        self.steps.append('raytrace')


    @utils.elapsed_time
    def synthetic_observation(self, show=False, cleanup=True, 
            script=None, simobserve=True, clean=True, exportfits=True, 
            obstime=1, resolution=None, obsmode='int', graphic=False, 
            use_template=False, telescope=None, verbose=False, 
            cmap=None, stretch=None,
            ):
        """ 
            Prepare the input for the CASA simulator from the RADMC3D output,
            and call CASA to run a synthetic observation.
        """

        print('')
        utils.print_('Running synthetic observation ...\n', bold=True)

        # Make sure RADMC3D is installed and callable
        utils.which('casa', msg='It is easy to install. ' +\
            'Go to https://casa.nrao.edu/casa_obtaining.shtml') 

        # If the observing wavelength is outside the working range of CASA, 
        # simplify the synthetic obs. to a PSF convolution and thermal noise
        if self.lam < 400 or self.lam > 4e6:

            utils.print_('Observing wavelength is outside mm/sub-mm range. ')
            utils.print_(
                'Limiting the observation to convolution and noise addition.')
    
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
            if script is None or script == '':

                script = synobs.CasaScript(self.lam)

                if use_template and self.lam not in [1300, 3000, 7000, 9000, 18000]: 
                    utils.print_(
                        f"There's no template available at {self.lam} um "+\
                        'I will create a simple one named casa_script.py.',
                        blue=True)

                    utils.print_('You can modify it later and re-run with ' +\
                        'synthesizer --synobs --script casa_script.py',
                        blue=True)

                    use_template = False

                if use_template:
                    utils.print_(
                        f'Using template casa script at {self.lam} microns')

                    template = {
                        1300: 'alma_1.3mm.py',
                        3000: 'alma_3mm.py',
                        7000: 'vla_7mm.py',
                        9000: 'vla_9mm.py',
                        18000: 'vla_18mm.py',
                    }[self.lam]

                    if self.polarization: 
                        template = template.replace('1.3mm', '1.3mm_pol')

                    script.name = str(source_dir/f'synobs/templates/{template}')
                    script.read(script.name)                    

                else:
                    # Create a minimal template CASA script
                    if self.npix is None: 
                        self.npix = fits.getheader(
                            'radmc3d_I.fits').get('NAXIS1')

                    # Create a map bigger than model to measure rms on borders
                    script.imsize = int(self.npix + 100)

                # Tailor the script to the pipeline setup
                script.resolution = resolution
                script.obsmode = obsmode
                script.telescope = telescope.lower()
                script.totaltime = f'{obstime}h'
                if script.resolution is not None: script.find_antennalist()
                script.polarization = self.polarization
                script.simobserve = simobserve
                script.clean = clean
                script.graphic = graphic if self.verbose else False
                script.overwrite = self.overwrite
                script.verbose = False
                script.write('casa_script.py')

            else:
                if 'http' in script: 
                    # Download the script if a URL is provided
                    utils.download_file(script)

                script_name = utils.basename(script)
                script = synobs.CasaScript(self.lam)
                script.read(script_name)

            self.script = script

            # Call CASA
            script.run()

            # Add the missing fits frequency keyword
            for s in ['I', 'Q', 'U'] if self.polarization else ['I']:
                utils.edit_header(f'synobs_{s}.fits', 'RESTFRQ', self.freq)

            # Clean-up and remove unnecessary files created by CASA 
            script.cleanup()

        # Show the new synthetic image
        if show:
            self.plot_synobs()

        # Register the pipeline step 
        self.steps.append('synthetic_observation')


    @utils.elapsed_time
    def plot_rt(self, distance=None, cmap=None, stretch=None):
        utils.print_('Plotting radmc3d_I.fits')

        # Override instance values in case it is called from parser.py
        if distance is not None:
            self.distance = distance 
        if cmap is not None:
            self.cmap = cmap 
        if stretch is not None:
            self.stretch = stretch 

        try:
            utils.file_exists('radmc3d_I.fits')

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
                    distance=self.distance, 
                    cmap=self.cmap,
                    stretch=self.stretch,
                )
            else:
                fig = utils.plot_map(
                    filename='radmc3d_I.fits', 
                    scalebar=50*u.au, 
                    distance=self.distance, 
                    bright_temp=False,
                    verbose=False,
                    stretch=self.stretch,
                    cmap=self.cmap,
                )
            fig.axis_labels.hide()
            fig.tick_labels.hide()
        except Exception as e:
            utils.print_(
                f'Unable to plot: {e}', bold=True)
        
        # Register the pipeline step
        self.steps.append('plot_rt')
 
    @utils.elapsed_time
    def plot_synobs(self, distance=None, cmap=None, stretch=None):
        utils.print_(f'Plotting the new synthetic image')

        # Override instance values in case it is called from parser.py
        if distance is not None:
            self.distance = distance 
        if cmap is not None:
            self.cmap = cmap 
        if stretch is not None:
            self.stretch = stretch 

        try:
            utils.file_exists('synobs_I.fits')
            utils.fix_header_axes('synobs_I.fits')

            utils.get_beam('synobs_I.fits', verbose=True)
                
            if self.polarization:
                utils.file_exists('synobs_Q.fits')
                utils.file_exists('synobs_U.fits')
                utils.fix_header_axes('synobs_Q.fits')
                utils.fix_header_axes('synobs_U.fits')

                fig = utils.polarization_map(
                    source='synobs', 
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
                    distance=self.distance,
                    cmap=self.cmap,
                    stretch=self.stretch,
                )
            else:
                fig = utils.plot_map(
                    filename='synobs_I.fits',
                    scalebar=50*u.au,
                    bright_temp=False,
                    verbose=False,
                    distance=self.distance,
                    cmap=self.cmap,
                    stretch=self.stretch,
                )
        except Exception as e:
            utils.print_(
                f'Unable to plot: {e}', bold=True)

        # Register the pipeline step
        self.steps.append('plot_synobs')

    @utils.elapsed_time
    def plot_tau(self, show=False, cmap=None, stretch=None):
        """ Calls RADMC3D with the tau mode to calculate an op. depth image """

        from matplotlib.colors import LogNorm

        if cmap is not None:
            self.cmap = cmap 
        if stretch is not None:
            self.stretch = stretch 

        if not os.path.exists('tau.fits') or \
            (os.path.exists('tau.fits') and self.overwrite):

            utils.print_(
                f'Generating optical depth map at {self.lam} microns ' +\
                f'using RADMC3D')

            if os.path.exists(filename:='image.out'):
                utils.print_(
                    'Backing up existing image.out to image.out_backup')
                os.system(f'cp {filename} {filename}_backup ')

            # Call RADMC3D with the tau mode
            cmd = f'radmc3d image tracetau '
            cmd += f'lambda {self.lam} '
            cmd += f'sizeau {self.sizeau} '
            cmd += f'npix {self.npix} '
            cmd += f'incl {self.incl} ' if self.incl is not None else ''
            cmd += f'phi {self.phi}' if self.phi is not None else ''

            utils.print_(f'Executing command: {cmd}')
            os.system(f'{cmd} > /dev/null')

            # Recover the original image.out 
            if utils.file_exists(filename, raise_=False):
                utils.print_(f'Restoring backup to {filename}')
                os.system(f'mv {filename} tau.out')
                os.system(f'cp {filename}_backup {filename} ')

            # Write the tau image to fits
            utils.radmc3d_casafits(
                tau=True,
                fitsfile='tau.fits',
                radmc3dimage='tau.out',
                dpc=self.distance, 
            )

        else:
            utils.print_('Reading from existing tau.fits')

        tau_map, tau_hdr = fits.getdata('tau.fits', header=True)

        if show:
            # Set the colormap normalization from the stretch parameter
            if self.stretch == 'log':
                norm = LogNorm(vmin=None, vmax=None)
            elif self.stretch == 'asinh':

                try:
                    from matplotlib.colors import AsinhNorm
                    norm = AsinhNorm(linear_width=1, 
                            vmin=tau_map.min(), vmax=tau_map.max())
                except ImportError:
                    utils.print_('Current version of Matplotlib does not ' +
                        'support asinh stretch. Consider upgrading to >=3.8.0')
                    norm = None
            else:
                norm = None

            # Find the size of the box to set the spatial scale 
            if self.bbox is None:
                bbox = tau_hdr['CDELT1'] * tau_hdr['NAXIS1'] / 2
                
            else:
                bbox = self.bbox
                if 'create_grid' in self.steps:
                    bbox *= u.cm.to(u.au)

            # Customize the figure
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.top'] = True
            plt.rcParams['ytick.right'] = True
            plt.rcParams['xtick.minor.visible'] = True
            plt.rcParams['ytick.minor.visible'] = True

            plt.title(fr'Optical depth at $\lambda = ${self.lam}$\mu$m')
            plt.imshow(
                tau_map, 
                origin='lower', 
                cmap=self.cmap, 
                norm=norm,
                extent = [-bbox, bbox] * 2
            )
            plt.xlabel('X (AU)')
            plt.ylabel('Y (AU)')
            plt.colorbar()
            plt.show()

        # Register the pipeline step
        self.steps.append('plot_tau')

    @utils.elapsed_time
    def plot_grid_2d(self, temp=False, cmap=None):
        """ 
            Plot the grid's density and temperature midplanes from files,
            in case they are not currently available from pipeline.grid
        """

        self.cmap = cmap if cmap is not None else 'BuPu'

        utils.file_exists('dust_density.inp')
        utils.print_('Reading density from dust_density.inp')
        dens = np.loadtxt('dust_density.inp').T
        ncells = int(dens[1])
        nspec = int(dens[2])
        if nspec == 1:
            dens = dens[3:]
        else: 
            utils.print_('Found two dust species, I plot only the first one')
            dens = dens[3: ncells+3]
        nx = int(np.cbrt(dens.size))
        dens = dens.reshape((nx, nx, nx))
        bbox = self._get_bbox()
        grid = gridder.CartesianGrid(nx, bbox)
        grid.plot_2d('density', data=dens, cmap=self.cmap)

        if temp: 
            utils.file_exists('dust_temperature.dat')
            utils.print_('Reading temperature from dust_temperature.dat')
            temp = np.loadtxt('dust_temperature.dat', skiprows=3)
            temp = temp.reshape((nx, nx, nx))
            grid = gridder.CartesianGrid(nx, bbox)
            grid.plot_2d('temperature', data=temp, cmap=self.cmap)
            del temp

        # Register the pipeline step
        self.steps.append('plot_grid_2d')
        
    @utils.elapsed_time
    def plot_grid_3d(self, temp=False, cmap=None):
        """ 
            Render the grid's density and temperature in 3D from files,
            in case they are not currently available from pipeline.grid
        """

        self.cmap = cmap if cmap is not None else 'inferno'

        utils.file_exists('dust_density.inp')
        utils.print_('Reading density from dust_density.inp')
        dens = np.loadtxt('dust_density.inp').T
        ncells = int(dens[1])
        nspec = int(dens[2])
        if nspec == 1:
            dens = dens[3:]
        else: 
            utils.print_('Found two dust species, I plot only the first one')
            dens = dens[3: ncells+3]
        nx = int(np.cbrt(dens.size))
        dens = dens.reshape((nx, nx, nx))
        bbox = self._get_bbox()
        utils.print_(f'Rendering a box of {nx}^3 pixels')
        grid = gridder.CartesianGrid(nx, bbox)
        grid.plot_3d('density', data=dens, cmap=self.cmap)

        if temp: 
            utils.file_exists('dust_temperature.dat')
            utils.print_('Reading temperature from dust_temperature.dat')
            temp = np.loadtxt('dust_temperature.dat', skiprows=3)
            temp = temp.reshape((nx, nx, nx))
            grid = gridder.CartesianGrid(nx, bbox)
            grid.plot_3d('temperature', data=temp, cmap=self.cmap)

        # Register the pipeline step
        self.steps.append('plot_grid_3d')

    @utils.elapsed_time
    def plot_nk(self):
        """ Plot optical constants from the .lnk tables """
        utils.file_exists('*.lnk')
        filename = utils.latest_file('*.lnk')
        dust = dustmixer.Dust()
        dust.set_nk(filename, skip=1, get_dens=True)
        dust.plot_nk()

        # Register the pipeline step
        self.steps.append('plot_nk')

    @utils.elapsed_time
    def plot_opacities(self):
        """ Plot opacities from file. Uses the lastly created dustkap* file """
        utils.file_exists('dustkap*.inp')
        filename = utils.latest_file('dustkap*.inp')
        utils.print_(f'Reading from the most recent opacity table: {filename}')
        
        header = np.loadtxt(filename, max_rows=2)
        iformat = int(header[0])
        nlam = int(header[1])
        skip = 3 if 'kapscat' in filename else 2
        d = ascii.read(filename, data_start=skip, data_end=nlam + skip)
        l = d['col1']
        k_abs = d['col2']
        k_sca = d['col3']
        k_ext = k_abs + k_sca

        fig = plt.figure()

        plt.loglog(l, k_sca, ls=':', c='black', label=r'$\kappa_{\rm sca}$')
        plt.loglog(l, k_abs, ls='--', c='black', label=r'$\kappa_{\rm abs}$')
        plt.loglog(l, k_ext, ls='-', c='black', label=r'$\kappa_{\rm ext}$')
        plt.legend()
        plt.xlabel('Wavelength (microns)')
        plt.ylabel(r'Dust opacity $\kappa$ (cm$^2$ g$^{-1}$)')
        plt.ylim(1e-2, 1e4)
        plt.xlim(1e-1, 3e4)
        plt.tight_layout()
        plt.show()

        # Register the pipeline step
        self.steps.append('plot_opacities')

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

        prefix = 'dustkapscatmat_' if self.polarization else 'dustkappa_'
        self.opacfile = prefix + opacname + '.inp'

        return opacname 
    
    def _get_opacity(self):
        """ Read in an opacity file, interpolate and find the opacity at lambda """
        from scipy.interpolate import interp1d

        # Supress a Numpy (>=1.22) warning from loadtxt
        warnings.filterwarnings('ignore')

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
            self.k_sca = d['col3']
            self.k_ext = self.k_abs + self.k_sca
            self.kappa = float(interp1d(l, self.k_ext)(self.lam))

            return self.kappa

        except Exception as e:
            utils.print_(
                f"{e} I couldn't obtain the opacity from {self.opacfile}. " +\
                "I will assume k_ext = 1 g/cm3.")
            self.kappa = 1.0

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

        if self.bbox is not None:
            return self.bbox
        else:
            gridid = int(np.loadtxt('amr_grid.inp', skiprows=1, max_rows=1))
            ncells = np.loadtxt('amr_grid.inp', skiprows=5, max_rows=1)
            nx, ny, nz = int(ncells[0]), int(ncells[1]), int(ncells[2])

            # Number of lines to skip from grid style (0: reg, 1: oct, 10: amr)
            skip = {0: 6, 1: 7, 10: 7}[gridid]

            # Number of lines to read
            nlines = nx * ny * nz + 3 

            # Read in the grid
            g = np.loadtxt('amr_grid.inp', skiprows=skip, max_rows=nlines)

            # Return bbox as the difference between the first and last vertex
            return (g[-1] - g[0]) / 2

