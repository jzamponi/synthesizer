""" 
    This module is meant to create minimal CASA script templates for
    simulating observations.
    For more customized scripts, please consider modifying a template 
    and provide it to synthesizer via command-line as
    $ synthesizer --synobs --script my_casa_script.py
"""

import os
import re
import random
import subprocess
import numpy as np
import astropy.units as u
import astropy.constants as c

from synthesizer import utils

class CasaScript():
    
    def __init__(self, lam, name='casa_script.py'):
        """ Set default values for a template CASA script. """
        self.name = name
        self.lam = lam
        self.freq = c.c.cgs.value / (lam * u.micron.to(u.cm))
        self.simobserve = True
        self.clean = True
        self.exportfits = True
        self.graphics = 'both'
        self.overwrite = True
        self.polarization = True

        # simobserve
        self.project = 'synobs_data'
        self.skymodel = lambda st: f'radmc3d_{st}.fits'
        self.fitsimage = lambda st: f'synobs_{st}.fits'
        self.inbright = ''
        self.incell = ''
        self.mapsize = ''
        self.resolution = None
        self.pix_per_beam = 5
        self.incenter = f'{self.freq}Hz'
        self.inwidth = '2GHz'
        self.setpointings = True
        self.integration = '2s'
        self.totaltime = '1h'
        self.indirection = 'J2000 16h32m22.63 -24d28m31.8'
        self.hourangle = 'transit'
        self.refdate = '2017/05/20' 
        self.obsmode = 'int'
        self.telescope = None
        self.antennalist = self._get_antenna_array(cycle=4, arr=7)
        self.thermalnoise = 'tsys-manual'
        self.seed = lambda: int(random.randint(0, 1000))

        # tclean
        self.vis = f'{self.project}/{self.project}.{self.arrayfile}.noisy.ms'
        self.imagename = lambda s: f'{self.project}/clean_{s}'
        self.imsize = 100
        if self.resolution is None:
            self.cell = '0.008arcsec'
        else:
            self.cell = f'{self.resolution / self.pix_per_beam}arcsec'
        self.reffreq = self.incenter
        self.specmode = 'mfs'
        self.gridder = 'standard'
        self.deconvolver = 'multiscale'
        self.scales = [1, 8, 24]
        self.weighting = 'briggs'
        self.robust = 0.5
        self.niter = 1e4
        self.threshold = '5e-5Jy'
        self.mask = ''
        self.uvrange = ''
        self.uvtaper = ''
        self.pbcor = True
        self.interactive = False
        self.verbose = False

        # exportfits
        self.dropstokes = True
        self.dropdeg = True

    def _find_telescope(self):
        """ Find a proper telescope given the observing wavelength (microns) """
        if self.lam > 400 or self.lam < 4500:
            self.telescope = 'alma'

        elif self.lam >= 4500 and self.lam < 4e6:
            self.telescope = 'vla'

        else:
            utils.not_implemented('Simulations for telescopes operating '+\
                'outside the sub-/milimeter wavelengths are currently not '+\
                'implemented. But they will.')

    def find_antennalist(self):
        """ Get the best antenna array that matches a desired angular resolution
            by interpolating the tabulated resolutions to the one requested.
        """

        from scipy.interpolate import interp1d

        obs = self.telescope
        res = self.resolution

        if obs == 'aca':
            self.antennalist = 'aca.cycle7.cfg'
            self.arrayfile = self.antennalist.strip('.cfg') 
            return
        
        elif obs == 'atca':
            self.antennalist = 'atca_all.cfg'
            self.arrayfile = self.antennalist.strip('.cfg') 
            return 

        elif obs == 'vlba':
            self.antennalist = 'vlba.cfg'
            self.arrayfile = self.antennalist.strip('.cfg') 
            return

        elif obs == 'meerkat':
            self.antennalist = 'meerkat.cfg'
            self.arrayfile = self.antennalist.strip('.cfg') 
            return 

        # List synthesized beams for diff. telescopes as matrices (Band, Array)
        beams = {
            # src: ALMA Technical Handbook Cycle 9
            'alma': np.array([
                [3.38, 2.30, 1.42, 0.92, 0.550, 0.310, 0.210, 0.096, 0.057, 0.042],
                [2.25, 1.53, 0.94, 0.61, 0.360, 0.200, 0.140, 0.064, 0.038, 0.028],
                [1.83, 1.24, 0.77, 0.50, 0.300, 0.170, 0.110, 0.052, 0.031, 0.023],
                [1.47, 1.00, 0.62, 0.40, 0.240, 0.130, 0.092, 0.042, 0.025, 0.018],
                [0.98, 0.67, 0.41, 0.27, 0.160, 0.089, 0.061, 0.028, 0.017, 0.012],
                [0.74, 0.50, 0.31, 0.20, 0.120, 0.067, 0.046, 0.021, 0.012, 0.009],
                [0.52, 0.35, 0.22, 0.14, 0.084, 0.047, 0.033, 0.015, 0.0088, np.NaN],
                [0.39, 0.26, 0.16, 0.11, 0.063, 0.035, 0.024, 0.011, np.NaN, np.NaN],
                ]), 
            # science.nrao.edu/facilities/vla/docs/manuals/oss/performance/resolution
            'vla': np.array([
                [24.0, 80.0, 260., 850],
                [5.60, 18.5, 60.0, 200],
                [1.30, 4.30, 14.0, 46.],
                [0.65, 2.10, 7.00, 23.],
                [0.33, 1.00, 3.50, 12.],
                [0.20, 0.60, 2.10, 7.2],
                [0.13, 0.42, 1.40, 4.6],
                [0.09, 0.28, 0.95, 3.1],
                [0.06, 0.19, 0.63, 2.1],
                [0.04, 0.14, 0.47, 1.5],
                ]),
            # src: http://sma1.sma.hawaii.edu/status.html#arrayconf
            'sma': np.array([
                [9, 3.5, 1.3, 0.5],
                [6, 2.5, 0.8, 0.3], 
            ]),
            # src: https://www.iram.fr/IRAMFR/GILDAS/doc/pdf/noema-intro.pdf
            'noema': np.array([
                [np.NaN, np.NaN], 
                [np.NaN, np.NaN], 
                [np.NaN, np.NaN], 
                [np.NaN, np.NaN], 
            ]),
        }
        # To do: the band grids should have N_band+1 items, (more like band 
        # grid walls) to capture values within the upper and lower band edges.

        # List wavelength bands in microns
        w = lambda nu: (c.c.cgs.value / (nu*u.GHz.to(u.Hz)))*u.cm.to(u.micron)
        bands = {
            'alma': np.array(
                [w(i) for i in [83.3, 115, 167, 214, 273, 375, 500, 750, 999]]),
            'vla': np.array(
                [w(i) for i in [0.074, 0.35, 1.5, 3, 6, 10, 15, 22, 33, 45]]),
            'sma': np.array([w(230), w(345)]),
            'noema': np.array([w(70.384), w(127), w(196), w(276)]), 
        }

        # Returns True if the desired resolution is within the listed syn. beams
        in_beams = lambda x: \
            res > np.nanmin(beams[x]) and res < np.nanmax(beams[x])

        # Raise error if the observing wavelength is outside the receiver bands
        if self.lam < bands[obs].min() or self.lam > bands[obs].max():

            raise ValueError(
                f'{utils.color.red}' +
                f'Observing wavelength {self.lam} µm is ' +
                f'outside the {self.telescope.upper()} receiver bands.' +
                f'{utils.color.none}'
            )

        # Store the number of antenna arrays defined in beams. 
        nband = bands[obs].shape[0]
        narrs = beams[obs].shape[1]
        
        # Find the matrix ID of the array that delivers the desired resolution
        bandid = int(interp1d(bands[obs], range(nband))(self.lam))

        try:
            arrid = int(interp1d(beams[obs][bandid], range(narrs))(res))

        except:
            raise ValueError(
                f'{utils.color.red}' +
                f'Requested resolution is not delivered by {obs} ' +
                f'at {self.lam} µm.\nYou can force it by manually setting ' +
                f'antennalist in your new casa_script.py {utils.color.none}'
            )

        if obs == 'alma':
            self.antennalist = f'alma.cycle7.{arrid + 1}.cfg'
            self.arrayfile = self.antennalist.strip('.cfg') 
            return 

        elif obs == 'vla':
            arr = ['a', 'b', 'c', 'd'][arrid]
            self.antennalist = f'vla.{arr}.cfg'
            self.arrayfile = self.antennalist.strip('.cfg') 
            return 

        elif obs == 'noema':
            arr = ['a', 'b', 'c', 'd'][arrid]
            self.antennalist = f'pdbi-{arr}.cfg'
            self.arrayfile = self.antennalist.strip('.cfg') 
            return 

        elif obs == 'sma':
            arr = ['subcompact', 'compact', 'extended', 'vextended'][arrid]
            self.antennalist = f'sma.{arr}.cfg'
            self.arrayfile = self.antennalist.strip('.cfg') 
            return 


    def _get_antenna_array(self, cycle, arr):
        """ Set the antennalist string for a given antenna configuration.
            All possible config files are found in CASA_PATH/data/alma/simmos/
        """
        if self.telescope is None:
            self._find_telescope()
        self.cycle = str(cycle).lower()
        self.arr = str(arr).lower()
        self.arrayfile = f'{self.telescope}.cycle{self.cycle}.{self.arr}'
        return self.arrayfile + '.cfg'

    def write(self, name):
        """ Write a CASA script using the CasaScript parameters """
        
        self.name = name

        stokes = ['I', 'Q', 'U'] if self.polarization else ['I']
    
        with open(self.name, 'w+') as f:
            utils.print_(f'Writing template script: {self.name}', blue=True)
            f.write('# Template CASA script to simulate observations. \n')
            f.write('# Written by the Synthesizer. \n\n')

            for s in stokes:

                # Overwrite string values if they were overriden from self.read()
                if isinstance(self.skymodel, str):
                    self.skymodel = f'radmc3d_{s}.fits'
                else:
                    self.skymodel = self.skymodel(s)

                if isinstance(self.imagename, str):
                    self.imagename = f'synobs_data/clean_{s}.fits'
                else:
                    self.imagename = self.imagename(s)

                if isinstance(self.fitsimage, str):
                    self.fitsimage = f'synobs_{s}.fits'
                else:
                    self.fitsimage = self.fitsimage(s)

                # Reset, to keep track in case arrayfile was updated
                self.vis = \
                    f'{self.project}/{self.project}.{self.arrayfile}.noisy.ms'

                self.mask = self.vis.replace('noisy.ms', 'skymodel')
    
                if self.simobserve:
                    f.write(f'print("\033[1m\\n[syn_obs] ')
                    f.write(f'Observing Stokes {s} ...\033[0m\\n")\n')
                    f.write(f'simobserve( \n')
                    f.write(f'    project = "{self.project}", \n')
                    f.write(f'    skymodel = "{self.skymodel}", \n')
                    f.write(f'    inbright = "{self.inbright}", \n')
                    f.write(f'    incell = "{self.incell}", \n')
                    f.write(f'    incenter = "{self.incenter}", \n')
                    f.write(f'    inwidth = "{self.inwidth}", \n')
                    f.write(f'    mapsize = "{self.mapsize}", \n')
                    f.write(f'    setpointings = {self.setpointings}, \n')
                    f.write(f'    indirection = "{self.indirection}", \n')
                    f.write(f'    integration = "{self.integration}", \n')
                    f.write(f'    totaltime = "{self.totaltime}", \n')
                    f.write(f'    hourangle = "{self.hourangle}", \n')
                    f.write(f'    obsmode = "{self.obsmode}", \n')
                    f.write(f'    refdate = "{self.refdate}", \n')
                    f.write(f'    antennalist = "{self.antennalist}", \n')
                    f.write(f'    thermalnoise = "{self.thermalnoise}", \n')
                    f.write(f'    seed = {self.seed()}, \n')
                    f.write(f'    graphics = "{self.graphics}", \n')
                    f.write(f'    overwrite = {self.overwrite}, \n')
                    f.write(f'    verbose = {self.verbose}, \n')
                    f.write(f') \n')
            
                if self.clean:
                    f.write(f'print("\033[1m\\n[syn_obs] ')
                    f.write(f'Cleaning Stokes {s} ...\033[0m\\n")\n')
                    f.write(f' \n')
                    f.write(f'tclean( \n')
                    f.write(f'    vis = "{self.vis}", \n')
                    f.write(f'    imagename = "{self.imagename}", \n')
                    f.write(f'    imsize = {self.imsize}, \n')
                    f.write(f'    cell = "{self.cell}", \n')
                    f.write(f'    reffreq = "{self.reffreq}", \n')
                    f.write(f'    specmode = "{self.specmode}", \n')
                    f.write(f'    gridder = "{self.gridder}", \n')
                    f.write(f'    deconvolver = "{self.deconvolver}", \n')
                    f.write(f'    scales = {self.scales}, \n')
                    f.write(f'    weighting = "{self.weighting}", \n')
                    f.write(f'    robust = {self.robust}, \n')
                    f.write(f'    niter = {int(self.niter)}, \n')
                    f.write(f'    threshold = "{self.threshold}", \n')
                    f.write(f'    uvrange = "{self.uvrange}", \n')
                    f.write(f'    uvtaper = "{self.uvtaper}", \n')
                    f.write(f'    mask = "{self.mask}", \n')
                    f.write(f'    pbcor = {self.pbcor}, \n')
                    f.write(f'    interactive = {self.interactive}, \n')
                    f.write(f'    verbose = {self.verbose}, \n')
                    f.write(f') \n')

                    f.write(f'imregrid( \n')
                    f.write(f'    "{self.imagename}.image", \n')
                    f.write(f'    template = "{self.mask}", \n')
                    f.write(f'    output = "{self.imagename}.image_modelsize", \n')
                    f.write(f'    overwrite = True, \n')
                    f.write(f') \n\n')

                if self.exportfits:
                    f.write(f'print("\033[1m\\n[syn_obs] ')
                    f.write(f'Exporting Stokes {s} ...\033[0m\\n")\n')
                    f.write(f' \n')
                    f.write(f'exportfits( \n')
                    f.write(f'    imagename = "{self.imagename}.image_modelsize", \n')
                    f.write(f'    fitsimage = "{self.fitsimage}", \n')
                    f.write(f'    dropstokes = {self.dropstokes}, \n')
                    f.write(f'    dropdeg = {self.dropdeg}, \n')
                    f.write(f'    overwrite = True, \n')
                    f.write(f') \n\n')
                

    def read(self, name):
        """ Read variables and parameters from an already existing file """

        # Raise an error if file doesn't exist, including wildcards
        utils.file_exists(name)

        self.name = name

        def strip_line(l):
            l = l.split('=')[1]
            l = l.strip('\n')
            l = l.strip(',')
            l = l.strip()
            l = l.strip(',')
            l = l.strip('"')
            l = l.strip("'")
            if isinstance(l, (list, tuple)): l = l[0]
            if ',' in l and not '[' in l: l = l.split(',')[0]
            return l

        f = open(str(name), 'r')

        for line in f.readlines():
            # Main boolean switches
            if 'Simobserve' in line and '=' in line: self.simobserve = strip_line(line)
            if 'Clean' in line and '=' in line: self.clean = strip_line(line)
            if 'polarization' in line and '=' in line: self.polarization = strip_line(line)
        
            # Simobserve
            if 'project' in line: self.project = strip_line(line)
            if 'skymodel ' in line and '=' in line: self.skymodel = strip_line(line)
            if 'inbright' in line: self.inbright = strip_line(line)
            if 'incell' in line: self.incell = strip_line(line)
            if 'mapsize' in line: self.mapsize = strip_line(line)
            if 'incenter' in line: self.incenter = strip_line(line)
            if 'inwidth' in line: self.inwidth = strip_line(line)
            if 'setpointings' in line: self.setpointings = strip_line(line)
            if 'integration' in line: self.integration = strip_line(line)
            if 'totaltime' in line: self.totaltime = strip_line(line)
            if 'indirection' in line: self.indirection = strip_line(line)
            if 'refdate' in line: self.refdate = strip_line(line)
            if 'hourangle' in line: self.hourangle = strip_line(line)
            if 'obsmode' in line: self.obsmode = strip_line(line)
            if 'antennalist' in line: self.antennalist = strip_line(line)
            if 'thermalnoise' in line: self.thermalnoise = strip_line(line)
            if 'graphics' in line: self.graphics = strip_line(line)
            if 'overwrite' in line: self.overwrite = strip_line(line)
            if 'verbose' in line: self.verbose = strip_line(line)
            self.arrayfile = self.antennalist.strip('.cfg')
    
            # tclean
            if 'vis' in line: self.vis = strip_line(line)
            if 'imagename' in line: self.imagename = strip_line(line)
            if 'imsize' in line: self.imsize = strip_line(line)
            if 'cell' in line: self.cell = strip_line(line)
            if 'reffreq' in line: self.reffreq = strip_line(line)
            if 'restfrq' in line: self.restfrq = strip_line(line)
            if 'specmode' in line: self.specmode = strip_line(line)
            if 'gridder' in line: self.gridder = strip_line(line)
            if 'deconvolver' in line: self.deconvolver = strip_line(line)
            if 'scales' in line: self.scales = strip_line(line)
            if 'weighting' in line: self.weighting = strip_line(line)
            if 'robust' in line: self.robust = strip_line(line)
            if 'niter' in line: self.niter = strip_line(line)
            if 'threshold' in line: self.threshold = strip_line(line)
            if 'uvrange' in line: self.uvrange = strip_line(line)
            if 'uvtaper' in line: self.uvtaper = strip_line(line)
            if 'mask' in line: self.mask = strip_line(line)
            if 'pbcor' in line: self.pbcor = strip_line(line)
            if 'interactive' in line: self.interactive = strip_line(line)

            # Exportfits
            if 'fitsimage' in line: self.fitsimage = strip_line(line)
            if 'dropstokes' in line: self.dropstokes = strip_line(line)
            if 'dropdeg' in line: self.dropdeg = strip_line(line)

        f.close()

    def _clean_project(self):
        """ Delete any previous project to avoid the CASA clashing """

        if self.overwrite and os.path.exists('synobs_data'):
            if not self.simobserve: 
                utils.print_(f'Deleting previous cleaning output.')
                subprocess.run('rm -r synobs_data/clean_I*', shell=True)

            if self.simobserve and self.clean:
                utils.print_(f'Deleting previous project: {self.project}.')
                subprocess.run('rm -r synobs_data', shell=True)
                

    def run(self):
        """ Run the ALMA/JVLA simulation script """

        self._clean_project()
        subprocess.run(f'casa -c {self.name} --nologger'.split())

    def cleanup(self):
        if utils.file_exists('casa-*.log', raise_=False) or\
                utils.file_exists('*.last', raise_=False):

            utils.print_('Cleaning up ... deleting casa-*.log and *.last files')
            subprocess.run('rm casa-*.log *.last', shell=True)
