import os
import sys
import copy
import numpy as np
from astropy.io import fits, ascii

from synthesizer import utils

class RADMC3D:
    def __init__(self, mode='image'):

        self.mode = mode
        self.lam = None
        self.npix = None
        self.incl = None
        self.sizeau = None
        self.noscat = None
        self.stokes = None
        self.alignment = False
        self.tau = None
        self.tau_surf = None
        self.cmd = ''
        self.logfile = 'radmc3d.out'
        self.imgfile = 'image.out'
        self.fitsfile = 'radmc3d_I.fits'

    def set_command(self, lam, sizeau, incl, npix, noscat, stokes, extra):
        """ Set the RADMC3D command string by concatenating options """
        self.lam = lam
        self.sizeau = sizeau
        self.incl = incl,
        self.npix = npix
        self.noscat = noscat
        self.stokes = stokes
        self.cmd += f'radmc3d {self.mode} '
        self.cmd += f'lambda {lam} '
        self.cmd += f'sizeau {sizeau} '
        self.cmd += f'noscat ' if noscat else ' '
        self.cmd += f'stokes ' if stokes else ' '
        self.cmd += f'incl {incl} ' if incl is not None else ' '
        self.cmd += f'npix {npix} ' if npix is not None else ' '
        self.cmd += f'{" ".join(extra)} '

    def run(self):
        utils.print_(f'Executing command: {self.cmd}')
        self._banner()

        try:
            os.system(f'{self.cmd} 2>&1 | tee -a {self.logfile}')

        except KeyboardInterrupt:
            raise KeyboardInterrupt('Received SIGKILL. Execution halted by user.')

        self._banner()
        
    def _banner(self):
        print(
            f'{utils.color.blue}{"="*31}  RADMC3D  {"="*31}{utils.color.none}')

    def tau_surf(self, tau):
        try:
            utils.print_('Generating tau surface at tau = {tau}')
            self.tau_surf = tau
            self.cmd.replace('image', f'tausurf {tau}')
            self.cmd.replace('stokes', '')
            self.cmd.replace('noscat', '')
            self.run()

            os.rename('image.out', 'tauimage.out')
        except Exception as e:
            utils.print_(f'Unable to generate tau surface.\n{e}\n', red=True)

    def check_inputs(self):
        utils.file_exists('radmc3d.inp')
        utils.file_exists('wavelength_micron.inp')
        utils.file_exists('stars.inp')
        utils.file_exists('dustopac.inp')
        utils.file_exists('dustkapscat*' if self.stokes else 'dustkappa*')
        if self.alignment: 
            utils.file_exists('dustkapalignfact*')
            utils.file_exists('grainalign_dir.inp')

    def catch_error(self):
        """ Raise an exception to halt synthesizer if RADMC3D ended in Error """

        # Read radmc3d.out and stop the pipeline if RADMC3D finished in error
        utils.file_exists(self.logfile)
        with open (self.logfile, 'r') as out:
            for line in out.readlines():
                line = line.lower()
                if 'error' in line or 'stop' in line:

                    msg = lambda m: f'{utils.color.red}\r{m}{utils.color.none}'
                    errmsg = msg(f'[RADMC3D] {line}...')

                    if 'g=<cos(theta)>' in line or 'the scat' in line:
                        errmsg += f' Try increasing --na or --nang '
                        errmsg += f'(currently, na: {self.na} nang {self.nang})'

                    raise Exception(errmsg)

    def write_output_fits(self, fitsfile='radmc3d_I.fits'):
        """ Generate FITS files from the image.out """

        utils.radmc3d_casafits(fitsfile, stokes='I', dpc=self.distance)

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

    def check_input_grid(self, temp=False):
        utils.file_exists('amr_grid.inp', 
            msg='You must create a model grid first. Use synthesizer --grid')

        utils.file_exists('dust_density.inp', 
            msg='You must create a density model first. Use synthesizer --grid')

        if temp:
            utils.file_exists('dust_temperature.dat',
                msg='You must create a temperature model first. '+\
                    'Use synthesizer -g --temperature or synthesizer -mc')

    def check_input_files(self):
        utils.file_exists('radmc3d.inp')
        utils.file_exists('wavelength_micron.inp')
        utils.file_exists('stars.inp')
        utils.file_exists('dustopac.inp')
        utils.file_exists('dustkapscat*' if self.stokes else 'dustkappa*')
        if self.alignment: 
            utils.file_exists('dustkapalignfact*')
            utils.file_exists('grainalign_dir.inp')

    def installer(self):
        shell = os.environ('SHELL')

        if 'bash' in shell:
            shfile = '.bashrc'
        elif 'zsh' in shell:
            shfile = '.bash_profile'
        elif 'tcshell' in shell:
            shfile = '.profile'
    
        utils.print_(f"""
            You can easily install it with the following commands:
                - git clone https://github.com/dullemond/radmc3d-2.0.git
                - cd radmc3d-2.0/src
                - make
                - echo "export PATH=$PWD:$PATH" >> ~/{shfile}
                - source ~/.{shfile}
                - cd ../../
                - synthesizer --raytrace ...
            """, blue=True)
        install = input('Should I install it and run for you? (yes/no)')

        if install in ['', 'yes']:
            os.system('git clone https://github.com/dullemond/radmc3d-2.0.git')
            os.system('cd radmc3d-2.0/src')
            os.system('make')
            os.system('export PATH=$PWD:$PATH')
            os.system(f'source ~/{shfile}')
            os.system('cd ../../')


    def prepare(self):
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
            self.k_sca = d['col3'] if iformat > 1 else np.zeros(l.shape)
            self.k_ext = self.k_abs + self.k_sca
            self.kappa = float(interp1d(l, self.k_ext)(self.lam))

            return self.kappa

        except Exception as e:
            utils.print_(
                f"{e} I couldn't obtain the opacity from {self.opacfile}. " +\
                "I will assume k_ext = 1 g/cm3.")
            self.kappa = 1.0

            return 1.0
