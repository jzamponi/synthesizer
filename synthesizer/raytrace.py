import os
import sys
import copy
import numpy as np
from astropy.io import fits, ascii

from synthesizer import utils

class RADMC3D:
    def __init__(self, mode='image'):

        self.mode = mode
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

    def command(self, cmd):
        self.cmd = cmd

    def run(self):
        utils.print_(f'Executing command: {cmd}')
        self._banner()

        try:
            os.system(f'{cmd} 2>&1 | tee -a {logfile}')

        except KeyboardInterrupt:
            raise Exception('Received SIGKILL. Execution halted by user.')

        self._banner()
        
    def _banner(self):
        print(
            f'{utils.color.blue}{"="*31}  RADMC3D  {"="*31}{utils.color.none}')

    def catch_error(self):
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
        utils.file_exists('dustkapscat*' if self.polarization else 'dustkappa*')
        if self.alignment: 
            utils.file_exists('dustkapalignfact*')
            utils.file_exists('grainalign_dir.inp')

    def installer_help(self):
        utils.print_("""
            You can easily install it with the following commands:
                - git clone https://github.com/dullemond/radmc3d-2.0.git
                - cd radmc3d-2.0/src
                - make
                - echo "export PATH=$PWD:$PATH" >> ~/.bashrc
                - source ~/.bashrc
                - cd ../../
                - synthesizer --raytrace
            """, blue=True)
        install = input('Should I install it and run for you? (yes/no)')

        if install in ['', 'yes']:
            os.system('git clone https://github.com/dullemond/radmc3d-2.0.git')
            os.system('cd radmc3d-2.0/src')
            os.system('make')
            os.system('export PATH=$PWD:$PATH')
            os.system('source ~/.bashrc')
            os.system('cd ../../')
            self.run()
