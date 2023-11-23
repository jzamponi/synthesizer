#!/usr/bin/env python3
"""
    This python program defines the Dust object, which can be used to
    calculate dust opacities from tabulated optical constants n & k. 
    This program also allows to mix different materials either using the
    Bruggeman rule for mixing optical constants into a new material or by 
    simply averaging the resultant opacities, weighted by a given mass fraction.

    The resultant extinction, scattering and absorption opacities can be 
    returned as arrays or written to file in radmc3d compatible format. 
    This file can optionally include the mueller matrix components for full 
    scattering and polarization modelling.

    At the end of this script an example of implementation can be found. 

    Wavelengths and grain sizes should be provided in microns, however all
    internal calculations are done in cgs.
"""

import os
import sys
import copy
import errno
import itertools
import progressbar
import numpy as np
from pathlib import Path
from astropy.io import ascii
from astropy import units as u
import matplotlib.pyplot as plt
from time import time, strftime, gmtime
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy.interpolate import interp1d

from synthesizer import utils
from synthesizer.dustmixer import bhmie, bhcoat


class Dust():
    """ Dust object. Defines the properties of a dust component. """
    verbose = False

    def __init__(self, name='', n=[], k=[], dens=0, l=None, 
            scatmatrix=False, pb=True):

        self.name = name
        self.l = np.logspace(-1, 5, 200) * u.micron.to(u.cm) if l is None else l
        self.n = n * np.ones(self.l.shape) if np.isscalar(n) else n
        self.k = k * np.ones(self.l.shape) if np.isscalar(k) else k
        self.dens = dens
        self.mf = 0
        self.vf = 0
        self.mass = 0
        self.mf_all = []
        self.Qext = None
        self.Qsca = None
        self.Qabs = None
        self.kext = None
        self.ksca = None
        self.kabs = None
        self.nlam = self.l.size
        self.scatmatrix = scatmatrix
        self.pb = pb

    def __str__(self):
        print(f'{self.name}')

    def __add__(self, other, nlam=None):
        """
            This 'magic' method allows dust opacities to be summed via:
            mixture = dust1 + dust2
            Syntax is important. Arithmetic operations with Dust objects 
            should ideally be grouped by parenthesis. 
        """
        if self.kext is None:
            raise ValueError('Dust opacities kext, ksca and kabs are not set')

        dust = copy.deepcopy(self)
        dust.kext = self.kext + other.kext
        dust.ksca = self.ksca + other.ksca
        dust.kabs = self.kabs + other.kabs
        dust.z11 = self.z11 + other.z11
        dust.z12 = self.z12 + other.z12
        dust.z22 = self.z22 + other.z22
        dust.z33 = self.z33 + other.z33
        dust.z34 = self.z34 + other.z34
        dust.z44 = self.z44 + other.z44
        dust.gsca = (self.gsca * self.ksca + other.gsca * other.ksca) / (
                    self.ksca + other.ksca)
        dust.name = ' + '.join([self.name, other.name])

        return dust

    def __mul__(self, mass_fraction):
        """
            This 'magic' method allows dust opacities to be rescaled as:
            dust = dust1 * 0.67
            The order is important. The number must be on the right side. 
        """
        if self.kext is None:
            raise ValueError('Dust opacities kext, ksca and kabs are not set')
    
        if isinstance(mass_fraction, Dust): 
            raise ValueError('Dust can only multiplied by scalars.')
        
        dust = copy.deepcopy(self)
        dust.kext = self.kext * mass_fraction
        dust.ksca = self.ksca * mass_fraction
        dust.kabs = self.kabs * mass_fraction
        dust.z11 = self.z11 * mass_fraction
        dust.z12 = self.z12 * mass_fraction
        dust.z22 = self.z22 * mass_fraction
        dust.z33 = self.z33 * mass_fraction
        dust.z34 = self.z34 * mass_fraction
        dust.z44 = self.z44 * mass_fraction
        dust.set_mfrac(mass_fraction)
        dust.mf_all.append(mass_fraction)

        return dust

    def __rmul__(self, other):
        """ Rightsided multiplication of Dust objects """
        return self * other

    def __div__(self, other):
        """ Division of Dust objects by scalars """
        return self * (1 / other)

    def __truediv__(self, other):
        """ Division of Dust objects by scalars """
        return self * (1 / other)

    def check_mfrac(self):
        if np.sum(self.mf_all) != 1:
            raise ValueError(
                f'Mass fractions should add up to 1. Values are {self.mf_all}') 
        else:
            utils.print('Mass fractions add up to 1. Values are {self.mf_all}')

    def set_density(self, dens, cgs=True):
        """ Set the bulk density of the dust component. """
        self.dens = dens if cgs else dens * (u.kg/u.m**3).to(u.g/u.cm**3)
    
    def set_mfrac(self, mf):
        """ Set the mass fraction of the dust component. """
        self.mf = mf

    def set_lgrid(self, lmin, lmax, nlam):
        """ Set the wavelength grid for the optical constants and opacities """
        self.lmin = lmin
        self.lmax = lmax
        self.nlam = nlam
        self.l = np.logspace(np.log10(lmin), np.log10(lmax), nlam)
        self.l = self.l * u.micron.to(u.cm)

    def interpolate(self, x, y, at):
        """ Linearly interpolate a quantity within the wavelength grid. """

        return interp1d(x, y)(at)

    def extrapolate(self, y, boundary):
        """ Extrapolate a quantity within the wavelength grid. """
    
        if boundary == 'lower':
            # Extrapolate as a constant, i.e., prepend a copy of the first value
            return np.insert(y, 0, y[0])

        elif boundary == 'upper':
            # Interval to find the extrapolation slope, from y1[-1] to y0[x0]  
            x0 = -8
            x = self.l_nk

            # Extrapolate in log-log: y = c·x^a
            a = (np.log(y[x0])-np.log(y[-1])) / (np.log(x[x0])-np.log(x[-1]))
            c = np.exp(np.log(y[-1]) - a * np.log(x[-1]))
            new_y = c * x.max()**a

            return np.insert(y, -1, new_y)

    def set_nk(self, path, skip=1, microns=True, meters=False, cm=False, 
            get_dens=True):
        """ Set n and k values by reading them from file. 
            Assumes wavelength is provided in microns unless specified otherwise
            Also, can optionally read the density from the file, assuming it is
            the second number in the header and comes in g/cm3.
         """
        
        # Download the optical constants from the internet if path is a url
        if "http" in path:
            utils.download_file(path)
            path = path.split('/')[-1]

        # Strip the filename from the full given url
        filename = path.split('/')[-1]

        # Read the table
        utils.print_(f'Reading optical constants from: {filename}', end='')
        self.datafile = ascii.read(path, data_start=skip)

        # Optionally read the density as the second number from the file header
        if get_dens:
            dens = float(ascii.read(path, data_end=1)['col2'])
            self.set_density(dens)

        print(f' | Density: {dens} g/cm3' if get_dens else '')
                
        # Override the default column names used by astropy.io.ascii.read
        self.l_nk = np.array(self.datafile['col1'])
        self.n = np.array(self.datafile['col2'])
        self.k = np.array(self.datafile['col3'])
        
        # Parse the wavelength units to ensure they are in cm
        if meters:
            self.l_nk = self.l_nk * u.m.to(u.cm)
        elif microns:
            self.l_nk = self.l_nk * u.micron.to(u.cm)

        # Extrapolate n and k lower and upper boundaries
        if self.l.min() < self.l_nk.min(): 
            utils.print_(
                f'Extrapolating lower {self.name} constants to ' +\
                f'{self.l.min()*u.cm.to(u.micron):.1e} microns')
            self.l_nk = np.insert(self.l_nk, 0, self.l.min())
            self.n = self.extrapolate(self.n, boundary='lower')
            self.k = self.extrapolate(self.k, boundary='lower')
        
        if self.l.max() > self.l_nk.max():
            utils.print_(
                f'Extrapolating upper {self.name} constants to ' +\
                f'{self.l.max()*u.cm.to(u.micron):.1e} microns')
            self.l_nk = np.insert(self.l_nk, -1, self.l.max())
            self.n = self.extrapolate(self.n, boundary='upper')
            self.k = self.extrapolate(self.k, boundary='upper')

        # Interpolate optical constants within the new wavelenght grid
        self.n = self.interpolate(self.l_nk, self.n, at=self.l)
        self.k = self.interpolate(self.l_nk, self.k, at=self.l)
        
    def mix(self, comps, rule='bruggeman', porosity=0):
        """
            Mix two dust components using the bruggeman rule. 
        """

        #rule = {'b': 'bruggeman', 'mg': 'maxwell-garnett'}[rule]
        if rule == 'b': rule = 'bruggeman'
        elif rule == 'mg': rule = 'maxwell-garnett'
    
        utils.print_(f'Mixing materials using the {rule} rule')
    
        # If just one other material is given, mix with itself
        if isinstance(comps, Dust): comps = [self, comps]

        # Mix the optical constants for every wavelenthg
        if rule in ['b', 'bruggeman']:
            self.n, self.k, self.dens = self.bruggeman_mixing(comps)

        elif rule in ['mg', 'maxwell-garnett']:
            self.n, self.k, self.dens = self.maxwell_garnett_mixing(comps)

        # Add porosity (i.e., mix this new mixture once again with vacuum)
        if porosity > 0:
            utils.not_implemented('Porosity is momentarily disabled.')
            vacuum = Dust(n=1, k=0, dens=0, name='vacuum')
            vacuum.set_mfrac(porosity)
            self.set_mfrac(1 - porosity)
            self.dens *= 1 - porosity
            self.n, self.k, _ = self.mix(vacuum, rule='mg')

    def bruggeman_mixing(self, comps):
        """ This function explicity mixes the n & k indices using the 
            Bruggeman mixing rule. 
            Receives: list of dust objects
            Returns: (n,k)
        """
        from mpmath import findroot

        # Mean density of the mixture
        dens = 1 / np.sum([i.mf / i.dens for i in comps])

        # Convert mass fractions to volume fractions
        vfrac = dens * np.array([i.mf / i.dens for i in comps]) 

        # Iterate over wavelenghts
        eps_mean = np.empty(np.shape(self.l)).astype('complex')

        for i, l in enumerate(self.l):

            # List the epsilon = m^2 = (n+ik)^2 for all compositions
            eps = np.array([complex(c.n[i], c.k[i])**2 for c in comps])

            # Define the expresion for mixing and solve for eps_mean
            function = lambda x: sum(vfrac * ((eps - x) / (eps + 2 * x)))
            eps_mean[i] = complex(findroot(function, complex(0.5, 0.5)))
        
        eps_mean = np.sqrt(eps_mean)

        return eps_mean.real.squeeze(), eps_mean.imag.squeeze(), dens

    def maxwell_garnett_mixing(self, comps):
        """ This function explicity mixes the n & k indices using the 
            Maxwell-Garnett mixing rule. 
            Receives: list of dust objects
            Returns: (n,k)
        """
        from mpmath import findroot

        # Mean density of the mixture
        dens = 1 / np.sum(
            [i.mf / i.dens if i.dens != 0 else i.mf for i in comps])

        # Convert mass fractions to volume fractions
        vfrac = dens * np.array(
            [i.mf / i.dens if i.dens != 0 else i.mf for i in comps]) 

        # Iterate over wavelenghts
        eps_mean = np.empty(np.shape(self.l)).astype('complex')

        for i, l in enumerate(self.l):

            # List the epsilon = m^2 = (n+ik)^2 for all compositions
            eps = np.array([complex(c.n[i], c.k[i])**2 for c in comps])

            eps_m = eps[0]
            eps_i = eps[1:]
            f_i = vfrac[1:]
            beta_i = 3 * eps_m / (eps_i + 2 * eps_m)
            eps_mean[i] = ((1-f_i.sum()) * eps_m + (f_i * beta_i * eps_i).sum()) / \
                (1 - f_i.sum() + (f_i * beta_i).sum())
        
        eps_mean = np.sqrt(eps_mean)

        return eps_mean.real.squeeze(), eps_mean.imag.squeeze(), dens
        
    def get_efficiencies(self, a, nang=3, algorithm='bhmie', coat=None, 
            verbose=True, parallel_counter=0):
        """ 
            Compute the extinction, scattering and absorption
            efficiencies (Q) by calling bhmie or bhcoat.

            Arguments: 
              - a: Size of the dust grain in cm
              - nang: Number of angles to sample scattering between 0 and 180
              - algorithm: 'bhmie' or 'bhcoat', algorithm used to calculate Q 
              - coat: Dust Object, material used as a iced coat for the grain
        
            Returns:
              - Qext: Dust extinction efficiency
              - Qsca: Dust scattering efficiency
              - Qabs: Dust absorption efficiency
              - gsca: Assymetry parameter for Henyey-Greenstein scattering
        """

        self.angles = np.linspace(0, 180, self.nang)
        self.mass  = (4 / 3 * np.pi) * self.dens * a**3
        self.Qext = np.zeros(self.l.size)
        self.Qsca = np.zeros(self.l.size)
        self.Qabs = np.zeros(self.l.size)
        self.Qbac = np.zeros(self.l.size)
        self.Gsca = np.zeros(self.l.size)
        self.s11 = np.zeros(nang)
        self.s12 = np.zeros(nang)
        self.s33 = np.zeros(nang)
        self.s34 = np.zeros(nang)
        self.Z11 = np.zeros((self.l.size, nang))
        self.Z12 = np.zeros((self.l.size, nang))
        self.Z22 = np.zeros((self.l.size, nang))
        self.Z33 = np.zeros((self.l.size, nang))
        self.Z34 = np.zeros((self.l.size, nang))
        self.Z44 = np.zeros((self.l.size, nang))
        self.current_a = a
        a_micron = a * u.cm.to(u.micron)

        utils.print_(f'Calulating {self.name} efficiencies Q for a grain size'+\
            f' of {np.round(a_micron, 1)} microns', verbose=verbose)

        # Calculate dust efficiencies for a bare grain
        if algorithm.lower() == 'bhmie':
            
            # Iterate over wavelength
            for i, l_ in enumerate(self.l):
                # Define the size parameter
                self.x = 2 * np.pi * a / l_

                # Define the complex refractive index (m)
                self.m = complex(self.n[i], self.k[i])
                
                # Compute dust efficiencies using BHMIE (Bohren & Huffman 1986)
                bhmie_ = bhmie.bhmie(self.x, self.m, self.angles)
                
                # Store the results
                s1 = bhmie_[0]
                s2 = bhmie_[1]
                self.Qext[i] = bhmie_[2]
                self.Qsca[i] = bhmie_[3]
                self.Qabs[i] = bhmie_[4]
                self.Qbac[i] = bhmie_[5]
                self.Gsca[i] = bhmie_[6]

                if self.scatmatrix:
                    # Compute the Scattering Matrix Elements
                    self.s11 = 0.5 * (np.abs(s2)**2 + np.abs(s1)**2)
                    self.s12 = 0.5 * (np.abs(s2)**2 - np.abs(s1)**2)
                    self.s33 = 0.5 * np.real(s1 * np.conj(s2) + s2 * np.conj(s1))
                    self.s34 = 0.5 * np.imag(s1 * np.conj(s2) - s2 * np.conj(s1))
                    
                    # Normlize the Scattering Matrix 
                    k = 2 * np.pi / l_
                    factor = 1 / (k**2 * self.mass)
                    self.Z11[i, :] = self.s11 / (k**2 * self.mass)
                    self.Z12[i, :] = self.s12 / (k**2 * self.mass)
                    self.Z22[i, :] = self.s11 / (k**2 * self.mass)
                    self.Z33[i, :] = self.s33 / (k**2 * self.mass)
                    self.Z34[i, :] = self.s34 / (k**2 * self.mass)
                    self.Z44[i, :] = self.s33 / (k**2 * self.mass)

        # Calculate dust efficiencies for a coated grain
        elif algorithm.lower() == 'bhcoat':

            utils.not_implemented('Opacities for coated grains.')

            if coat is None:
                raise ValueError(f'In order to use bhcoat you must provide '+\
                    'a Dust object to use as a coat')

            # Set the grain sizes for the coat
            a_coat = a * coat.vf

            for i, l_ in enumerate(self.l):
                # Define the size parameter
                self.x = 2 * np.pi * a / l_
                self.y = 2 * np.pi * a_coat / l_

                # Set the complex refractive index for the core
                self.m_core = complex(self.n_interp[i], self.k_interp[i])

                # Set the complex refractive index for the mantle
                self.m_mant = complex(coat.n_interp[i], coat.k_interp[i])
                
                # Calculate the efficiencies for a coated grain
                bhcoat_ = bhcoat.bhcoat(self.x, self.y, self.m_core, self.m_mant)
                
                self.Qext[i] = bhcoat_[0]
                self.Qsca[i] = bhcoat_[1]
                self.Qabs[i] = bhcoat_[2]
                self.Qbac[i] = bhcoat_[3]

        else:
            raise ValueError(f'Invalid value for algorithm = {algorithm}.')

        # Print a simpler progress meter if using multiprocessing
        if self.nproc > 1 and self.pb:
            i = parallel_counter
            counter = i * 100 / self.a.size
            endl = '\r' if i != self.a.size-1 else '\n'
            bar = (' ' * 20).replace(" ", "⣿⣿", int(counter / 100 * 20))
            sys.stdout.write(f'[get_efficiencies] Using {self.nproc} processes'+\
                f' | Progress: {counter} % |{bar:>20}| {endl}')
            sys.stdout.flush()

        return self.Qext, self.Qsca, self.Qabs, self.Gsca, \
            self.Z11, self.Z12, self.Z22, self.Z33, self.Z34, self.Z44

        
    def get_opacities(self, a=np.logspace(-1, 2, 100), q=-3.5, 
            nang=2, nproc=1, algorithm='bhmie'):
        """ 
            Convert the dust efficiencies into dust opacities by integrating 
            them over a range of grain sizes. Assumes grain sizes are given in
            microns and a power-law size distribution with slope q.

            Arguments:  for a={int(self.a)} microns for a={int(self.a)} microns
              - a: Array containing the sizes of the grain size distribution
              - q: Exponent of the power-law grain size distribution
              - algorithm: 'bhmie' or 'bhcoat', algorithm for get_efficiencies
              - nang: Number of angles used in get_efficiencies
        
            Returns:
              - kext: Dust extinction opacity (cm^2/g_dust)
              - ksca: Dust scattering opacity (cm^2/g_dust)
              - kabs: Dust absorption opacity (cm^2/g_dust)
        """

        if np.isscalar(a): self.a = np.array([a]) 
        self.a = a * u.micron.to(u.cm)
        self.amin = self.a.min()
        self.amax = self.a.max()
        self.q = q
        self.na = np.size(self.a)
        self.nang = nang
        self.angles = np.linspace(0, 180, nang)
        self.kext = np.zeros(self.l.size)
        self.ksca = np.zeros(self.l.size)
        self.kabs = np.zeros(self.l.size)
        self.gsca = np.zeros(self.l.size)
        self.z11 = np.zeros((self.l.size, nang))
        self.z12 = np.zeros((self.l.size, nang))
        self.z22 = np.zeros((self.l.size, nang))
        self.z33 = np.zeros((self.l.size, nang))
        self.z34 = np.zeros((self.l.size, nang))
        self.z44 = np.zeros((self.l.size, nang))
        self.Qext_a = []
        self.Qsca_a = []
        self.Qabs_a = []
        self.gsca_a = []
        self.z11_a = []
        self.z12_a = []
        self.z22_a = []
        self.z33_a = []
        self.z34_a = []
        self.z44_a = []
        self.nproc = nproc
 
        utils.print_(f'Calculating efficiencies for {self.name} using ' +\
            f'{self.na} sizes between {a.min()} and {int(a.max())} microns ...')

        # In case of a single grain size, skip parallelization and integration
        if self.amin == self.amax or self.na == 1:
            qe, qs, qa, gs, z11, z12, z22, z33, z34, z44 = \
                self.get_efficiencies(self.a[0], nang, algorithm)

            self.kext = qe * np.pi * self.a[0]**2
            self.ksca = qa * np.pi * self.a[0]**2
            self.kabs = qa * np.pi * self.a[0]**2
            self.gsca = gs
            self.z11 = z11
            self.z12 = z12
            self.z22 = z22
            self.z33 = z33
            self.z34 = z34
            self.z44 = z44
            
            return self.kext, self.ksca, self.kabs, self.gsca, \
                self.z11, self.z12, self.z22, self.z33, self.z34, self.z44

        # Serial execution
        if self.nproc == 1:
            # Customize the progressbar
            widgets = [f'[get_opacities] ', progressbar.Timer(), ' ', 
                progressbar.GranularBar(' ⡀⡄⡆⡇⣇⣧⣷⣿')]

            if self.pb:
                pb = progressbar.ProgressBar(maxval=self.a.size, widgets=widgets)
                pb.start()

            # Calculate the efficiencies for the range of grain sizes
            for j, a_ in enumerate(self.a):
                qe, qs, qa, gs, z11, z12, z22, z33, z34, z44 = \
                        self.get_efficiencies(a_, nang, algorithm, None, False)
                self.Qext_a.append(qe) 
                self.Qsca_a.append(qs) 
                self.Qabs_a.append(qa) 
                self.gsca_a.append(gs) 
                self.z11_a.append(z11) 
                self.z12_a.append(z12) 
                self.z22_a.append(z22) 
                self.z33_a.append(z33) 
                self.z34_a.append(z34) 
                self.z44_a.append(z44) 
                if self.pb: pb.update(j)
            if self.pb: pb.finish()

        # Multiprocessing (Parallelized)
        else:
            # Calculate the efficiencies for the range of grain sizes
            with multiprocessing.Pool(processes=self.nproc) as pool:
                params = zip(
                    self.a, 
                    itertools.repeat(nang), 
                    itertools.repeat(algorithm),
                    itertools.repeat(None),
                    itertools.repeat(False),
                    range(self.a.size), 
                )
                # Parallel map function
                qe, qs, qa, gs, z11, z12, z22, z33, z34, z44 = \
                    pool.starmap(self.get_efficiencies, params)
                
                # Reorder from (a, Q, l) to (Q, a, l)
                self.Qext_a = qe
                self.Qsca_a = qs
                self.Qabs_a = qa
                self.gsca_a = gs
                self.z11_a = z11
                self.z12_a = z12
                self.z22_a = z22
                self.z33_a = z33
                self.z34_a = z34
                self.z44_a = z44
    
        # Transpose from (a, l) to (l, a) to later integrate over l
        self.Qext_a = np.transpose(self.Qext_a)
        self.Qsca_a = np.transpose(self.Qsca_a)
        self.Qabs_a = np.transpose(self.Qabs_a)
        self.gsca_a = np.transpose(self.gsca_a)
        self.z11_a = np.swapaxes(self.z11_a, 0, 1)
        self.z12_a = np.swapaxes(self.z12_a, 0, 1)
        self.z22_a = np.swapaxes(self.z22_a, 0, 1)
        self.z33_a = np.swapaxes(self.z33_a, 0, 1)
        self.z34_a = np.swapaxes(self.z34_a, 0, 1)
        self.z44_a = np.swapaxes(self.z44_a, 0, 1)
        
        utils.print_(f'Integrating opacities ', end='')
        print(f'and scattering matrix ' if self.scatmatrix else '', end='')
        print(f'using a power-law slope of {q = }')

        # Mass integral: int (a^q * a^3) da = [amax^(q+4) - amin^(q-4)]/(q-4)
        q4 = self.q + 4
        int_da = (self.amax**q4 - self.amin**q4) / q4
        
        # Total mass normalization constant
        mass_norm = 4 / 3 * np.pi * self.dens * int_da

        # Size distribution
        phi = self.a**self.q

        # Calculate mass weight for the size integration of Z11
        mass = 4/3 * np.pi * self.a**3 * self.dens 
        m_of_a = (self.a*u.cm.to(u.micron))**(self.q + 1) * mass
        mtot = np.sum(m_of_a)
        mfrac = m_of_a / mtot

        # Integrate quantities over size distribution per wavelength 
        for i, l_ in enumerate(self.l):
            sigma_geo = np.pi * self.a**2
            Cext = self.Qext_a[i] * sigma_geo
            Csca = self.Qsca_a[i] * sigma_geo
            Cabs = self.Qabs_a[i] * sigma_geo

            # Integrate Zij
            for j in range(self.nang): 
                self.z11[i][j] = np.sum(self.z11_a[i, :, j] * mfrac)
                self.z12[i][j] = np.sum(self.z12_a[i, :, j] * mfrac)
                self.z22[i][j] = np.sum(self.z22_a[i, :, j] * mfrac)
                self.z33[i][j] = np.sum(self.z33_a[i, :, j] * mfrac)
                self.z34[i][j] = np.sum(self.z34_a[i, :, j] * mfrac)
                self.z44[i][j] = np.sum(self.z44_a[i, :, j] * mfrac)
            
            # Angular integral of Z11
            mu = np.cos(self.angles * np.pi / 180)
            int_Z11_dmu = -np.trapz(self.z11[i, :], mu)
            int_Z11_mu_dmu = -np.trapz(self.z11[i, :] * mu, mu)

            if self.scatmatrix:
                self.ksca[i] = 2 * np.pi * int_Z11_dmu
                self.gsca[i] = 2 * np.pi * int_Z11_mu_dmu / self.ksca[i]
            else:
                self.ksca[i] = np.trapz(Csca * phi, self.a) / mass_norm
                self.gsca[i] = np.sum(self.gsca_a[i] * mfrac)

            self.kext[i] = np.trapz(Cext * phi, self.a) / mass_norm
            self.kabs[i] = np.trapz(Cabs * phi, self.a) / mass_norm

            # Calculate the relative error between kscat and int Z11 dmu
            if self.scatmatrix:
                self.compare_ksca_vs_z11(i)

        if self.scatmatrix:
            self.check_ksca_z11_error(tolerance=0.1, show=False)

        return self.kext, self.ksca, self.kabs, self.gsca, \
            self.z11, self.z12, self.z22, self.z33, self.z34, self.z44

    def compare_ksca_vs_z11(self, lam_i):
        """ Compute the relative error between ksca and int Z11 dmu """
        self.err_i = np.zeros(self.l.size)
        mu = np.cos(self.angles * np.pi / 180)
        dmu = np.abs(mu[1:self.nang] - mu[0:self.nang-1])
        zav = 0.5 * (self.z11[lam_i, 1: self.nang] + 
            self.z11[lam_i, 0:self.nang-1])
        dum = 0.5 * zav * dmu
        self.dumsum = 4 * np.pi * dum.sum()
        err = np.abs(self.dumsum / self.ksca[lam_i] - 1)
        self.err_i[lam_i] = np.max(err, 0)
    
    def check_ksca_z11_error(self, tolerance, show=False):
        """ Warn if the error between kscat and int Z11 dmu is large """
        if np.any(self.err_i > tolerance):
            maxerr = np.round(self.err_i.max(), 1)
            lam_maxerr = self.l[utils.maxpos(self.err_i)] * u.cm.to(u.micron)
            utils.print_(
                'The relative error between ksca and the ' +\
                f'angular integral of Z11 is larger than {tolerance}.',red=True)
            utils.print_(
                f'Max Error: {maxerr} at {lam_maxerr:.1f} microns', bold=True)

            if show:
                plt.semilogx(self.l*u.cm.to(u.micron), self.err_i)
                plt.xlabel('Wavelength (microns)')
                plt.xlabel('Relative error')
                plt.annotate(
                    r'$err=\frac{\kappa_{\rm sca}}{\int Z_{11}(\mu)d\mu}$',
                    xy = (0.1, 0.9), xycoords='axes fraction', size=20)
                plt.show()

    def write_opacity_file(self, name=None):
        """ Write the dust opacities into a file ready for radmc3d """ 

        # Parse the table filename 
        name = self.name if name is None else name
        outfile = f'dustkappa_{name}.inp'
        if self.scatmatrix: 
            outfile = outfile.replace('kappa', 'kapscatmat') 

        utils.print_(f'Writing out radmc3d opacity file: {outfile}')
        with open(outfile, 'w+') as f:
            # Write a comment with info
            f.write(f'# Opacity table generated by Dustmixer\n')
            f.write(f'# Material = {self.name}\n')
            f.write(f'# Density = {self.dens} g/cm3\n')
            f.write(f'# Minimum grain size = {np.round(self.amin*1e4, 3)}um\n')
            f.write(f'# Maximum grain size = {np.round(self.amax*1e4, 3)}um\n')
            f.write(f'# Number of sizes = {self.na}\n')
            f.write(f'# Distribution slope = {self.q}\n')
            f.write(f'# Number of scattering angles: {self.nang}\n')
            f.write(f'# Columns are: wavelength (microns), k_abs (cm2/g), ')
            f.write(f'k_sca (cm2/g), g_sca\n')

            # Write file header
            f.write('1\n' if self.scatmatrix else '3\n')
            f.write(f'{self.l.size}\n')
            if self.scatmatrix:
                f.write(f'{self.nang}\n')
            
            # Write the opacities and g parameter per wavelenght
            for i, l in enumerate(self.l):
                f.write(f'{l*u.cm.to(u.micron):.6e}\t{self.kabs[i]:13.6e}\t')
                f.write(f'{self.ksca[i]:13.6e}\t{self.gsca[i]:13.6e}\n')

            if self.scatmatrix:
                for j, ang in enumerate(self.angles):
                    # Write scattering angle sampling points in degrees
                    f.write(f'{ang}\n')

                for i, l in enumerate(self.l):
                    for j, ang in enumerate(self.angles):
                        # Write the Mueller matrix components
                        f.write(f'{self.z11[i, j]:13.6e} ')
                        f.write(f'{self.z12[i, j]:13.6e} ')
                        f.write(f'{self.z22[i, j]:13.6e} ')
                        f.write(f'{self.z33[i, j]:13.6e} ')
                        f.write(f'{self.z34[i, j]:13.6e} ')
                        f.write(f'{self.z44[i, j]:13.6e}\n')

    def write_align_factor(self, name=None):
        """ Write the dust alignment factor into a file ready for radmc3d """ 

        # Parse the table filename 
        name = self.name if name is None else name
        outfile = f'dustkapalignfact_{name}.inp'
        utils.print_(f'Writing out radmc3d align factor file: {outfile}')

        # Create a mock alignment model. src:radmc3d/examples/run_simple_1_align
        mu = np.linspace(1, 0, self.nang)
        eta = np.arccos(mu) * 180 / np.pi
        amp = 0.5
        orth = np.ones(self.nang)
        para = (1 - amp * np.cos(mu * np.pi)) / (1 + amp)

        with open(outfile, 'w+') as f:
            f.write('1\n')
            f.write(f'{self.l.size}\n')
            f.write(f'{self.nang}\n')

            for l in self.l:
                f.write(f'{l*u.cm.to(u.micron):13.6e}\n')

            for i in eta:
                f.write(f'{i:13.6e}\n')

            for j in range(self.l.size):
                for a in range(self.nang):
                    f.write(f'{orth[a]:13.6e}\t{para[a]:13.6e}\n')

    def plot_nk(self, show=True, savefig=None):
        """ Plot the interpolated values of the refractive index (n & k). """
        
        if len(self.n) == 0 or len(self.k) == 0: 
            utils.print_('Optical constants n and k have not been set.')
            return

        plt.close()
        fig, p = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        twin_p = p.twinx()
        utils.print_(f'Plotting optical constants n & k from {self.name}')

        l = self.l * u.cm.to(u.micron)
        n = p.semilogx(l, self.n, ls='-', color='black')
        k = twin_p.loglog(l, self.k, ls=':', color='black')
        p.text(0.10, 0.95, self.name, fontsize=13, transform=p.transAxes)
        p.legend(n+k, ['n','k'], loc='upper left')
        p.set_xlabel('Wavelength (microns)')
        p.set_ylabel('n')
        twin_p.set_ylabel('k')
        p.set_xlim(l.min(), l.max())
        p.set_ylim(self.n.min(), self.n.max())
        twin_p.set_ylim(self.k.min(), self.k.max())
        plt.tight_layout()

        return utils.plot_checkout(fig, show, savefig)

    def plot_efficiencies(self, show=True, savefig=None):
        """ Plot the extinction, scattering & absorption eficiencies.  """

        if self.Qext is None or self.Qsca is None: 
            utils.print_('Dust efficiencies have not been calculated.')
            return

        plt.close()
        fig, p = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        a = np.round(self.current_a * u.cm.to(u.micron), 3)
        utils.print_(f'Plotting dust efficiencies for {a = } microns')
        
        p.loglog(self.l*u.cm.to(u.micron), self.Qext, ls='-', c='black',)
        p.loglog(self.l*u.cm.to(u.micron), self.Qsca, ls=':', c='black',)
        p.loglog(self.l*u.cm.to(u.micron), self.Qabs, ls='--', c='black')
        p.legend([r'$Q_{\rm ext}$', r'$Q_{\rm sca}$', r'$Q_{\rm abs}$'])
        p.annotate(r'$a = $'+f' {a} '+r'$\mu$m', xy=(0.1, 0.1), 
            xycoords='axes fraction', size=20)
        p.text(0.05, 0.95, self.name, fontsize=13, transform=p.transAxes)
        p.set_xlabel('Wavelength (microns)')
        p.set_ylabel(r'$Q$')
        plt.tight_layout()

        return utils.plot_checkout(fig, show, savefig)

    def plot_gsca(self, show=True, savefig=None):
        """ Plot the g scattering parameter (Henyey & Greenstein).  """

        if self.gsca is None: 
            utils.print_('Scattering parameter gsca has not been calculated.')
            return

        plt.close()
        fig, p = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        utils.print_(f'Plotting scattering parameter gsca')

        p.semilogx(self.l*u.cm.to(u.micron), self.gsca, ls='-', c='black')
        p.text(0.55, 0.90, r'$g=\int_{-1}^1 p(\mu)\mu d\mu,\,\,\mu=\cos \theta$',
            fontsize=18, transform=p.transAxes)
        p.set_xlabel('Wavelength (microns)')
        p.set_ylabel(r'$g_{\rm sca}$')
        p.set_ylim(-1, 1)
        #p.set_xlim(1e-1, 3e4)
        plt.tight_layout()
        return utils.plot_checkout(fig, show, savefig)

    def plot_opacities(self, show=True, savefig=None):
        """ Plot the extinction, scattering & absorption eficiencies.  """

        if self.kext is None or self.ksca is None: 
            utils.print_('Dust opacities have not been calculated.')
            return

        plt.close()
        fig, p = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        utils.print_(f'Plotting dust opacities')

        p.loglog(self.l*u.cm.to(u.micron), self.kext, ls='-', c='black')
        p.loglog(self.l*u.cm.to(u.micron), self.ksca, ls=':', c='black')
        p.loglog(self.l*u.cm.to(u.micron), self.kabs, ls='--', c='black')
        p.legend([r'$k_{\rm ext}$', r'$k_{\rm sca}$', r'$k_{\rm abs}$'])
        p.text(0.05, 0.95, self.name, fontsize=13, transform=p.transAxes)
        p.set_xlabel('Wavelength (microns)')
        p.set_ylabel(r'Dust opacity $k$ (cm$^2$ g$^{-1}$)')
        p.set_ylim(1e-2, 1e4)
        plt.tight_layout()

        return utils.plot_checkout(fig, show, savefig)

    def _get_kappa_at_lam(self, lam):
        """ Return the extinction dust opacity at a given wavelength """

        return float(self.interpolate(
            self.l, self.kext, at=lam*u.micron.to(u.cm)))

    def plot_z12z11(self, lam, a, nang, show=True, savefig=None):
        """ Plot the ratio -Z12/Z11 as a function of the scat. angle """

        angs = np.linspace(0, 180, nang)
        z12 = np.zeros(angs.shape)
        z11 = np.zeros(angs.shape)

        for i, ang in enumerate(angs):
            z12[i] = self.interpolate(
                self.l, self.z12[:, i], at=lam*u.micron.to(u.cm))
            z11[i] = self.interpolate(
                self.l, self.z11[:, i], at=lam*u.micron.to(u.cm))

        plt.close()
        fig, p = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        utils.print_(f'Plotting degree of pol. per scattering angle')

        p.plot(angs, -z12 / z11, color='black')
        p.set_xlabel('Scattering angle [degrees]')
        p.set_ylabel(r'$-Z_{12}/Z_{21}$')
        p.set_ylim(-1, 1)
        p.set_xlim(0, 180)
        plt.tight_layout()

        return utils.plot_checkout(fig, show, savefig)


if __name__ == "__main__":
    """
        Below you find a simple implementation of the dustmixer.
    """

    # Create Dust materials
    sil = Dust(name='Silicate', scatmatrix=True)
    gra = Dust(name='Graphite', scatmatrix=True)

    # Load refractive indices n and k from files
    sil.set_nk('nk/astrosil-Draine2003.lnk', microns=True, skip=2, get_dens=True)
    gra.set_nk('nk/c-gra-Draine2003.lnk', microns=True, skip=2, get_dens=True)

    # Convert the refractive indices into dust opacities
    sil.get_opacities(a=np.logspace(-1, 1, 20), nang=181)
    gra.get_opacities(a=np.logspace(-1, 1, 20), nang=181)

    # Mix silicate and graphite opacities weighted by a mass fracion
    mixture = (0.625 * sil) + (0.375 * gra)
    
    # Diagnostic plots
    sil.plot_nk(show=False)
    gra.plot_nk(show=False)
    sil.plot_gsca(show=True)
    gra.plot_gsca(show=True)
    mixture.plot_gsca(show=True)

    sil.plot_opacities(show=True)
    gra.plot_opacities(show=True)
    mixture.plot_opacities(show=True)

    # Write the opacity table of the mixed material including scattering matrix
    mixture.write_opacity_file(name='sg-a10um')
