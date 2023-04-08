import h5py
import numpy as np
import astropy.units as u
import astropy.constants as c
from abc import ABC, abstractmethod

from synthesizer.gridder.vector_field import VectorField
from synthesizer import utils

class SPHCode(ABC):
    """ Base Abstract class to set minimum requirements for all sph codes. """ 
    
    @property
    @abstractmethod
    def rho_g(self):
        pass

    @property
    def temp(self):
        return self._temp

    @temp.setter
    def temp(self, temp):
        if type(temp) != np.ndarray: 
            raise ValueError(
                f'temp must be a numpy array, not a {type(temp)}.')

        if temp.shape != dens.shape:
            raise ValueError(f'Array shape of temp must be equal to dens.' +\
                f'{temp.shape = }   {self._dens.shape = }')

        self._temp = temp

    @property
    def vx(self):
        return self._vx
    
    @property
    def vy(self):
        return self._vy

    @property
    def vz(self):
        return self._vz

    @vx.setter
    def vx(self, vx):
        self._vx = vx

    @vy.setter
    def vy(self, vy):
        self._vy = vy

    @vz.setter
    def vz(self, vz):
        self._vz = vz


class SPHng(SPHCode):
    """ Handle binary snapshots from the SPHng code. """
    def __init__(self, filename, temp=False, vfield=False, remove_sink=True):
        """
            Notes:

            This reader assumes your file is binary and formatted 
            with a header stating the quantities, assumed to be 
            f4 floats. May not be widely applicable.

            Header is: id t x y z vx vy vz mass hsml rho T u, where

            id = particle ID number
            t = simulation time [years]
            x, y, z = cartesian particle position [au]
            vx,vy,vz = cartesian particle velocity [need to check units]
            mass = particle mass [ignore]
            hmsl = particle smoothing length [ignore]
            rho = particle density [g/cm3]
            T = particle temperature [K]
            u = particle internal energy [ignore]
            
            There's an outlier (probably a sink particle) with index = 31330
        """
        
        self.add_temp = temp
        self.vfield = vfield

        # Read file in binary format
        try:
            with open(filename, "rb") as f:
                names = f.readline()[1:].split()
                self.data = np.frombuffer(f.read()).reshape(-1, len(names))
                self.data = self.data.astype("f4")
        except ValueError as e:
            utils.print_('Error trying to read SPHng binary file.', red=True)
            utils.print_(
                'File is probably from another code. Set --source', bold=True)
            raise 

        # Turn data into CGS 
        self.data[:, 2:5] *= u.au.to(u.cm)
        self.data[:, 8] *= u.M_sun.to(u.g)

        # Remove the cell data of the outlier, probably a sink particle
        if remove_sink:
            sink_id = 31330
            self.data[sink_id, 5] = 0
            self.data[sink_id, 6] = 0
            self.data[sink_id, 7] = 0
            self.data[sink_id, 8] = self.data[:,8].min()
            self.data[sink_id, 9] = self.data[:,9].min()
            self.data[sink_id, 10] = 3e-11 
            self.data[sink_id, 11] = 900
        
        self.x = self.data[:, 2]
        self.y = self.data[:, 3]
        self.z = self.data[:, 4]

    @property
    def rho_g(self): return self.data[:, 10]

    @property
    def temp(self):
        if self.add_temp: return self.data[:, 11]

    @property
    def vx(self):
        if self.vfield: return self.data[:, 5]

    @property
    def vy(self):
        if self.vfield: return self.data[:, 6]

    @property
    def vz(self):
        if self.vfield: return self.data[:, 7]


class Gizmo(SPHCode):
    """ Handle HDF5 snapshots from the GIZMO code. """
    def __init__(self, filename, temp=False, vfield=False):

        # Read in particle coordinates and density
        self.data = h5py.File(filename, 'r')['PartType0']
        coords = np.array(self.data['Coordinates'])
        self.x = coords[:, 0] * u.pc.to(u.cm)
        self.y = coords[:, 1] * u.pc.to(u.cm)
        self.z = coords[:, 2] * u.pc.to(u.cm)

        # Recenter the particles based on the center of mass
        self.x -= np.average(self.x, weights=self.rho_g)
        self.y -= np.average(self.y, weights=self.rho_g)
        self.z -= np.average(self.z, weights=self.rho_g)

        self.add_temp = temp
        self.vfield = vfield

    @property
    def rho_g(self):
        return np.array(self.data['Density']) * 1.3816327e-23

    @property
    def temp(self):
        # Read in temperature if available
        if 'Temperature' in self.data:
            return np.array(self.data['Temperature'])

        # Or maybe the krome temperature, when coupled with KROME
        elif 'KromeTemperature' in self.data:
            return np.array(self.data['KromeTemperature'])
        
        # Or derive from a barotropic equation of state
        elif 'InternalEnergy' in self.data.keys() and \
            'Pressure' in self.data.keys():
            return self.data['InternalEnergy'] / np.array(self.data['Pressure'])

        else:
            return np.zeros(self.rho_g.shape)

    @property
    def vx(self):
        if self.vfield: return self.data['MagneticField'][:, 0]

    @property
    def vy(self):
        if self.vfield: return self.data['MagneticField'][:, 1]

    @property
    def vz(self):
        if self.vfield: return self.data['MagneticField'][:, 2]


class Gadget(SPHCode):
    """ Handle snapshots from the Gadget code. """
    def __init__(self, filename, temp=False, vfield=False):
    
        self.add_temp = temp
        self.vfield = vfield 

        # Read in particle coordinates and density
        self.data = h5py.File(filename, 'r')['PartType0']
        coords = np.array(self.data['Coordinates'])
        self.x = coords[:, 0] * u.au.to(u.cm)
        self.y = coords[:, 1] * u.au.to(u.cm)
        self.z = coords[:, 2] * u.au.to(u.cm) 

        # Recenter the particles based on the center of mass
        self.x -= np.average(self.x, weights=self.rho_g)
        self.y -= np.average(self.y, weights=self.rho_g)
        self.z -= np.average(self.z, weights=self.rho_g)
    
    @property
    def rho_g(self):
        dens = np.array(self.data['Density'])
        return dens * 1e-10 * (u.Msun / u.au**3).to(u.g/u.cm**3)

    @property
    def temp(self):
        if self.add_temp:
            # Read in temperature if available
            if 'Temperature' in self.data.keys():
                return self.data['Temperature']

            # Or derive from a barotropic equation of state
            elif 'Pressure' in self.data.keys():
                kB = c.k_B.cgs.value
                mu = 2.3 * (c.m_e + c.m_p).cgs.value
                return (mu / kB) * np.array(self.data['Pressure']) / self.rho_g

            else:
                return np.zeros(self.rho_g.shape)

    @property
    def vx(self):
        if self.vfield: return self.data['MagneticField'][:, 0]

    @property
    def vy(self):
        if self.vfield: return self.data['MagneticField'][:, 1]

    @property
    def vz(self):
        if self.vfield: return self.data['MagneticField'][:, 2]


class Arepo(SPHCode):
    """ Handle snapshots from the AREPO code. """
    def __init__(self, filename, temp=False, vfield=False):

        self.add_temp = temp
        self.vfield = vfield

        # Read in particle coordinates and density
        self.data = h5py.File(filename, 'r')['PartType0']
        coords = np.array(self.data.get('Coordinates'))
        mass = np.array(self.data.get('Masses'))
        press = np.array(self.data.get('Pressure'))
        energy = np.array(self.data.get('InternalEnergy'))
        self.x = coords[:, 0] * u.au.to(u.cm)
        self.y = coords[:, 1] * u.au.to(u.cm)
        self.z = coords[:, 2] * u.au.to(u.cm)
        self.mass = mass * 1e-10 * u.Msun.to(u.g) 
        self.press = press * 1e-10 * (u.Msun/u.au/u.s**2).to(u.g/u.cm/u.s**2) 
        self.u = energy * 1e-10 * (u.au**2*u.Msun/u.s**2).to(u.cm**2*u.g/u.s**2)

        # Recenter the particles based on the center of mass
        self.x -= np.average(self.x, weights=self.mass)
        self.y -= np.average(self.y, weights=self.mass)
        self.z -= np.average(self.z, weights=self.mass)

    @property
    def rho_g(self):
        dens = np.array(self.data.get('Density'))
        return dens * 1e-10 * (u.Msun/u.au**3).to(u.g/u.cm**3)

    @property
    def temp(self):
        if self.add_temp:
            # Read in temperature if available
            if 'Temperature' in self.data.keys():
                return self.data.get('Temperature')

            # Derive from a barotropic EOS (Wurster et al. 2018, Eq.5)
            elif 'Pressure' in self.data.keys():
                k_B = c.k_B.cgs.value
                m_H = (c.m_e + c.m_p).cgs.value
                rho_c = 1e-14 * np.ones(self.rho_g.shape)
                rho_d = 1e-10 * np.ones(self.rho_g.shape)
                T_iso = 14 * np.ones(self.rho_g.shape)
                cs_iso2 = k_B * T_iso / 2.33 / m_H

                #T1 = np.where(self.rho_g >= rho_c, 
                T1 = np.where(
                    (rho_c <= self.rho_g) & (self.rho_g < rho_d), 
                    T_iso * (self.rho_g / rho_c)**(7/5), 
                    T_iso
                )

                T2 = np.where(
                    self.rho_g >= rho_d, 
                    T_iso * (self.rho_g / rho_d)**(7/5) * \
                            (self.rho_g / rho_d)**(11/10), 
                    T1
                )

                return T2

            else:
                return np.zeros(self.rho_g.shape)

    @property
    def vx(self):
        if self.vfield: return self.data['MagneticField'][:, 0]

    @property
    def vy(self):
        if self.vfield: return self.data['MagneticField'][:, 1]

    @property
    def vz(self):
        if self.vfield: return self.data['MagneticField'][:, 2]


class Phantom(SPHCode):
    """ Handle snapshots from the PHANTOM code. """
    def __init__(self, filename, temp=False, vfield=False):

        utils.not_implemented('PHANTOM Interface')
        
        # Read in particle coordinates and density
        self.data = h5py.File(filename, 'r')['particles']
        coords = self.data['xyz'].value

        # Derive density from the particle mass / h**3
        # src: https://github.com/dmentipl/plonk/blob/main/src/plonk/snap/readers/phantom.py
    
        self.add_temp = temp
        self.vfield = vfield

        self.x = coords[:, 0] * u.au.to(u.cm)
        self.y = coords[:, 1] * u.au.to(u.cm)
        self.z = coords[:, 2] * u.au.to(u.cm)

    @property
    def rho_g(self):
        dens = np.array(self.data['Density'])
        return dens * 1e-10 * (u.Msun / u.au**3).to(u.g/u.cm**3)

    @property
    def temp(self):
        return np.zeros(self.rho_g.shape)

    @property
    def vx(self):
        if self.vfield: return self.data['bfield'][:, 0]

    @property
    def vy(self):
        if self.vfield: return self.data['bfield'][:, 1]

    @property
    def vz(self):
        if self.vfield: return self.data['bfield'][:, 2]


class Nbody6(SPHCode):
    """ Handle snapshots from the Nbody6 code. """
    def __init__(self, filename, temp=False, vfield=False):

        self.add_temp = temp
        self.vfield = vfield
        
        # Read in data from HDF5 file
        if filename.endswith(('h5', 'hdf5')):
            utils.not_implemented('Nbody6-HDF5')
            self.data = h5py.File(filename, 'r')

        # Read in data from ASCII table
        else:
            self.data = np.loadtxt(filename)
            self.mass = self.data[:, 0] * u.Msun.to(u.g)
            self.radius = self.data[:, 1] * u.Rsun.to(u.cm)
            self.x = self.data[:, 3] * u.pc.to(u.cm)
            self.y = self.data[:, 4] * u.pc.to(u.cm)
            self.z = self.data[:, 5] * u.pc.to(u.cm)
            self.id = self.data[:, 10]

    @property
    def rho_g(self):
        return self.data[:, 2] * (u.Msun/u.Rsun**3).to(u.g/u.cm**3)

    @property
    def temp(self):
        return self.data[:, 9] 

    @property
    def vx(self):
        if self.vfield: return self.data[:, 6] * (u.km/u.s).to(u.cm/u.s)

    @property
    def vy(self):
        if self.vfield: return self.data[:, 7] * (u.km/u.s).to(u.cm/u.s)

    @property
    def vz(self):
        if self.vfield: return self.data[:, 8] * (u.km/u.s).to(u.cm/u.s)

