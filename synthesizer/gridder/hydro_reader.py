import h5py
import numpy as np
import astropy.units as u
import astropy.constants as c
from abc import ABC, abstractmethod

from synthesizer.gridder.vector_field import VectorField
from synthesizer import utils

class HydroCode(ABC):
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


### SPH particle based codes

class SPHng(HydroCode):
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


class Gizmo(HydroCode):
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


class Gadget(HydroCode):
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


class Arepo(HydroCode):
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


class Phantom(HydroCode):
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


class Nbody6(HydroCode):
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


### AMR Grid based codes

class Athena(HydroCode):
    """ 
        Object designed to read in spatial and physical quantities from 
        Athena++ snapshots.
    """
    def __init__(self, filename, geometry, temp=False, vfield=False):
        self.add_temp = temp
        self.vfield = vfield

        import h5py

        # radmc3d standard: (x,y,z) --> (r,th,ph)
        self.data = h5py.File(filename, "r")
        self.x = np.ravel(self.data['x1f'])
        self.y = np.ravel(self.data['x2f']) 
        self.z = np.ravel(self.data['x3f'])

        self.x = self.x * u.au.to(u.cm)
        if geometry == 'cartesian':
            self.y = self.y * u.au.to(u.cm)
            self.z = self.z * u.au.to(u.cm)

    @property
    def rho_g(self):
        return self.data['prim'][0].ravel()

    @property
    def temp(self):
        return self.data[:, 9] 

    @property
    def vx(self):
        return self.data['prim'][2].ravel() * (u.km/u.s).to(u.cm/u.s)

    @property
    def vy(self):
        return self.data['prim'][3].ravel() * (u.km/u.s).to(u.cm/u.s)

    @property
    def vz(self):
        return self.data['prim'][4].ravel() * (u.km/u.s).to(u.cm/u.s)


class Flash(HydroCode):
    """ 
        Object designed to read in spatial and physical quantities from 
        FLASH snapshots.
    """
    def __init__(self):
        utils.not_implemented()

class Enzo(HydroCode):
    """ 
        Object designed to read in spatial and physical quantities from 
        ENZO snapshots.
    """
    def __init__(self):
        utils.not_implemented()

class Ramses(HydroCode):
    """ 
        Object designed to read in spatial and physical quantities from 
        RAMSES snapshots.
    """
    def __init__(self):
        utils.not_implemented()

class ZeusTW(HydroCode):
    """ 
        This object is an adapted copy of Zeus2Polaris._read_data()
        src: https://github.com/jzamponi/zeus2polaris
    """ 
    def __init__(self):
        self.rho = None
        self.rho = None
        self.Br  = None
        self.Bth = None
        self.Bph = None
        self.Vr  = None
        self.Vth = None
        self.Vph = None
        self.cordsystem = 150

    def read(self, filename, coord=False):
        """ Read the binary files from zeusTW.
            Reshape to 3D only if it is a physical quantity and not a coordinate
        """
        # Load binary file
        with open(filename, "rb") as binfile:
            data = np.fromfile(file=binfile, dtype=np.double, count=-1)

        if coord:
            return data
        else:
            shape = (self.r.size, self.th.size, self.ph.size)
            return data.reshape(shape, order='F')

    def generate_coords(self, r, th, ph):
        self.r = self.read(r, coord=True)
        self.th = self.read(th, coord=True)
        self.ph = self.read(ph, coord=True)
        self.x = self.r
        self.y = self.th
        self.z = self.ph

    def trim_ghost_cells(self, field_type, ng=3):
        if field_type == 'coords':
            # Trim ghost zones for the coordinate fields
            self.r = self.r[ng:-ng]
            self.th = self.th[ng:-ng]
            self.ph = self.ph[ng:-ng]
            self.x = self.x[ng:-ng]
            self.y = self.y[ng:-ng]
            self.z = self.z[ng:-ng]

        elif field_type == 'scalar':
            # Trim ghost cells for scalar fields
            self.rho = self.rho[ng:-ng, ng:-ng, ng:-ng]

        elif field_type == 'vector':
            # Trim ghost cells for vector fields
            self.Vr = 0.5 * (self.Vr[ng:-ng, ng:-ng, ng:-ng] + \
                            self.Vr[ng+1:-ng+1, ng:-ng, ng:-ng])
            self.Vth = 0.5 * (self.Vth[ng:-ng, ng:-ng, ng:-ng] + \
                            self.Vth[ng:-ng, ng+1:-ng+1, ng:-ng])
            self.Vph = 0.5 *  (self.Vph[ng:-ng, ng:-ng, ng:-ng] + \
                            self.Vph[ng:-ng, ng:-ng, ng+1:-ng+1])

            self.Br = 0.5 * (self.Br[ng:-ng, ng:-ng, ng:-ng] + \
                            self.Br[ng+1:-ng+1, ng:-ng, ng:-ng])
            self.Bth = 0.5 * (self.Bth[ng:-ng, ng:-ng, ng:-ng] + \
                            self.Bth[ng:-ng, ng+1:-ng+1, ng:-ng])
            self.Bph = 0.5 * (self.Bph[ng:-ng, ng:-ng, ng:-ng] + 
                            self.Bph[ng:-ng, ng:-ng, ng+1:-ng+1])
    
    def generate_temperature(self):
        """ Formula taken from Appendix A of Zhao et al. (2018) """
        rho_cr = 1e-13
        csound = 1.88e-4
        mu = 2.36
        T0 = csound**2 * mu * c.m_p.cgs.value / c.k_B.cgs.value
        T1 = T0 + 1.5 * self.rho/rho_cr
        T2 = np.where(self.rho >= 10*rho_cr, (T0+15) * \
            (self.rho/rho_cr/10)**0.6, T1)
        T3 = np.where(self.rho >= 100*rho_cr, 10**0.6 * (T0+15) * \
            (self.rho/rho_cr/100)**0.44, T2)
        self.temp = T3

    def LH_to_Gaussian(self):
        self.Br *= np.sqrt(4 * np.pi)
        self.Bth *= np.sqrt(4 * np.pi)
        self.Bph *= np.sqrt(4 * np.pi)
    
    def generate_cartesian(self):
        """ Convert spherical coordinates and vector components to cartesian """

        self.cordsystem = 1 

        # Create a coordinate grid.
        r, th, ph = np.meshgrid(self.r, self.th, self.ph, indexing='ij')

        # Convert coordinates to cartesian
        self.x = r * np.cos(ph) * np.sin(th)
        self.y = r * np.sin(ph) * np.sin(th)
        self.z = r * np.cos(th)

        # Transform vector components to cartesian
        self.Vx = self.Vr * np.sin(th) * np.cos(ph) + \
            self.Vth * np.cos(th) * np.cos(ph) - self.Vph * np.sin(ph)
        self.Vy = self.Vr * np.sin(th) * np.sin(ph) + \
            self.Vth * np.cos(th) * np.sin(ph) + self.Vph * np.cos(ph)
        self.Vz = self.Vr * np.cos(th) - self.Vth * np.sin(th)

        self.Bx = self.Br * np.sin(th) * np.cos(ph) + \
            self.Bth * np.cos(th) * np.cos(ph) - self.Bph * np.sin(ph)
        self.By = self.Br * np.sin(th) * np.sin(ph) + \
            self.Bth * np.cos(th) * np.sin(ph) + self.Bph * np.cos(ph)
        self.Bz = self.Br * np.cos(th) - self.Bth * np.sin(th)

