import h5py
import numpy as np
import astropy.units as u
import astropy.constants as c

from synthesizer.gridder.vector_field import VectorField
from synthesizer import utils

class SPHng:
    """ Handle binary snapshots from the SPHng code. """
    def __init__(self, filename, remove_sink=True, cgs=True, verbose=True):
        """
            Notes:

            Assumes your binary file is formatted with a header stating the quantities,
            assumed to be f4 floats. May not be widely applicable.

            For these filenames, the header is of the form...

            # id t x y z vx vy vz mass hsml rho T u

            where:

            id = particle ID number
            t = simulation time [years]
            x, y, z = cartesian particle position [au]
            vx,vy,vz = cartesian particle velocity [need to check units]
            mass = particle mass [ignore]
            hmsl = particle smoothing length [ignore]
            rho = particle density [g/cm3]
            T = particle temperature [K]
            u = particle internal energy [ignore]
            
            outlier_index = 31330
        """

        try:
            # Read file in binary format
            with open(filename, "rb") as f:
                names = f.readline()[1:].split()
                data = np.frombuffer(f.read()).reshape(-1, len(names))
                data = data.astype("f4")
        except ValueError as e:
            utils.print_('Error trying to read SPHng binary file.', red=True)
            utils.print_(
                'File is probably from another code. Set --source', bold=True)
            raise ValueError(e)

        # Turn data into CGS 
        if cgs:
            data[:, 2:5] *= u.au.to(u.cm)
            data[:, 8] *= u.M_sun.to(u.g)

        # Remove the cell data of the outlier, probably a sink particle
        if remove_sink:
            sink_id = 31330
            data[sink_id, 5] = 0
            data[sink_id, 6] = 0
            data[sink_id, 7] = 0
            data[sink_id, 8] = data[:,8].min()
            data[sink_id, 9] = data[:,9].min()
            data[sink_id, 10] = 3e-11 
            data[sink_id, 11] = 900
        
        self.x = data[:, 2]
        self.y = data[:, 3]
        self.z = data[:, 4]
        self.rho_g = data[:, 10]
        self.temp = data[:, 11]

class Gizmo:
    """ Handle HDF5 snapshots from the GIZMO code. """
    def __init__(self, filename):

        # Read in particle coordinates and density
        data = h5py.File(filename, 'r')['PartType0']
        coords = np.array(data['Coordinates'])
        self.x = coords[:, 0] * u.pc.to(u.cm)
        self.y = coords[:, 1] * u.pc.to(u.cm)
        self.z = coords[:, 2] * u.pc.to(u.cm)
        self.rho_g = np.array(data['Density']) * 1.3816327e-23

        # Read in temperature if available
        if 'Temperature' in data:
            self.temp = np.array(data['Temperature'])
        # Or maybe the krome temperature, when coupled with KROME
        elif 'KromeTemperature' in data:
            self.temp = np.array(data['KromeTemperature'])
        
        # Or derive from a barotropic equation of state
        elif 'InternalEnergy' in data.keys() and 'Pressure' in data.keys():
            self.temp = data['InternalEnergy'] / np.array(data['Pressure'])

        else:
            self.temp = np.zeros(self.rho_g.shape)

        # Recenter the particles based on the center of mass
        self.x -= np.average(self.x, weights=self.rho_g)
        self.y -= np.average(self.y, weights=self.rho_g)
        self.z -= np.average(self.z, weights=self.rho_g)

class Gadget:
    """ Handle snapshots from the Gadget code. """
    def __init__(self, filename):
        # Read in particle coordinates and density
        data = h5py.File(filename, 'r')['PartType0']
        coords = np.array(data['Coordinates'])
        dens = np.array(data['Density'])
        self.x = coords[:, 0] * u.au.to(u.cm)
        self.y = coords[:, 1] * u.au.to(u.cm)
        self.z = coords[:, 2] * u.au.to(u.cm)
        self.rho_g = dens * 1e-10 * (u.Msun / u.au**3).to(u.g/u.cm**3)

        # Read in temperature if available
        if 'Temperature' in data.keys():
            self.temp = data['Temperature']

        # Or derive from a barotropic equation of state
        elif 'InternalEnergy' in data.keys() and 'Pressure' in data.keys():
            self.temp = data['InternalEnergy'] / np.array(data['Pressure'])

        else:
            self.temp = np.zeros(self.rho_g.shape)

class Arepo:
    """ Handle snapshots from the AREPO code. """
    def __init__(self, filename):

        # Read in particle coordinates and density
        data = h5py.File(filename, 'r')['PartType0']
        coords = np.array(data['Coordinates'])
        dens = np.array(data['Density'])
        self.x = coords[:, 0] * u.au.to(u.cm)
        self.y = coords[:, 1] * u.au.to(u.cm)
        self.z = coords[:, 2] * u.au.to(u.cm)
        self.rho_g = dens * 1e-10 * (u.Msun / u.au**3).to(u.g/u.cm**3)

        # Read in temperature if available
        if 'Temperature' in data.keys():
            self.temp = data['Temperature']

        # Or derive from a barotropic equation of state
        elif 'InternalEnergy' in data.keys() and 'Pressure' in data.keys():
            self.temp = data['InternalEnergy'] / np.array(data['Pressure'])

        else:
            self.temp = np.zeros(self.rho_g.shape)

class Phantom:
    """ Handle snapshots from the PHANTOM code. """
    def __init__(self, filename):
        
        # Read in particle coordinates and density
        data = h5py.File(filename, 'r')['particles']
        coords = data['xyz'].value
        # Derive density from the particle mass / h**3
        # src: https://github.com/dmentipl/plonk/blob/main/src/plonk/snap/readers/phantom.py
    
        dens = np.array(data['Density'])
        self.x = coords[:, 0] * u.au.to(u.cm)
        self.y = coords[:, 1] * u.au.to(u.cm)
        self.z = coords[:, 2] * u.au.to(u.cm)
        self.rho_g = dens * 1e-10 * (u.Msun / u.au**3).to(u.g/u.cm**3)