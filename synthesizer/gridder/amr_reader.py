import h5py
import numpy as np
import astropy.units as u
import astropy.constants as c

from synthesizer import utils

class Athena:
    """ 
        Object designed to read in spatial and physical quantities from 
        Athena++ snapshots.
    """
    def __init__(self):
        utils.not_implemented()

class Flash:
    """ 
        Object designed to read in spatial and physical quantities from 
        FLASH snapshots.
    """
    def __init__(self):
        utils.not_implemented()

class Enzo:
    """ 
        Object designed to read in spatial and physical quantities from 
        ENZO snapshots.
    """
    def __init__(self):
        utils.not_implemented()

class Ramses:
    """ 
        Object designed to read in spatial and physical quantities from 
        RAMSES snapshots.
    """
    def __init__(self):
        utils.not_implemented()

class ZeusTW:
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

