import numpy as np
import astropy.units as u
import astropy.constants as const
from abc import ABC, abstractmethod

from synthesizer import utils
from synthesizer.gridder.vector_field import VectorField

# Global physical constants (CGS)
m_H2 = (2.3 * (const.m_p + const.m_e).cgs.value)
G = const.G.cgs.value
kB = const.k_B.cgs.value

class BaseModel(ABC):
    """ 
        This is an Astract Base class for the rest of the models below.
        All following objects should inherit from BaseModel, to ensure 
        self.dens, self.temp and self.vfield are set.
    """

    def __init__(self, x, y, z, field='z'):
        self.x = x
        self.y = y
        self.z = z
        self.plotmin = None
        self.plotmax = None
        self.vfield = VectorField(x, y, z, morphology=field)
        self.r_c = x.max()

    @property
    @abstractmethod
    def dens(self):
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
    def vfield(self):
        return self._vfield
        
    @vfield.setter
    def vfield(self, vfield):
        self._vfield = vfield
        if type(vfield) != VectorField:
            raise ValueError(
                f'vfield must be a VectorField object, not a {type(vfield)}.')
    

class Constant(BaseModel):
    """ Box with a constant density """

    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        return 1e-12 * np.ones(self.x.shape)

    @property
    def temp(self):
        return 15 * np.ones(self.dens.shape)

class PowerLaw(BaseModel):
    """ Radial Power Law density distribution """

    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.x
        y = self.y
        z = self.z
        slope = 2
        rho_c = 9e-18
        self.r = np.sqrt(x**2 + y**2 + z**2)
        self.plotmin = 1e-19
        return rho_c * (self.r / self.r_c)**(-slope)

    @property
    def temp(self):
        # Radial temperature profile for a massive star with Teff = 35000 K
        return 35000 * 0.5**-0.25 * np.sqrt(0.5 * 2.3e13 / self.r)

class PrestellarCore(BaseModel):
    """ Prestellar Core: Bonnort-Eber sphere """
    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.x
        y = self.y
        z = self.z
        rho_c = 1.07e-15
        r = np.sqrt(x**2 + y**2 + z**2)
        return rho_c * self.r_c**2 / (r**2 + self.r_c**2)

    @property
    def temp(self):
        return 15 * np.ones(self.dens.shape)
        
class L1544(BaseModel):
    """ L1544 Prestellar Core (Chacon-Tanarro et al. 2019) """
    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.x
        y = self.y
        z = self.z
        rho_0 = 1.6e6 * m_H2
        alpha = 2.6
        r_flat = 2336 * u.au.to(u.cm)
        r = np.sqrt(x**2 + y**2 + z**2)
        return rho_0 / (1 + (r / r_flat)**alpha)

    @property
    def temp(self):
        return 15 * np.ones(self.dens.shape)

class PPdisk(BaseModel):
    """ Protoplanetary disk with a gap and soft inner rim """

    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        # Protoplanetary disk model 
        x = self.x
        y = self.y
        z = self.z
        rin = 1 * u.au.to(u.cm)
        h_0 = 0.1 
        rho_slope = 0.9
        flaring = 0.0
        Mdisk = 5e-3 * u.Msun.to(u.g)
        Mstar = 1 * u.Msun.to(u.g)
        rho_bg = 1e-30
        self.plotmin = rho_bg
        T_0 = 30
        T_slope = 3 / 7

        # Inner rim and gap parameters
        rim_rout = 0
        srim_rout = 0.8
        rim_slope = 0
        sigma_gap = 5 * u.au.to(u.cm)
        rin_gap = 10 * u.au.to(u.cm)
        rout_gap = 50 * u.au.to(u.cm)
        gap_c = 100 * u.au.to(u.cm)
        gap_stddev = 5 * u.au.to(u.cm)
        densredu_gap = 1e-7

        # Surface density
        r = np.sqrt(x**2 + y**2)
        r3d = np.sqrt(x**2 + y**2 + z**2)
        self.T_r = T_0 * (r3d/self.r_c)**-T_slope
        c_s = np.sqrt(kB * self.T_r / m_H2)
        v_K = np.sqrt(G * Mstar / r**3)
        h = c_s / v_K * (r/self.r_c)**flaring
        sigma_0 = (2 - rho_slope) * (Mdisk / 2 / np.pi / self.r_c**2)
        sigma_g = sigma_0 * (r/self.r_c)**-rho_slope * \
            np.exp(-(r/self.r_c)**(2-rho_slope))

        # Add a smooth inner rim
        if rim_rout > 0:
            h_rin = np.sqrt(kB * T_0 * (rin/self.r_c)**-T_slope / m_H2) / \
                np.sqrt(G * Mstar / rin**3)
            a = h_0 * (r/self.r_c)**rho_slope
            b = h_0 * (rim_rout * rin / h_0)**rho_slope
            c = h_rin * (r/rin)**(np.log10(b / h_rin) / np.log10(rim_rout))
            h = (a**8 + c**8)**(1/8) * r
            sigma_rim = sigma_0 * (srim_rout*rin/rout)**-rho_slope * \
                np.exp(-(r/rout)**(2-rho_slope)) * \
                (r/srim_rout/rin)**rim_slope
            sigma_g = (sigma_g**-5 + sigma_rim**-5)**(1/-5)
        
        # Add a gap (as a radial gaussian density decrease)
        gap = np.exp(-0.5 * (r-gap_c)**2 / sigma_gap**2) * (1/densredu_gap-1)
        sigma_g /= (gap + 1)

        # Density profile from hydrostatic equilibrium
        rho_g = sigma_g / np.sqrt(2*np.pi) / h * np.exp(-z*z / (2*h*h))
        rho_g = rho_g + rho_bg

        return rho_g

    @property
    def temp(self):
        # Radial temperature profile
        return self.T_r

class GIdisk(BaseModel):
    """
     Viscous gravitationally unstable accretion disk
     E. Vorobyov's suggesiton: 
     For the surface density, take a look at Galactic Dynamics (Biney...)
     For the scale height, Vorobyov & Basu 2008 or Rafikov 2009, 2015
    """

    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        utils.not_implemented(f'Model: GIdisk')
        pass

    @property
    def temp(self):
        pass

class SpiralDisk(BaseModel):
    """ PP Disk with logarithmic spiral arms (Huang et al. 2018b,c) """

    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        utils.not_implemented(f'Model: SpiralDisk')
        x = self.x
        y = self.y
        z = self.z
        theta = np.tan(0.23)
        r_theta = self.r_c * np.exp(b*theta)

    @property
    def temp(self):
        pass

class Filament(BaseModel):
    """
     Gas filament, modelled as a Plummer distribution (Arzoumanian+ 2011)
     Or from Koertgen+2018 and Tasker and Tan 2009
    """

    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        utils.not_implemented(f'Model: Filament')
        x = self.x
        y = self.y
        z = self.z
        p = 2
        r_flat = 0.03 * u.pc.to(u.cm) 
        r = np.sqrt(x**2 + y**2)
        rho_ridge = 4e-19
        return rho_ridge / (1 + (r/r_flat)**2)**(p/2)

    @property
    def temp(self):
        return 15 * np.ones(self.temp.shape)
