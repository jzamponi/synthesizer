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
    

# General physical models

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

    def __init__(self, x, y, z, field, rc, alpha, rho0):
        super().__init__(x, y, z, field)
        self.rc = 8500 * u.au.to(u.cm) if rc is None else rc
        self.rho0 = 9e-18 if rho0 is None else rho0
        self.alpha = 2 if alpha is None else alpha

    @property
    def dens(self):
        x = self.x
        y = self.y
        z = self.z
        self.r = np.sqrt(x**2 + y**2 + z**2)
        self.plotmin = 1e-19
        return self.rho0 * (self.r / self.rc)**(-self.alpha)

    @property
    def temp(self):
        # Radial temperature profile for a massive star with Teff = 35000 K
        return 35000 * 0.5**-0.25 * np.sqrt(0.5 * 2.3e13 / self.r)

class PrestellarCore(BaseModel):
    """ Prestellar Core: Bonnort-Eber sphere """
    def __init__(self, x, y, z, field, rc):
        super().__init__(x, y, z, field)
    
        self.rc = 0.1 * u.pc.to(u.cm) if rc is None else rc

    @property
    def dens(self):
        x = self.x
        y = self.y
        z = self.z
        rho_c = 1.07e-15
        r = np.sqrt(x**2 + y**2 + z**2)
        return rho_c * self.rc**2 / (r**2 + self.rc**2)

    @property
    def temp(self):
        return 15 * np.ones(self.dens.shape)
        
class L1544(BaseModel):
    """ L1544 Prestellar Core (Chacon-Tanarro et al. 2019) """
    def __init__(self, x, y, z, field, rc, alpha, rho0):
        super().__init__(x, y, z, field)

        self.rho0 = 1.6e6 * m_H2 if rho0 is None else rho0
        self.alpha = 2.6 if alpha is None else alpha
        self.rc = 2336 * u.au.to(u.cm) if rc is None else rc

    @property
    def dens(self):
        x = self.x
        y = self.y
        z = self.z
        r = np.sqrt(x**2 + y**2 + z**2)
        return self.rho0 / (1 + (r / self.rc)**self.alpha)

    @property
    def temp(self):
        return 15 * np.ones(self.dens.shape)

class PPdisk(BaseModel):
    """ 
        Protoplanetary disk 
        Lynden-Bell & Pringle (1974) and Hartmann et al. (1998)
        Equations can be found in Ballering et al. (2019) and many others.
    """

    def __init__(self, x, y, z, field, rin, rout, rc, r0, h0, alpha, flare, 
                mdisk):
        super().__init__(x, y, z, field)

        # Default model parameters
        self.rin = 1 * u.au.to(u.cm) if rin is None else rin
        self.rout = 200 * u.au.to(u.cm) if rout is None else rout
        self.rc = 140 * u.au.to(u.cm) if rc is None else rc
        self.h0 = 5 * u.au.to(u.cm) if h0 is None else h0
        self.r0 = 5 * u.au.to(u.cm) if r0 is None else r0
        self.alpha = 5 * u.au.to(u.cm) if alpha is None else alpha
        self.flare = 0.3 if flare is None else flare
        self.mdisk = 1e-1 * u.Msun.to(u.g) if mdisk is None else mdisk

    @property
    def dens(self):

        x = self.x
        y = self.y
        z = self.z
        r = np.sqrt(x**2 + y**2)
        r3d = np.sqrt(x**2 + y**2 + z**2)

        Mstar = 0 * u.Msun.to(u.g)
        rho_bg = 1e-30
        self.plotmin = rho_bg

        # Temperature profile
        T_0 = 30
        T_slope = 3 / 7
        self.T_r = T_0 * (r3d/self.rc)**-T_slope

        # Surface density
        exp_r = np.exp(-(self.rout/self.rc)**(2-alpha)) - \
                np.exp(-(self.rin/self.rc)**(2-alpha))

        sigma_0 = (alpha - 2) * (self.mdisk / 2 / np.pi / self.rc**2) / exp_r
        sigma_g = sigma_0 * (r/self.rc)**-alpha * \
                np.exp(-(r/self.rc)**(2-alpha))

        # If Mstar is set, set scale height from s. speed and kep. vel.
        if Mstar > 0:
            c_s = np.sqrt(kB * self.T_r / m_H2)
            v_K = np.sqrt(G * Mstar / r**3)
            h0 = c_s / v_K 

        # Scale height & Flaring 
        h = self.h0 * (r/self.r0)**self.flare

        # Gas density profile 
        rho_g = sigma_g / np.sqrt(2*np.pi) / h * np.exp(-z*z / (2*h*h))

        # Add a background gas density
        rho_g = rho_g + rho_bg
        rho_g[r3d < self.rin] = rho_bg
        rho_g[r3d > self.rout] = rho_bg

        # Trim temperature to region with background density
        self.T_r[rho_g == rho_bg] = 2.73

        return rho_g

    @property
    def temp(self):
        # Radial temperature profile
        return self.T_r


class PPdiskGapRim(BaseModel):

    """ Protoplanetary disk with a gap and soft inner rim """

    def __init__(self, x, y, z, field, rin, rout, rc, r0, h0, alpha, flare, 
                mdisk, r_rim, r_gap, w_gap, dr_gap):
        super().__init__(x, y, z, field)

        # Default model parameters
        self.rin = 1 * u.au.to(u.cm) if rin is None else rin
        self.rout = 200 * u.au.to(u.cm) if rout is None else rout
        self.rc = 140 * u.au.to(u.cm) if rc is None else rc
        self.h0 = 5 * u.au.to(u.cm) if h0 is None else h0
        self.r0 = 30 * u.au.to(u.cm) if r0 is None else r0
        self.alpha = 1 if alpha is None else alpha
        self.flare = 1 if flare is None else flare
        self.mdisk = 1e-3 * u.Msun.to(u.g) if mdisk is None else mdisk
        self.r_rim = 1 * u.au.to(u.cm) if r_rim is None else r_rim
        self.r_gap = 100 * u.au.to(u.cm) if r_gap is None else r_gap
        self.w_gap = 5 * u.au.to(u.cm) if w_gap is None else w_gap
        self.dr_gap = 1e-5 if dr_gap is None else dr_gap

    @property
    def dens(self):
        # Protoplanetary disk model 

        x = self.x
        y = self.y
        z = self.z
        r = np.sqrt(x**2 + y**2)
        r3d = np.sqrt(x**2 + y**2 + z**2)

        Mstar = 2 * u.Msun.to(u.g)
        rho_bg = 1e-30
        self.plotmin = rho_bg

        # Temperature profile
        T_0 = 30
        T_slope = 3 / 7
        self.T_r = T_0 * (r3d/self.rc)**-T_slope

        # Surface density
        exp_r = np.exp(-(self.rout/self.rc)**(2-self.alpha)) - \
                np.exp(-(self.rin/self.rc)**(2-self.alpha))

        sigma_0 = (self.alpha - 2) * (self.mdisk/2/np.pi/self.rc**2) / exp_r
        sigma_g = sigma_0 * (r/self.rc)**-self.alpha * \
                np.exp(-(r/self.rc)**(2-self.alpha))

        # If Mstar is set, set scale height from s. speed and kep. vel.
        if Mstar > 0:
            c_s = np.sqrt(kB * self.T_r / m_H2)
            v_K = np.sqrt(G * Mstar / r**3)
            h0 = c_s / v_K 

        # Scale height & Flaring 
        h = self.h0 * (r/self.r0)**self.flare

        # Gas density profile 
        rho_g = sigma_g / np.sqrt(2*np.pi) / h * np.exp(-z*z / (2*h*h))

        # Add a background gas density
        rho_g = rho_g + rho_bg
        rho_g[r3d < self.rin] = rho_bg
        rho_g[r3d > self.rout] = rho_bg

        # Trim temperature to region with background density
        self.T_r[rho_g == rho_bg] = 2.73

        # Additional inner rim and gap parameters
        srim_rout = 0.5
        rim_slope = 2

        # Add a smooth inner rim
        if self.r_rim > 0:
            h_rin = np.sqrt(kB * T_0 * (self.rin/self.rc)**-T_slope / m_H2) / \
                np.sqrt(G * Mstar / self.rin**3)
            a = self.h0 * (r/self.rc)**self.alpha
            b = self.h0 * (self.r_rim * self.rin / self.h0)**self.alpha
            c = h_rin * (r/self.rin)**(np.log10(b / h_rin) / \
                np.log10(self.r_rim))
            h = (a**8 + c**8)**(1/8) #* r
            sigma_rim = sigma_0 * (srim_rout*self.rin/self.rout)**-self.alpha * \
                np.exp(-(r/self.rout)**(2-self.alpha)) * \
                (r/srim_rout/self.rin)**rim_slope
            sigma_g = (sigma_g**-5 + sigma_rim**-5)**(1/-5)
        
        # Add a gap (as a radial gaussian density decrease)
        gap = np.exp(-0.5 * (r-self.r_gap)**2 / self.w_gap**2) * \
            (1/self.dr_gap-1)
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

     For a more recent model, follow the prescription from Yamamuro et al 2023.
    """

    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
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
        x = self.x
        y = self.y
        z = self.z
        theta = np.tan(0.23)
        r_theta = rc * np.exp(b*theta)

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


# Tailored models for specific sources

class HLTau(BaseModel):
    """
    Physical model for the protoplanetary disk HL Tay
    Kataoka et al (2016)
    """
    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.x
        y = self.y
        z = self.z

        return sigma / (np.sqrt(2*np.pi)*h_d) * np.exp(-z*z/2/h_d/h_d)

    @property
    def temp(self):
        T_0 = 250
        q = 0.5
        return T_0 * (r*u.cm.to(u.au))**-q
    
class TWHya(BaseModel):
    """
    Physical model for the protoplanetary disk TW Hydrae
    Andrews et al (2012, 2016)
    Hogerheijde et al (2016)
    Ueda et al (2020)
    """
    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.x
        y = self.y
        z = self.z

        r = np.sqrt(x*x + y*y)
        sigma_10 = 10
        sigma = sigma_10 * (r * u.cm.to(u.au) / 10)**(-0.5)
        h_d = 0.63 * (r * u.cm.to(u.au) / 10)**(1.1) 
        rho_g = 100 * sigma / (np.sqrt(2*np.pi)*h_d) * np.exp(-z**2 / 2 / h_d**2) 
        rho_g[r < 1 * u.au.to(u.cm)] = 0
        rho_g[r > 50 * u.au.to(u.cm)] = 0
        return rho_g

    @property
    def temp(self):
        T_10 = 30
        q = 0.4
        r = np.sqrt(self.x**2 + self.y**2)
        T_r = T_10 * (r*u.cm.to(u.au))**-q
        T_r[r < 1 * u.au.to(u.cm)] = 0
        T_r[r > 50 * u.au.to(u.cm)] = 0
        return T_r
    
class HD163296(BaseModel):
    """
    Physical model for the protoplanetary disk HD 163296
    Dullemond et al (2018) (DSHARP)
    Lin et al (2020b)
    """
    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.x
        y = self.y
        z = self.z

        return sigma / (np.sqrt(2*np.pi)*h_d) * np.exp(-z*z/2/h_d/h_d)

    @property
    def temp(self):
        T_0 = 250
        q = 0.5
        return T_0 * (r*u.cm.to(u.au))**-q

class IMLup(BaseModel):
    """
    Physical model for the protoplanetary disk IM Lup
    Huang et al (2018) (DSHARP)
    """
    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.x
        y = self.y
        z = self.z

        return sigma / (np.sqrt(2*np.pi)*h_d) * np.exp(-z*z/2/h_d/h_d)

    @property
    def temp(self):
        T_0 = 250
        q = 0.5
        return T_0 * (r*u.cm.to(u.au))**-q

class WaOph6(BaseModel):
    """
    Physical model for the protoplanetary disk WaOph6
    Huang et al (2018) (DSHARP)
    """
    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.X
        y = self.y
        z = self.z

        return sigma / (np.sqrt(2*np.pi)*h_d) * np.exp(-z*z/2/h_d/h_d)

    @property
    def temp(self):
        T_0 = 250
        q = 0.5
        return T_0 * (r*u.cm.to(u.au))**-q

class Elias27(BaseModel):
    """
    Physical model for the protoplanetary disk WaOph6
    Huang et al (2018) (DSHARP)
    """
    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.X
        y = self.y
        z = self.z

        return sigma / (np.sqrt(2*np.pi)*h_d) * np.exp(-z*z/2/h_d/h_d)

    @property
    def temp(self):
        T_0 = 250
        q = 0.5
        return T_0 * (r*u.cm.to(u.au))**-q

class AS209(BaseModel):
    """
    Physical model for the protoplanetary disk AS209
    Dullemond et al (2018) (DSHARP)
    """
    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.X
        y = self.y
        z = self.z

        return sigma / (np.sqrt(2*np.pi)*h_d) * np.exp(-z*z/2/h_d/h_d)

    @property
    def temp(self):
        # Equation 5
        T_0 = 250
        q = 0.5
        return T_0 * (r*u.cm.to(u.au))**-q

class GWLup(BaseModel):
    """
    Physical model for the protoplanetary disk GW Lup
    Dullemond et al (2018) (DSHARP)
    """
    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.X
        y = self.y
        z = self.z

        return sigma / (np.sqrt(2*np.pi)*h_d) * np.exp(-z*z/2/h_d/h_d)

    @property
    def temp(self):
        # Equation 5
        T_0 = 250
        q = 0.5
        return T_0 * (r*u.cm.to(u.au))**-q

class HD143006(BaseModel):
    """
    Physical model for the protoplanetary disk HD 143006
    Dullemond et al (2018) (DSHARP)
    """
    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.X
        y = self.y
        z = self.z

        return sigma / (np.sqrt(2*np.pi)*h_d) * np.exp(-z*z/2/h_d/h_d)

    @property
    def temp(self):
        # Equation 5
        T_0 = 250
        q = 0.5
        return T_0 * (r*u.cm.to(u.au))**-q

class HTLup(BaseModel):
    """
    Physical model for the protoplanetary disk HT Lup
    Kurtovic et al (2018) (DSHARP)
    """
    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.X
        y = self.y
        z = self.z

        return sigma / (np.sqrt(2*np.pi)*h_d) * np.exp(-z*z/2/h_d/h_d)

    @property
    def temp(self):
        # Equation 5
        T_0 = 250
        q = 0.5
        return T_0 * (r*u.cm.to(u.au))**-q

class AS205(BaseModel):
    """
    Physical model for the protoplanetary disk AS 205
    Kurtovic et al (2018) (DSHARP)
    """
    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.X
        y = self.y
        z = self.z

        return sigma / (np.sqrt(2*np.pi)*h_d) * np.exp(-z*z/2/h_d/h_d)

    @property
    def temp(self):
        # Equation 5
        T_0 = 250
        q = 0.5
        return T_0 * (r*u.cm.to(u.au))**-q

class HH212(BaseModel):
    """
    Physical model for the Class 0 protostellar disk HH 212
    Lin et al (2021a)
    """
    def __init__(self, x, y, z, field):
        super().__init__(x, y, z, field)

    @property
    def dens(self):
        x = self.X
        y = self.y
        z = self.z

        return sigma / (np.sqrt(2*np.pi)*h_d) * np.exp(-z*z/2/h_d/h_d)

    @property
    def temp(self):
        T_0 = 250
        q = 0.5
        return T_0 * (r*u.cm.to(u.au))**-q
