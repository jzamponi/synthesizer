import numpy as np
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt

from .vector_field import VectorField
from synthesizer import utils 

class AnalyticalModel():
    def __init__(self, model, bbox, ncells=100, g2d=100, temp=False, nspec=1, 
        csubl=0, sootline=300):
        """
        Create an analytical density model indexed by the variable model.
        All quantities should be treated in cgs unless explicitly converted.
        """

        self.dens = np.zeros((ncells, ncells, ncells))
        self.temp = np.zeros((ncells, ncells, ncells))
        self.vfield = None
        self.add_temp = temp
        self.model = model 
        self.ncells = ncells
        self.g2d = g2d
        self.nspec = nspec
        self.csubl = csubl
        self.sootline = sootline
        self.carbon = 0.375
        self.subl_mfrac = 1 - self.carbon * self.csubl / 100

        if bbox is None:
            raise ValueError(
                '--model requires a box half-size (au) to be given with --bbox')
        else:
            self.bbox = bbox * u.au.to(u.cm)

    def create_model(self):
        """ Setup a density model box """

        utils.print_(f'Creating density model: {self.model}')

        # Coordinate grid: cell walls and cell centers
        self.xw = np.linspace(-self.bbox, self.bbox, self.ncells + 1)
        self.yw = np.linspace(-self.bbox, self.bbox, self.ncells + 1)
        self.zw = np.linspace(-self.bbox, self.bbox, self.ncells + 1)
        xc = 0.5 * (self.xw[0: self.ncells] + self.xw[1: self.ncells + 1])
        yc = 0.5 * (self.yw[0: self.ncells] + self.yw[1: self.ncells + 1])
        zc = 0.5 * (self.zw[0: self.ncells] + self.zw[1: self.ncells + 1])
        x, y, z = np.meshgrid(xc, yc, zc, indexing='xy')
        self.X = x
        self.Y = y
        self.Z = z
        r_c = self.bbox
        self.plotmin = None
        self.plotmax = None

        # Constant density  
        if self.model == 'constant':          
            self.dens = 1e-12 * np.ones(self.dens.shape)
            if self.add_temp:
                self.temp = 15 * np.ones(self.temp.shape)

        # Radial power-law 
        elif self.model == 'plaw':
            slope = 2
            rho_c = 9e-18 / self.g2d
            r = np.sqrt(x**2 + y**2 + z**2)
            self.dens = rho_c * (r / r_c)**(-slope)
            if self.add_temp:
                self.temp = 35000 * 0.5**-0.25 * np.sqrt(0.5 * 2.3e13 / r)
            self.plotmin = 1e-19

        # Prestellar Core: Bonnort-Eber sphere
        elif self.model == 'pcore':
            rho_c = 1.07e-15
            r_c = 1100 * u.au.to(u.cm)
            r = np.sqrt(x**2 + y**2 + z**2)
            self.dens = rho_c * r_c**2 / (r**2 + r_c**2)
            self.plotmin = 1e-15
            
        # Flared Protoplanetary disk: Shakura & Sunyaev 1973
        elif self.model == 'ppdiski':
            rho_slope = 1.8
            flaring = 2.6
            r_0 = 100 * u.au.to(u.cm)
            h_0 = 10 * u.au.to(u.cm)
            rho_0 = 1e-12
            r = np.sqrt(x**2 + y**2)
            h = lambda r: h_0 * (r / r_0)**flaring
            rho_g = rho_0 * (r / r_0)**-rho_slope * np.exp(-0.5 * (z / h(r))**2)
            self.plotmin = 1e-16
    
        elif self.model == 'ppdisk':
            # From hydrostatic equilibrium
            sigma_g = 100 
            M_star = 2.4 * u.Msun.to(u.g)
            T_star = 1500 
            mu_g = 2.3 * (c.m_p + c.m_e).cgs.value
            kB = c.k_B.cgs.value
            G = c.G.cgs.value
            r = np.sqrt(x**2 + y**2)
            h = lambda r: np.sqrt(kB * T_star * r**3 / mu_g / G / M_star)
            rho_g = sigma_g / (2*np.pi)**0.5 / h(r) * np.exp(-0.5*(z / h(r))**2)
            self.dens = rho_g / self.g2d
            self.plotmin = 1e-16


    def write_grid_file(self):
        """ Write the regular cartesian grid file """
        with open('amr_grid.inp','w+') as f:
            # iformat
            f.write('1\n')
            # Regular grid
            f.write('0\n')
            # Coordinate system: cartesian
            f.write('0\n')
            # Gridinfo
            f.write('0\n')
            # Number of cells
            f.write('1 1 1\n')
            # Size of the grid
            f.write(f'{self.ncells:d} {self.ncells:d} {self.ncells:d}\n')

            # Write the cell walls
            for i in self.xw:
                f.write(f'{i:13.6e}\n')
            for j in self.yw:
                f.write(f'{j:13.6e}\n')
            for k in self.zw:
                f.write(f'{k:13.6e}\n')


    def write_density_file(self):
        """ Write the density file """
        utils.print_('Writing dust density file')

        # Flatten the array into a 1D fortran-style indexing
        density = self.dens.ravel(order='F')
        with open('dust_density.inp','w+') as f:
            f.write('1\n')
            f.write(f'{density.size:d}\n')
            f.write(f'{self.nspec}\n')

            if self.nspec == 1:
                # Write a single dust species
                for d in density:
                    f.write(f'{d:13.6e}\n')
            else:
                utils.print_(f'Writing two density species ...')
                # Write two densities: 
                # one with original value outside the sootline and zero within 
                temp1d = self.interp_temp.ravel(order='F')
                for i, d in enumerate(density):
                    if temp1d[i] < self.sootline:
                        f.write(f'{d:13.6e}\n')
                    else:
                        f.write('0\n')

                # one with zero outside the sootline and reduced density within
                for i, d in enumerate(density):
                    if temp1d[i] < self.sootline:
                        f.write('0\n')
                    else:
                        f.write(f'{(d * self.subl_mfrac):13.6e}\n')

    def write_temperature_file(self):
        """ Write the temperature file """
        utils.print_('Writing dust temperature file')

        temperature = self.temp.ravel(order='F')
        with open('dust_temperature.dat','w+') as f:
            f.write('1\n')
            f.write(f'{temperature.size:d}\n')
            f.write(f'{self.nspec}\n')

            # Write the temperature Nspec times for Nspec dust species
            for i in range(self.nspec):
                for t in temperature:
                    f.write(f'{t:13.6e}\n')

    def write_vector_field(self, morphology):
        """ Create a vector field for dust alignment """
 
        utils.print_('Writing grain alignment direction file')
 
        self.vfield = VectorField(self.X, self.Y, self.Z, morphology)
 
        with open('grainalign_dir.inp','w+') as f:
            f.write('1\n')
            f.write(f'{int(self.ncells**3)}\n')
            for iz in range(self.ncells):
                for iy in range(self.ncells):
                    for ix in range(self.ncells):
                        f.write(f'{self.vfield.vx[ix, iy, iz]:13.6e} ' +\
                                f'{self.vfield.vy[ix, iy, iz]:13.6e} ' +\
                                f'{self.vfield.vz[ix, iy, iz]:13.6e}\n')

    def plot_dens_midplane(self):
        """ Plot the density midplane at z=0 using Matplotlib """
        try:
            from matplotlib.colors import LogNorm
            utils.print_(f'Plotting the density grid midplane')
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.top'] = True
            plt.rcParams['ytick.right'] = True
            plt.rcParams['xtick.minor.visible'] = True
            plt.rcParams['ytick.minor.visible'] = True

            if self.bbox is None:
                extent = self.bbox
            else:
                bbox = self.bbox * u.cm.to(u.au)
                extent = [-bbox, bbox] * 2
        
            midplane = self.ncells//2 - 1
            if self.model == 'ppdiskp':
                slice_ = self.dens[:, midplane, :].T
            else:
                slice_ = self.dens[..., midplane].T

            if slice_.min() > 0:
                vmin = slice_.min() if self.plotmin is None else self.plotmin 
            else:
                vmin = None

            if slice_.max() < np.inf:
                vmin = slice_.max() if self.plotmin is None else self.plotmax 
            else:
                vmax = None

            plt.title(r'Dust Density Midplane (g cm$^-3$)')
            plt.imshow(slice_, 
                norm=LogNorm(vmin=vmin, vmax=vmax), 
                cmap='CMRmap',
                extent=extent,
            )
            plt.colorbar()
            plt.xlabel('AU')
            plt.ylabel('AU')

            if self.vfield is not None:
                if self.model == 'ppdisk':
                    v1 = self.vfield.vx[:, midplane, :].T, 
                    v2 = self.vfield.vz[:, midplane, :].T,
                else:
                    v1 = self.vfield.vx[..., midplane].T, 
                    v2 = self.vfield.vy[..., midplane].T,
                plt.streamplot(
                    np.linspace(-bbox, bbox, self.ncells), 
                    np.linspace(-bbox, bbox, self.ncells), 
                    v1, 
                    v2,
                    color='k', 
                    density=0.7,
                    arrowsize=0.1, 
                )
            plt.show()

        except Exception as e:
            utils.print_('Unable to show the 2D grid slice.',  red=True)
            utils.print_(e, bold=True)

    def plot_temp_midplane(self):
        """ Plot the temperature midplane at z=0 using Matplotlib """
        try:
            from matplotlib.colors import LogNorm
            utils.print_(f'Plotting the temperature grid midplane at z = 0')
            plt.rcParams['font.family'] = 'Times New Roman'

            if self.bbox is None:
                extent = self.bbox
            else:
                extent = [-self.bbox*u.cm.to(u.au), 
                        self.bbox*u.cm.to(u.au)] * 2

            plt.imshow(self.temp[..., self.ncells//2-1].T, 
                cmap='inferno',
                extent=extent,
            )
            plt.colorbar()
            plt.xlabel('AU')
            plt.ylabel('AU')
            plt.title(r'Temperature Midplane at $z=0$ (K)')
            plt.show()

        except Exception as e:
            utils.print_('Unable to show the 2D grid slice.',  red=True)
            utils.print_(e, bold=True)


    def plot_dens_3d(self): 
        """ Render the 3D field using Mayavi """
        try:
            utils.print_('Visualizing the 3D field ...')
            from mayavi import mlab
            fig = mlab.figure(
                size=(900, 700) , bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))

            mlab.contour3d(
                self.dens, vmin=self.plotmin, vmax=self.plotmax, opacity=0.7)

            if self.vfield is not None:
                # Add streamlines to mayavi
                utils.print_(
                    'Plotting 3D vector fields is yet to be implemented.')
                #mlab.flow(self.vfield.vx, self.vfield.vy, self.vfield.vz) 

            mlab.show()

        except Exception as e:
            utils.print_('Unable to show the 3D grid.',  red=True)
            utils.print_(e, bold=True)


    def plot_temp_3d(self): 
        """ Render the 3D field using Mayavi """
        try:
            utils.print_('Visualizing the 3D field ...')
            from mayavi import mlab
            mlab.contour3d(self.temp, contours=20, opacity=0.2)
            mlab.show()

        except Exception as e:
            utils.print_('Unable to show the 3D grid.',  red=True)
            utils.print_(e, bold=True)


    def create_vtk(self, dust_density=False, dust_temperature=True, rename=False):
        """ Call radmc3d to create a VTK file of the grid """
        self.radmc3d_banner()

        if dust_density:
            subprocess.run('radmc3d vtk_dust_density 1'.split())
            if rename:
                subprocess.run('mv model.vtk model_dust_density.vtk'.split())

        if dust_temperature:
            subprocess.run('radmc3d vtk_dust_temperature 1'.split())
            if rename:
                subprocess.run('mv model.vtk model_dust_temperature.vtk'.split())

        if not dust_density and not dust_temperature:
            subprocess.run('radmc3d vtk_grid'.split())

        self.radmc3d_banner()

    def render(self, state=None, dust_density=False, dust_temperature=True):
        """ Render the new grid in 3D using ParaView """
        if isinstance(state, str):
            subprocess.run(f'paraview --state {state} 2>/dev/null'.split())
        else:
            try:
                if dust_density:
                    subprocess.run(
                    f'paraview model_dust_density.vtk 2>/dev/null'.split())
                elif dust_temperature:
                    subprocess.run(
                    f'paraview model_dust_temperature.vtk 2>/dev/null'.split())
            except Exception as e:
                utils.print_('Unable to render using ParaView.',  bold=True)
                utils.print_(e, bold=True)


