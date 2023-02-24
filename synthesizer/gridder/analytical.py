import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt

from synthesizer.gridder.vector_field import VectorField
from synthesizer.gridder.custom_model import CustomModel
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
            # Set default half-box sizes for predefined models
            self.bbox = {
                'constant': 1 * u.au.to(u.cm), 
                'plaw': 8500 * u.au.to(u.cm),    
                'pcore': 0.1 * u.pc.to(u.cm), 
                'ppdisk': 300 * u.au.to(u.cm),
                'l1544': 5000 * u.au.to(u.cm),
                'filament': 1 * u.pc.to(u.cm), 
                'user': 100 * u.au.to(u.cm), 
            }.get(model, 100 * u.au.to(u.cm))
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
        x, y, z = np.meshgrid(xc, yc, zc, indexing='ij')
        self.X = x
        self.Y = y
        self.Z = z
        r_c = self.bbox
        m_H2 = (2.3 * (const.m_p + const.m_e).cgs.value)
        G = const.G.cgs.value
        kB = const.k_B.cgs.value
        self.plotmin = None
        self.plotmax = None

        # Custom user-defined model
        if self.model == 'user':
            if self.bbox == 100*u.au.to(u.cm):
                utils.print_('Using a default half-box size of 100 au. '+\
                    f'You can change it with --bbox')

            model = CustomModel(x, y, z)
            self.dens = model.dens
            self.temp = model.temp
            self.vfield = model.vfield                

        # Constant density  
        elif self.model == 'constant':          
            self.dens = 1e-12 * np.ones(self.dens.shape)
            if self.add_temp:
                self.temp = 15 * np.ones(self.temp.shape)

        # Radial power-law 
        elif self.model == 'plaw':
            slope = 2
            rho_c = 9e-18 / self.g2d
            r = np.sqrt(x**2 + y**2 + z**2)
            self.dens = rho_c * (r / r_c)**(-slope)
            self.plotmin = 1e-19
            if self.add_temp:
                self.temp = 35000 * 0.5**-0.25 * np.sqrt(0.5 * 2.3e13 / r)

        # Prestellar Core: Bonnort-Eber sphere
        elif self.model == 'pcore':
            rho_c = 1.07e-15
            r = np.sqrt(x**2 + y**2 + z**2)
            self.dens = rho_c * r_c**2 / (r**2 + r_c**2)
            
        # L1544 Prestellar Core (Chacon-Tanarro et al. 2019)
        elif self.model == 'l1544':
            rho_0 = 1.6e6 * m_H2
            alpha = 2.6
            r_flat = 2336 * u.au.to(u.cm)
            r = np.sqrt(x**2 + y**2 + z**2)
            self.dens = rho_0 / (1 + (r / r_flat)**alpha)
            if self.add_temp:
                self.temp = 15 * np.ones(self.dens.shape)

        elif self.model == 'ppdisk':
            # Protoplanetary disk model 
            rin = 1 * u.au.to(u.cm)
            h_0 = 0.1 
            rho_slope = 0.9
            flaring = 0.0
            Mdisk = 5e-3 * u.Msun.to(u.g)
            Mstar = 1 * u.Msun.to(u.g)
            rho_bg = 1e-30
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
            T_r = T_0 * (r3d/r_c)**-T_slope
            c_s = np.sqrt(kB * T_r / m_H2)
            v_K = np.sqrt(G * Mstar / r**3)
            h = c_s / v_K * (r/r_c)**flaring
            sigma_0 = (2 - rho_slope) * (Mdisk / 2 / np.pi / r_c**2)
            sigma_g = sigma_0 * (r/r_c)**-rho_slope * \
                np.exp(-(r/r_c)**(2-rho_slope))

            # Add a smooth inner rim
            if rim_rout > 0:
                h_rin = np.sqrt(kB * T_0 * (rin/r_c)**-T_slope / m_H2) / \
                    np.sqrt(G * Mstar / rin**3)
                a = h_0 * (r/r_c)**rho_slope
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
            self.dens = rho_g / self.g2d
            self.plotmin = rho_bg

            # Radial temperature profile
            if self.add_temp:
                self.temp = T_r

        # Viscous gravitationally unstable accretion disk
        elif self.model == 'gidisk':
            utils.not_implemented(f'Model: {self.model}')
            # E. Vorobyov's suggesiton: 
            # For the surface density, take a look at Galactic Dynamics (Biney...)
            # For the scale height, Vorobyov & Basu 2008 or Rafikov 2009, 2015

        # PP Disk with logarithmic spiral arms (Huang et al. 2018b,c)
        elif self.model == 'spiral-disk':
            utils.not_implemented(f'Model: {self.model}')
            theta = np.tan(0.23)
            r_theta = r_c * np.exp(b*theta)

        # Gas filament, modelled as a Plummer distribution (Arzoumanian+ 2011)
        # Or from Koertgen+2018 and Tasker and Tan 2009
        elif self.model == 'filament':
            utils.not_implemented(f'Model: {self.model}')
            p = 2
            r_flat = 0.03 * u.pc.to(u.cm) 
            r = np.sqrt(x**2 + y**2)
            rho_ridge = 4e-19
            self.dens = rho_ridge / (1 + (r/r_flat)**2)**(p/2)
            if self.add_temp:
                self.temp = 15 * np.ones(self.temp.shape)


    def write_grid_file(self):
        """ Write the regular cartesian grid file """
        with open('amr_grid.inp','w+') as f:
            # iformat
            f.write('1\n')
            # Regular grid
            f.write('0\n')
            # Coordinate system: cartesian
            f.write('1\n')
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
 
        if self.model != 'user':
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
            plt.close('all')

            if self.bbox is None:
                extent = self.bbox
            else:
                bbox = self.bbox * u.cm.to(u.au)
                extent = [-bbox, bbox] * 2
        
            plane = self.ncells//2 - 1
            if self.model == 'ppdisk':
                slice_ = self.dens[:, plane, :].T
            else:
                slice_ = self.dens[..., plane].T

            if slice_.min() > 0:
                vmin = slice_.min() if self.plotmin is None else self.plotmin 
            else:
                vmin = None

            if slice_.max() < np.inf:
                vmax = slice_.max() if self.plotmax is None else self.plotmax 
            else:
                vmax = None

            if np.all(slice_ == np.zeros(slice_.shape)): 
                raise ValueError('Model density is exactly 0.')

            plt.title(r'Dust Density Midplane (g cm$^-3$)')
            plt.imshow(
                slice_, 
                norm=LogNorm(vmin=vmin, vmax=vmax), 
                cmap='CMRmap',
                extent=extent,
            )
            plt.colorbar()
            plt.xlabel('AU')
            plt.ylabel('AU')

            if self.vfield is not None:
                if self.model == 'ppdisk':
                    v1 = self.vfield.vx[:, plane, :].T, 
                    v2 = self.vfield.vz[:, plane, :].T,
                else:
                    v1 = self.vfield.vx[..., plane].T, 
                    v2 = self.vfield.vy[..., plane].T,

                try:
                    plt.streamplot(
                        np.linspace(-bbox, bbox, self.ncells), 
                        np.linspace(-bbox, bbox, self.ncells), 
                        np.array(v1).squeeze(), 
                        np.array(v2).squeeze(),
                        color='white', 
                        linewidth=0.5,
                        density=0.7,
                        arrowsize=0.1, 
                    )
                except Exception as e:
                    utils.print_('Unable to add vector stream lines',  red=True)
                    utils.print_(e, bold=True)
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
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.top'] = True
            plt.rcParams['ytick.right'] = True
            plt.rcParams['xtick.minor.visible'] = True
            plt.rcParams['ytick.minor.visible'] = True
            plt.close('all')

            if self.bbox is None:
                extent = self.bbox
            else:
                bbox = self.bbox * u.cm.to(u.au)
                extent = [-bbox, bbox] * 2

            plane = self.ncells//2 - 1
            if self.model == 'ppdisk':
                slice_ = self.temp[:, plane, :].T
            else:
                slice_ = self.temp[..., plane].T

            if np.all(slice_ == np.zeros(slice_.shape)): 
                raise ValueError('Model temperature is exactly 0.')
                
            vmin = slice_.min() if slice_.min() > 0 else None 
            vmax = slice_.max() if slice_.max() < np.inf else None 

            plt.imshow(slice_, 
                cmap='inferno',
                norm=LogNorm(vmin=vmin, vmax=vmax), 
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

    def plot_3d(self, density=False, temperature=False): 
        """ Render the interpolated 3D field using Mayavi """
        try:
            from mayavi import mlab
            from mayavi.api import Engine
            from mayavi.modules.text3d import Text3D
            from mayavi.modules.grid_plane import GridPlane

            utils.print_('Visualizing the interpolated field ...')

            if density:
                data = self.dens
                title = r'Dust Density (g cm$^{-3}$)'

            elif temperature:
                data = self.temp
                title = r'Gas Temperature (g cm$^{-3}$)'
                self.plotmin = None
            
            else:
                return False

            # Initialize the figure and scene
            engine = Engine()
            engine.start()
            fig = mlab.figure(
                size=(1100, 1000),  bgcolor=(1, 1, 1),  fgcolor=(0.2, 0.2, 0.2))

            # Render data
            plot = mlab.contour3d(
                data, contours=100, opacity=0.2, colormap='CMRmap',
                vmin=self.plotmin, vmax=self.plotmax)

            # Add a colorbar
            cbar = mlab.colorbar(plot, orientation='vertical', title=title)

            # Add a bounding box frame             
            bbox = int(np.round(self.bbox * u.cm.to(u.au), 1))
            mlab.outline(
                figure=fig, 
                extent=[0, self.ncells] * 3, 
            )
            mlab.axes(
                ranges=[-bbox, bbox] * 3,
                xlabel='AU', ylabel='AU', zlabel='AU', nb_labels=3
            )
        
            # Handle figure items
            manager = engine.scenes[0].children[0].children[0]

            # Customize the colobar
            lut = manager.scalar_lut_manager
            lut.title_text_property.italic = False
            lut.title_text_property.font_family = 'times'
            lut.title_text_property.font_size = 23

            # Paint one face of the box to represent the observing plane
            obs_plane = GridPlane()
            engine.add_filter(obs_plane, manager)
            obs_plane.grid_plane.axis = 'z'
            obs_plane.grid_plane.position = 0
            obs_plane.actor.property.representation = 'surface'
            obs_plane.actor.property.opacity = 0.2
            obs_plane.actor.property.color = (0.76, 0.72, 0.87)
            
            # Label the plane in 3D
            obs_label = Text3D()
            engine.add_filter(obs_label, manager)
            obs_label.text = 'Observer Plane'
            obs_label.actor.property.color = (0.76, 0.72, 0.87)
            obs_label.position = np.array([30, 15, 0])
            obs_label.scale = np.array([6, 6, 6])
            obs_label.orientation = np.array([0, 0, 90])
            obs_label.orient_to_camera = False

            utils.print_('HINT: If you wanna play with the figure, press '+\
                'the nice icon in the upper left corner.', blue=True)
            utils.print_(
                "Try, IsoSurface -> Actor -> Representation = Wireframe. ",
                blue=True)
            utils.print_(
                "If you don't see much, it's probably a matter of adjusting "+\
                "the contours. ", blue=True)

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


