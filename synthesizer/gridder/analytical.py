import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
from astropy.io import fits

from synthesizer.gridder.vector_field import VectorField
from synthesizer.gridder.custom_model import CustomModel
from synthesizer.gridder import models
from synthesizer import utils 

# Global physical constants (CGS)
m_H2 = (2.3 * (const.m_p + const.m_e).cgs.value)
G = const.G.cgs.value
kB = const.k_B.cgs.value

class AnalyticalModel():
    def __init__(self, model, bbox, ncells=100, g2d=100, temp=False, nspec=1, 
            csubl=0, sootline=300, rin=1, rout=200, rc=100, r0=30, h0=10, 
            alpha=1, flare=1, mdisk=0.001, r_rim=1, r_gap=100, w_gap=5, 
            dr_gap=1e-5, rho0=1.6e6*m_H2, rflat=2336, 
        ):
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
        self.rin = rin
        self.rout = rout
        self.rc = rc
        self.r0 = r0
        self.h0 = h0
        self.alpha = alpha
        self.flare = flare
        self.mdisk = mdisk
        self.r_rim = r_rim
        self.r_gap = r_gap
        self.w_gap = w_gap
        self.dr_gap = dr_gap
        self.rho0 = rho0
        self.rflat = rflat

        if bbox is None:
            # Set default half-box sizes for predefined models
            self.bbox = {
                'constant': 1 * u.au.to(u.cm), 
                'plaw': 8500 * u.au.to(u.cm),    
                'pcore': 0.1 * u.pc.to(u.cm), 
                'ppdisk': 200 * u.au.to(u.cm),
                'ppdisk-gap-rim': 200 * u.au.to(u.cm),
                'l1544': 5000 * u.au.to(u.cm),
                'filament': 1 * u.pc.to(u.cm), 
                'user': 100 * u.au.to(u.cm), 
            }.get(model, 100 * u.au.to(u.cm))
        else:
            self.bbox = bbox


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
        field = self.vfield
        self.plotmin = None
        self.plotmax = None

        # Custom user-defined model
        if self.model == 'user':
            if self.bbox == 100*u.au.to(u.cm):
                utils.print_(
                    'Using a default half-box size of 100 au. '+\
                    f'You can change it with --bbox')

            model = CustomModel(x, y, z, field)

        # Constant density  
        elif self.model == 'constant':          
            model = models.Constant(x, y, z, field)

        # Radial power-law 
        elif self.model == 'plaw':
            model = models.PowerLaw(x, y, z, field,
                        self.rc, self.alpha, self.rho0)

        # Prestellar Core 
        elif self.model == 'pcore':
            model = models.PrestellarCore(x, y, z, field, self.rc)
            
        # L1544 Prestellar Core 
        elif self.model == 'l1544':
            model = models.L1544(x, y, z, field, self.rc, self.alpha, self.rho0)

        # Protoplanetary Disk
        elif self.model == 'ppdisk':
            model = models.PPdisk(x, y, z, field, self.rin, self.rout, 
                        self.rc, self.r0, self.h0, self.alpha, self.flare, 
                        self.mdisk)

        # Protoplanetary Disk with a gap and inner rim
        elif self.model == 'ppdisk-gap-rim':
            model = models.PPdiskGapRim(x, y, z, field, self.rin, 
                        self.rout, self.rc, self.r0, self.h0, self.alpha, 
                        self.flare, self.mdisk, self.r_rim, self.r_gap, 
                        self.w_gap, self.dr_gap)

        # Gravitationally unstable disk
        elif self.model == 'gidisk':
            utils.not_implemented(f'Model: {self.model}')
            model = models.GIdisk(x, y, z, field)

        # Disk with spiral arms
        elif self.model == 'spiral-disk':
            utils.not_implemented(f'Model: {self.model}')
            model = models.SpiralDisk(x, y, z, field)

        # Gas filament 
        elif self.model == 'filament':
            utils.not_implemented(f'Model: {self.model}')
            model = models.Filament(x, y, z, field)

        # PPDisk: HL Tau
        elif self.model == 'hltau':
            utils.not_implemented(f'Model: {self.model}')
            model = models.HLTau(x, y, z, field)
    
        # PPDisk: TW Hya
        elif self.model == 'twhya':
            #utils.not_implemented(f'Model: {self.model}')
            model = models.TWHya(x, y, z, field)
    
        # PPDisk: HD 163296
        elif self.model == 'hd16293':
            utils.not_implemented(f'Model: {self.model}')
            model = models.HD16293(x, y, z, field)

        # PPDisk: IM Lup
        elif self.model == 'imlup':
            utils.not_implemented(f'Model: {self.model}')
            model = models.IMLup(x, y, z, field)

        # PPDisk: WaOph 6
        elif self.model == 'waoph6':
            utils.not_implemented(f'Model: {self.model}')
            model = models.WaOph6(x, y, z, field)

        # PPDisk: Elias 27
        elif self.model == 'elias27':
            utils.not_implemented(f'Model: {self.model}')
            model = models.Elias27(x, y, z, field)

        # PPDisk: Elias 26
        elif self.model == 'elias26':
            utils.not_implemented(f'Model: {self.model}')
            model = models.Elias26(x, y, z, field)

        # PPDisk: AS 209
        elif self.model == 'as209':
            utils.not_implemented(f'Model: {self.model}')
            model = models.AS209(x, y, z, field)

        # PPDisk: GW Lup
        elif self.model == 'gwlup':
            utils.not_implemented(f'Model: {self.model}')
            model = models.GWLup(x, y, z, field)

        # PPDisk: HD 143006
        elif self.model == 'hd143006':
            utils.not_implemented(f'Model: {self.model}')
            model = models.HD143006(x, y, z, field)

        # PPDisk: HT Lup
        elif self.model == 'htlup':
            utils.not_implemented(f'Model: {self.model}')
            model = models.htlup(x, y, z, field)

        # PPDisk: AS 205
        elif self.model == 'as205':
            utils.not_implemented(f'Model: {self.model}')
            model = models.AS205(x, y, z, field)

        # PPDisk: HH 212
        elif self.model == 'hh212':
            utils.not_implemented(f'Model: {self.model}')
            model = models.HH212(x, y, z, field)

        else:
            raise ValueError(
                f'{utils.color.red}' +\
                f'Model: {self.model} cannot be found.' +\
                f'{utils.color.none}')

        # Copy the model density and temperature to the current obj (anaytical)
        self.dens = model.dens / self.g2d
        if self.add_temp: self.temp = model.temp
        if self.vfield is not None: self.vfield = model.vfield                
        if model.plotmin is not None: self.plotmin = model.plotmin
        if model.plotmax is not None: self.plotmax = model.plotmax


    def write_grid_file(self, regular=True):
        """ Write the regular cartesian grid file """
        with open('amr_grid.inp','w+') as f:
            # iformat
            f.write('1\n')
            # Regular grid
            f.write('0\n' if regular else '1\n')
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

    def plot_2d(self, field, data=None, cmap=None):
        """ Plot the density midplane at z=0 using Matplotlib """
        try:
            from matplotlib.colors import LogNorm
            utils.print_(f'Plotting the {field} grid midplane')
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.top'] = True
            plt.rcParams['ytick.right'] = True
            plt.rcParams['xtick.minor.visible'] = True
            plt.rcParams['ytick.minor.visible'] = True
            plt.close('all')

            # Detect what field to use
            if data is None:
                data = {
                    'density': self.dens, 
                    'temperature': self.temp
                }[field]
            else:
                data = data.T
            
            # Set the plot title for the right field
            title = {
                'density': r'Dust Density (g cm$^{-3}$)', 
                'temperature': r'Dust Temperature (K)',
            }[field]

            # Set the bbox if existent
            if self.bbox is None:
                extent = self.bbox
            else:
                bbox = self.bbox * u.cm.to(u.au)
                extent = [-bbox, bbox] * 2
        
            # Extract the middle plane 
            plane = self.ncells//2 - 1

            data_xy = data[:, :, plane].T
            data_xz = data[:, plane, :].T

            if np.all(data_xy == np.zeros(data_xy.shape)): 
                raise ValueError(f'{field} midplane is exactly 0.')

            if data_xy.min() > 0:
                if field == 'density' and self.plotmin is not None:
                    vmin = self.plotmin
                else:
                    vmin = data_xy.min() 
            else:
                vmin = None

            if data_xy.max() < np.inf:
                if field == 'density' and self.plotmax is not None:
                    vmax = self.plotmax
                else:
                    vmax = data_xy.max() 
            else:
                vmax = None

            fig, p = plt.subplots(nrows=1, ncols=2, figsize=(10, 5.5))

            # Set the colormap in case it was not given
            if cmap is None:
                cmap = 'BuPu' if field == 'density' else 'inferno'

            pxy = p[0].imshow(
                data_xy, 
                norm=LogNorm(vmin=vmin, vmax=vmax), 
                cmap=cmap,
                extent=extent,
            )
            pxz = p[1].imshow(
                data_xz, 
                norm=LogNorm(vmin=vmin, vmax=vmax), 
                cmap=cmap,
                extent=extent,
            )
            cxy = fig.colorbar(pxy, ax=p[0], pad=0.01, orientation='horizontal')
            cxz = fig.colorbar(pxy, ax=p[1], pad=0.01, orientation='horizontal')
            cxy.set_label(title)
            cxz.set_label(title)
            p[0].set_title('Midplane XY')
            p[1].set_title('Midplane XZ')
            p[0].set_ylabel('Y (AU)')
            p[1].set_ylabel('Z (AU)')
            p[0].set_xticklabels([])
            p[1].set_xticklabels([])

            try:
                if self.vfield is not None and field == 'density':
                    p[0].streamplot(
                        np.linspace(-bbox, bbox, self.ncells), 
                        np.linspace(-bbox, bbox, self.ncells), 
                        self.vfield.vx[..., plane].T, 
                        self.vfield.vy[..., plane].T, 
                        linewidth=0.5,
                        color='gray', 
                        density=0.7, 
                        arrowsize=0.1, 
                    )
                    p[1].streamplot(
                        np.linspace(-bbox, bbox, self.ncells), 
                        np.linspace(-bbox, bbox, self.ncells), 
                        self.vfield.vx[:, plane, :].T, 
                        self.vfield.vz[:, plane, :].T, 
                        linewidth=0.5,
                        color='gray', 
                        density=0.7, 
                        arrowsize=0.1, 
                    )
            except Exception as e:
                utils.print_('Unable to add vector stream lines',  red=True)
                utils.print_(e, bold=True)
            
            plt.tight_layout()
            plt.show()

        except Exception as e:
            utils.print_('Unable to show the 2D grid slice.',  red=True)
            utils.print_(e, bold=True)

        # Write maps to FITS files
        utils.write_fits(
            f'{field}_midplane.fits', 
            data=np.array([data_xy, data_xz]),
            header=fits.Header({
                'BTYPE': title.split('(')[0],
                'BUNIT': title.split('(')[1][:-1].replace('$',''),
                'CDELT1': 2 * bbox / self.ncells,
                'CRPIX1': self.ncells // 2,
                'CRVAL1': 0,
                'CUNIT1': 'AU',
                'CDELT2': 2 * bbox / self.ncells,
                'CRPIX2': self.ncells // 2,
                'CRVAL2': 0,
                'CUNIT2': 'AU',
            }),
            overwrite=True,
            verbose=True,
        )

    def plot_3d(self, field, data=None, tau=False, cmap=None): 
        """ Render the interpolated 3D field using Mayavi """
        try:
            from mayavi import mlab
            from mayavi.api import Engine
            from mayavi.modules.text3d import Text3D
            from mayavi.modules.grid_plane import GridPlane

            utils.print_('Visualizing the interpolated field ...')

            # Detect what field to use
            if data is None:
                data = {
                    'density': self.dens, 
                    'temperature': self.temp
                }[field]
            else:
                data = data.T
            
            # Set the plot title for the right field
            title = {
                'density': r'Dust Density (g cm$^{-3}$)', 
                'temperature': r'Gas Temperature (K)',
            }[field]

            # Initialize the figure and scene
            engine = Engine()
            engine.start()
            fig = mlab.figure(
                size=(1100, 1000),  bgcolor=(1, 1, 1),  fgcolor=(0.2, 0.2, 0.2))

            # Set the colormap in case it was given
            if cmap is None: cmap = 'inferno'


            # Render data
            plot = mlab.contour3d(
                data, contours=20, opacity=0.2, colormap=cmap)

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

            # Add an optional tau = 1 surface
            if tau:
                utils.print_('Adding optical depth surface at tau = 1')
                dl = bbox * u.au.to(u.cm) * 2 / data.shape[0]
                kappa = 1
                tau1 = np.cumsum(data * kappa * dl, axis=0).T

                if tau1.max() < 1:
                    utils.print_(
                        f'The highest optical depth is tau = {tau1.max()}. ' +
                        'No tau = 1 surface will be displayed.')
                else:
                    tausurf = mlab.contour3d(
                        tau1, contours=[1], opacity=0.5, color=(0, 0, 1))

            utils.print_('HINT: If you wanna play with the figure, press '+\
                'the nice icon in the upper left corner.', blue=True)
            utils.print_(
                "      Try, IsoSurface -> Actor -> Representation = Wireframe. ",
                blue=True)
            utils.print_(
                "      If you don't see much, it's probably a matter of "+\
                "adjusting the contours. ", blue=True)

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

        # Make sure ParaView is installed and callable
        utils.which('paraview')
    
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


