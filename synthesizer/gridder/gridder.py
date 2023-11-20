import os, sys
import subprocess
import numpy as np
import astropy.units as u
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from synthesizer.gridder.vector_field import VectorField
from synthesizer.gridder.sph_reader  import *
from synthesizer.gridder.amr_reader  import *
from synthesizer import utils


class Grid():
    """ 
        Base class for different grid geometries.

        To Do: Abstract tecartesian and spherical grid object methods 
               using this base class
    """
    pass

class CartesianGrid(Grid):
    def __init__(self, ncells, bbox=None, rout=None, nspec=1, csubl=0, 
            sootline=300, g2d=100, temp=False, vfield=None):
        """ 
        Create a cartesian grid from a set of 3D points.

        The input SPH particle coordinates should be given as cartesian 
        coordinates in units of cm.

        The box can be trimmed to a given size calling self.trim_box. The 
        size can be given either as single half-length for a retangular box, 
        as cartesian vertices of a bounding box or as the radius of a sphere. 
        Any quantity should be given in units of cm.

        Examples: bbox = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]  
                  bbox = 50 
                  rout = 50

        """

        self.cordsystem = 1
        self.fx = 1
        self.fy = 1
        self.fz = 1
        self.nx = ncells
        self.ny = ncells
        self.nz = ncells
        self.ncells = ncells
        self.dens = np.zeros((self.nx, self.ny, self.nz))
        self.temp = np.zeros((self.nx, self.ny, self.nz))
        self.vx = np.zeros((self.nx, self.ny, self.nz))
        self.vy = np.zeros((self.nx, self.ny, self.nz))
        self.vz = np.zeros((self.nx, self.ny, self.nz))
        self.grid_dens = None
        self.grid_temp = None
        self.grid_vx = None
        self.grid_vy = None
        self.grid_vz = None
        self.vfield = vfield
        self.add_temp = temp
        self.bbox = bbox
        self.rout = rout
        self.g2d = g2d
        self.nspec = nspec
        self.csubl = csubl
        self.sootline = sootline
        self.carbon = 0.375
        self.subl_mfrac = 1 - self.carbon * self.csubl / 100

    def read_sph(self, filename, source='sphng'):
        """ Read SPH data. 
            Below are defined small interfaces to read in data from SPH codes.

            To add new code sources, synthesizer needs (all values in CGS):
                self.x, self.y, self.z, self.dens and self.temp (optional) 
         """

        # Download the script if a URL is provided
        if 'http' in filename: 
            utils.download_file(filename)
            filename = filename.split('/')[-1]

        utils.print_(
            f'Reading data from: {filename} | Format: {source.upper()}', end='')
        if not os.path.exists(filename): 
            raise FileNotFoundError(f'{utils.color.red}' +\
                f'Input SPH file does not exist{utils.color.none}')

        if source.lower() == 'sphng':
            self.sph = SPHng(filename, self.add_temp, self.vfield)

        elif source.lower() == 'gizmo':
            self.sph = Gizmo(filename, self.add_temp, self.vfield)

        elif source.lower() == 'gadget':
            self.sph = Gadget(filename, self.add_temp, self.vfield)

        elif source.lower() == 'arepo':
            self.sph = Arepo(filename, self.add_temp, self.vfield)

        elif source.lower() == 'phantom':
            utils.not_implemented()
            self.sph = Phantom(filename, self.add_temp, self.vfield)

        elif source.lower() == 'nbody6':
            self.sph = Nbody6(filename, self.add_temp, self.vfield)

        else:
            print('')
            raise ValueError(f'Source = {source} is currently not supported')
            
        self.x = self.sph.x
        self.y = self.sph.y
        self.z = self.sph.z
        self.dens = self.sph.rho_g / self.g2d
        self.npoints = len(self.dens)
        print(f' | Particles: {self.npoints}')

        if self.add_temp:
            self.temp = self.sph.temp
        else:
            self.temp = np.zeros(self.dens.shape)
        
        if self.vfield:
            self.vx = self.sph.vx
            self.vy = self.sph.vy
            self.vz = self.sph.vz

    def read_amr(self, filename, source='athena++'):
        """ Read AMR data """

        source = source.lower()

        # Download the script if a URL is provided
        if 'http' in filename: 
            utils.download_file(filename)
            filename = filename.split('/')[-1]

        utils.print_(f'Reading data from: {filename} | Format: {source}')
        if not os.path.exists(filename): 
            if source != 'zeustw':
                utils.file_exists(filename,
                    msg='Input AMR file does not exist')

        if source == 'athena++':
            utils.not_implemented()

        elif source == 'zeustw':
            # Generate a Data instance
            data = ZeusTW()

            # Set the coordinate system
            self.cordsystem = data.cordsystem

            # Read coordinates: x?a are cell edges and x?b are cell centers
            data.generate_coords(r="z_x1ap", th="z_x2ap", ph="z_x3ap")

            frame = str(filename).zfill(5)
            data.rho = data.read(f"o_d__{frame}")
            data.Br = data.read(f"o_b1_{frame}")
            data.Bth = data.read(f"o_b2_{frame}")
            data.Bph = data.read(f"o_b3_{frame}")
            data.Vr = data.read(f"o_v1_{frame}")
            data.Vth = data.read(f"o_v2_{frame}")
            data.Vph = data.read(f"o_v3_{frame}")

            data.trim_ghost_cells(field_type='coords', ng=3)
            data.trim_ghost_cells(field_type='scalar', ng=3)
            data.trim_ghost_cells(field_type='vector', ng=3)
            data.LH_to_Gaussian()
            data.generate_temperature()
            #data.generate_cartesian()

            self.xc = data.x
            self.yc = data.y
            self.zc = data.z
            self.nx = data.x.size
            self.ny = data.y.size
            self.nz = data.z.size
            self.grid_dens = data.rho * self.g2d
            self.grid_temp = data.temp

        elif source == 'flash':
            utils.not_implemented()

        elif source == 'enzo':
            utils.not_implemented()
            
        elif source == 'ramses':
            utils.not_implemented()
            
        else:
            raise ValueError(f'Source = {source} is currently not supported')

    def trim_box(self):
        """ 
        Trim the original grid to a given size, can be rectangular or spherical. 
        """

        # Emtpy dynamic array to store the indices of the particles to remove
        to_remove = []

        if self.bbox is not None:

            utils.print_(f'Deleting particles outside a box ' +
                f'half-length of {self.bbox * u.cm.to(u.au)} au')

            # Iterate over particles and delete upon reject
            for i in range(self.x.size):
                if self.x[i] < -self.bbox or self.x[i] > self.bbox:
                    to_remove.append(i)

            for j in range(self.y.size):
                if self.y[j] < -self.bbox or self.y[j] > self.bbox:
                    to_remove.append(j)

            for k in range(self.z.size):
                if self.z[k] < -self.bbox or self.z[k] > self.bbox:
                    to_remove.append(k)

        elif self.rout is not None:
            # Override any previous value of rout
            utils.print_('Deleting particles outside a radius of ' +
                f'{self.rout * u.cm.to(u.au)} au ...')

            # Convert cartesian to polar coordinates to define a radial trim
            r = np.sqrt(self.x**2 + self.y**2 + self.z**2)

            for i in range(self.x.size):
                if r[i] > self.rout:
                    to_remove.append(i)

        # Remove the particles from each quantity
        self.x = np.delete(self.x, to_remove)
        self.y = np.delete(self.y, to_remove)
        self.z = np.delete(self.z, to_remove)
        self.dens = np.delete(self.dens, to_remove)
        self.temp = np.delete(self.temp, to_remove)

        if self.vfield:
            self.vx = np.delete(self.vx, to_remove)
            self.vy = np.delete(self.vy, to_remove)
            self.vz = np.delete(self.vz, to_remove)

        utils.print_(f'Particles included: {self.x.size} | ' +
            f'Particles excluded: {self.npoints - self.x.size} ')

        if self.x.size == 0:
            raise ValueError(
                utils.color.red +
                'All particles were removed. Try increasing --bbox or --rout' +
                utils.color.none
            )

    def plot_particles(self):
        """ 
            Render the locations of the 3D points weighted by the 
            normalized density 
        """

        from mayavi import mlab

        utils.print_('Rendering particles in 3D weighted by density')

        fig = mlab.figure(
            size=(1100, 1000),  bgcolor=(1, 1, 1),  fgcolor=(0.2, 0.2, 0.2)
        )
        mlab.points3d(self.x, self.y, self.z, self.dens)
        mlab.show()

    def find_resolution(self):
        """
        Find the minimum distance between points. 
        If particles are too many, it uses only those that are 
        closer to the geometrical center than the mean distance.
        """
        
        from scipy.spatial import distance

        utils.print_('Calculating minimum distance between all particles')

        try:
            dmin = distance.pdist(np.vstack([self.x, self.y, self.z]).T).min()

        except MemoryError as e:
            # Use only the 10000 closest particles to the center
            n = 10000
            x = np.sort(np.abs(self.x))[-n:]
            y = np.sort(np.abs(self.y))[-n:]
            z = np.sort(np.abs(self.z))[-n:]
            dmin = distance.pdist(np.vstack([x, y, z]).T).min()

        if self.cellsize > self.resolution:
            utils.print_(
                f'The current cell size {self.cellsize*u.cm.to(u.au):.1}au '+\
                f'is larger than the minimum particle separation '+\
                f'{self.resolution*u.cm.to(u.au):.1}au. ', blue=True)
            utils.print_('You may want to increase ncells or use a bbox', 
                blue=True)

        return dmin

    def interpolate(self, field, method='linear', fill='min'):
        """
            Interpolate a set of points in cartesian coordinates along with their
            values into a rectangular grid.
        """

        # Construct the rectangular grid
        rmin = np.min([self.x.min(), self.y.min(), self.z.min()])
        rmax = np.max([self.x.max(), self.y.max(), self.z.max()])
        self.bbox = (rmax - rmin) / 2
        self.cellsize = self.bbox / self.ncells
        # self.resolution = self.find_resolution()

        self.xc = np.linspace(rmin, rmax, self.nx)
        self.yc = np.linspace(rmin, rmax, self.ny)
        self.zc = np.linspace(rmin, rmax, self.nz)
        self.X, self.Y, self.Z = np.meshgrid(self.xc, self.yc, self.zc)

        if field == 'dens':
            utils.print_(
                f'Creating a box of size [{rmin*u.cm.to(u.au):.1f} au, ' +
                f'{rmax*u.cm.to(u.au):.1f} au] with {self.ncells} cells ' +
                f'per side.'
            )

        # Determine which quantity is to be interpolated
        if field == 'dens':
            utils.print_(
                f'Interpolating density values onto the grid')
            values = self.dens
        elif field == 'temp':
            utils.print_(
                f'Interpolating temperature values onto the grid')
            values = self.temp
        elif field == 'vx':
            utils.print_(
                f'Interpolating X vector field components onto the grid')
            values = self.vx
        elif field == 'vy':
            utils.print_(
                f'Interpolating Y vector field components onto the grid')
            values = self.vy
        elif field == 'vz':
            utils.print_(
                f'Interpolating Z vector field components onto the grid')
            values = self.vz
        else:
            raise ValueError('field must be "dens", "temp", "vx", "vy" or "vz"')

        # Set a number or key to fill the values outside the interpol. range
        fill = {
            'min': np.min(values), 
            'max': np.max(values), 
            'mean': np.mean(values), 
        }.get(fill, fill)

        # Interpolate the point values at the grid points
        interp = griddata(
            points=np.vstack([self.x, self.y, self.z]).T, 
            values=values, 
            xi=(self.X, self.Y, self.Z), 
            method=method, 
            fill_value=fill,
        )
 
        # Store the interpolated field
        if field == 'dens':
            self.grid_dens = interp
        elif field == 'temp':
            self.grid_temp = interp
        elif field == 'vx':
            self.grid_vx = interp
        elif field == 'vy':
            self.grid_vy = interp
        elif field == 'vz':
            self.grid_vz = interp


    def write_grid_file(self, regular=True):
        """ Write the regular cartesian grid file """
        with open('amr_grid.inp','w+') as f:
            # iformat
            f.write('1\n')                     
            # Regular grid
            f.write('0\n')                      
            # Coordinate system
            f.write(f'{self.cordsystem}\n')                     
            # Gridinfo
            f.write('0\n')                       
            # Number of cells
            f.write('1 1 1\n')                   
            # Size of the grid
            f.write(f'{self.nx:d} {self.ny:d} {self.nz:d}\n')

            # Shift the cell centers right to set the cell walls
            if regular:
                dx = np.diff(self.xc)[0]
                dy = np.diff(self.yc)[0]
                dz = np.diff(self.zc)[0]
                self.xw = self.xc + dx
                self.yw = self.yc + dy
                self.zw = self.zc + dz

                # Add the missing wall at the beginning of each axis
                self.xw = np.insert(self.xw, 0, self.xc[0])
                self.yw = np.insert(self.yw, 0, self.yc[0])
                self.zw = np.insert(self.zw, 0, self.zc[0])
            else:
                self.xw = self.xc
                self.yw = self.yc
                self.zw = self.zc

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
        density = self.grid_dens.ravel(order='F')
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
                temp1d = self.grid_temp.ravel(order='F')
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
        
        temperature = self.grid_temp.ravel(order='F')
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
 
        if self.vfield:
            self.vfield = VectorField(self.X, self.Y, self.Z, morphology)
            self.vfield.vx = self.grid_vx 
            self.vfield.vy = self.grid_vy 
            self.vfield.vz = self.grid_vz 
 
        with open('grainalign_dir.inp','w+') as f:
            f.write('1\n')
            f.write(f'{int(self.nx * self.ny * self.nz)}\n')
            for iz in range(self.nx):
                for iy in range(self.ny):
                    for ix in range(self.nz):
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
                    'density': self.grid_dens, 
                    'temperature': self.grid_temp
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

            vmin = data_xy.min() if data_xy.min() > 0 else None
            vmax = data_xy.max() if data_xy.max() < np.inf else None

            if np.all(data_xy == np.zeros(data_xy.shape)): 
                raise ValueError(f'{field} midplane is exactly 0.')

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
                if self.vfield and field == 'density':
                    p[0].streamplot(
                        np.linspace(-bbox, bbox, self.ncells), 
                        np.linspace(-bbox, bbox, self.ncells), 
                        self.vfield.vx[..., plane], 
                        self.vfield.vy[..., plane],
                        linewidth=0.5,
                        color='k', 
                        density=0.7, 
                    )
                    p[1].streamplot(
                        np.linspace(-bbox, bbox, self.ncells), 
                        np.linspace(-bbox, bbox, self.ncells), 
                        self.vfield.vx[:, plane, :], 
                        self.vfield.vz[:, plane, :],
                        linewidth=0.5,
                        color='k', 
                        density=0.7, 
                    )
            except Exception as e:
                raise
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

            if data is None:
                data = {
                    'density': self.grid_dens, 
                    'temperature': self.grid_temp
                }[field]
            
            title = {
                'density': r'Dust Density (g cm$^{-3}$)', 
                'temperature': r'Gas Temperature (g cm$^{-3}$)',
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
                extent=[0, self.nx, 0, self.ny, 0, self.nz], 
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


    def create_vtk(self, dust_density=False, dust_temperature=False, rename=False):
        """ Call radmc3d to create a VTK file of the grid """

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


    def render(self, dust_density=False, dust_temperature=False):
        """ Render the new grid in 3D using ParaView """
    
        # Make sure ParaView is installed and callable
        utils.which('paraview')
    
        try:
            if dust_density:
                subprocess.run(
                f'paraview model_dust_density.vtk 2>/dev/null'.split())
            elif dust_temperature:
                subprocess.run(
                f'paraview model_dust_temperature.vtk 2>/dev/null'.split())
        except Exception as e:
            utils.print_('Unable to render using ParaView.',  red=True)
            utils.print_(e, bold=True)


class SphericalGrid(Grid):
    def __init__(self):
        """ This is yet to be implemented.
            It will be translated from old scripts and from the very own
            radmc3d's implementation.
            src:https://github.com/dullemond/radmc3d-2.0/blob/master/python/
                radmc3d_tools/sph_to_sphergrid.py
         """

        utils.not_implemented()
