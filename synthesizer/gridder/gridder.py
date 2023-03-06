import os, sys
import subprocess
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from synthesizer.gridder.vector_field import VectorField
from synthesizer.gridder.sph_reader  import *
from synthesizer.gridder.amr_reader  import *
from synthesizer import utils

class CartesianGrid():
    def __init__(self, ncells, bbox=None, rout=None, nspec=1, csubl=0, 
            sootline=300, g2d=100, temp=False):
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
        self.dens = np.zeros((ncells, ncells, ncells))
        self.temp = np.zeros((ncells, ncells, ncells))
        self.interp_dens = None
        self.interp_temp = None
        self.vfield = None
        self.add_temp = temp
        self.ncells = ncells
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

        utils.print_(
            f'Reading data from: {filename} | Format: {source.upper()}', end='')
        if not os.path.exists(filename): 
            raise FileNotFoundError(f'{utils.color.red}' +\
                f'Input SPH file does not exist{utils.color.none}')

        if source.lower() == 'sphng':
            self.sph = SPHng(filename, temp=self.add_temp)

        elif source.lower() == 'gizmo':
            self.sph = Gizmo(filename, temp=self.add_temp)

        elif source.lower() == 'gadget':
            self.sph = Gadget(filename, temp=self.add_temp)

        elif source.lower() == 'arepo':
            self.sph = Arepo(filename, temp=self.add_temp)

        elif source.lower() == 'phantom':
            utils.not_implemented()
            self.sph = Phantom(filename, temp=self.add_temp)

        elif source.lower() == 'nbody6':
            self.sph = Nbody6(filename, temp=self.add_temp)

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

    def read_amr(self, filename, sourle='athena++'):
        """ Read AMR data """

        utils.print_(f'Reading data from: {filename} | Format: {source}')
        if not os.path.exists(filename): 
            raise FileNotFoundError('Input AMR file does not exist')

        if source.lower() == 'athena++':
            utils.not_implemented()

        elif source.lower() == 'zeustw':
            utils.not_implemented()

            # Generate a Data instance
            data = ZeusTW()

            # Read coordinates: x?a are cell edges and x?b are cell centers
            data.generate_coords(r="z_x1ap", th="z_x2ap", ph="z_x3ap")

            frame = str(filename).zfill(5)
            data.rho = data.read(f"o_d__{frame}")
            data.trim_ghost_cells(field_type='coords')
            data.trim_ghost_cells(field_type='scalar')
            data.trim_ghost_cells(field_type='vector')
            data.LH_to_Gaussian()
            data.generate_temperature()
            data.generate_cartesian()

            self.xw = data.x
            self.yw = data.y
            self.zw = data.z
            self.dens = data.rho
            self.temp = data.temp

        elif source.lower() == 'flash':
            utils.not_implemented()

        elif source.lower() == 'enzo':
            utils.not_implemented()
            
        elif source.lower() == 'ramses':
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
        utils.print_(f'Particles included: {self.x.size} | ' +
            f'Particles excluded: {self.npoints - self.x.size} ')

    def interpolate_points(self, field, method='linear', fill='min'):
        """
            Interpolate a set of points in cartesian coordinates along with their
            values into a rectangular grid.
        """

        # Construct the rectangular grid
        rmin = np.min([self.x.min(), self.y.min(), self.z.min()])
        rmax = np.max([self.x.max(), self.y.max(), self.z.max()])
        self.xc = np.linspace(rmin, rmax, self.ncells)
        self.yc = np.linspace(rmin, rmax, self.ncells)
        self.zc = np.linspace(rmin, rmax, self.ncells)
        self.X, self.Y, self.Z = np.meshgrid(self.xc, self.yc, self.zc)
        self.bbox = (rmax - rmin) / 2

        utils.print_(f'Creating a box of size [{rmin*u.cm.to(u.au):.1f} au, ' +\
            f'{rmax*u.cm.to(u.au):.1f} au] with {self.ncells} cells per side.')

        # Determine which quantity is to be interpolated
        if field == 'dens':
            utils.print_(f'Interpolating density values onto the grid')
            values = self.dens
        elif field == 'temp':
            utils.print_(f'Interpolating temperature values onto the grid')
            values = self.temp
        else:
            raise ValueError('field must be "dens" or "temp".')

        # Interpolate the point values at the grid points
        interp = griddata(
            points=np.vstack([self.x, self.y, self.z]).T, 
            values=values, 
            xi=(self.X, self.Y, self.Z), 
            method=method, 
            fill_value=np.min(values) if fill == 'min' else fill,
        )
 
        # Store the interpolated field
        if field == 'dens':
            self.interp_dens = interp
        elif field == 'temp':
            self.interp_temp = interp


    def write_grid_file(self):
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
            f.write(f'{self.ncells:d} {self.ncells:d} {self.ncells:d}\n')

            # Shift the cell centers right to set the cell walls
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
        density = self.interp_dens.ravel(order='F')
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
        
        temperature = self.interp_temp.ravel(order='F')
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
 
        if self.vfield is None:
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

    def plot_midplane(self, field, data=None):
        """ Plot the density midplane at z=0 using Matplotlib """
        try:
            from matplotlib.colors import LogNorm
            utils.print_(f'Plotting the density grid midplane')
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
                    'density': self.interp_dens, 
                    'temperature': self.interp_temp
                }[field]
            else:
                data = data.T
            
            # Set the plot title for the right field
            title = {
                'density': r'Dust Density (g cm$^{-3}$)', 
                'temperature': r'Gas Temperature (g cm$^{-3}$)',
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

            pxy = p[0].imshow(
                data_xy, 
                norm=LogNorm(vmin=vmin, vmax=vmax), 
                cmap='BuPu' if field == 'density' else 'inferno',
                extent=extent,
            )
            pxz = p[1].imshow(
                data_xz, 
                norm=LogNorm(vmin=vmin, vmax=vmax), 
                cmap='BuPu' if field == 'density' else 'inferno',
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

    def plot_3d(self, field, data=None): 
        """ Render the interpolated 3D field using Mayavi """
        try:
            from mayavi import mlab
            from mayavi.api import Engine
            from mayavi.modules.text3d import Text3D
            from mayavi.modules.grid_plane import GridPlane

            utils.print_('Visualizing the interpolated field ...')

            if data is None:
                data = {
                    'density': self.interp_dens, 
                    'temperature': self.interp_temp
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

            # Render data
            plot = mlab.contour3d(
                data, contours=100, opacity=0.2, colormap='inferno')

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
                utils.print_('Unable to render using ParaView.',  red=True)
                utils.print_(e, bold=True)


class SphericalGrid():
    def __init__(self):
        """ This is yet to be implemented.
            It will be translated from old scripts and from the very own
            radmc3d's implementation.
            src:https://github.com/dullemond/radmc3d-2.0/blob/master/python/
                radmc3d_tools/sph_to_sphergrid.py
         """

        utils.not_implemented()
