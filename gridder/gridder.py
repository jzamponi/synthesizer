import os, sys
import subprocess
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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

        self.dens = np.zeros((ncells, ncells, ncells))
        self.temp = np.zeros((ncells, ncells, ncells))
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

    def read_sph(self, filename, source='sphng-bin'):
        """ Read SPH data. 
            Below are defined small interfaces to read in data from SPH codes.

            To add new code sources, synthesizer needs (all values in CGS):
                self.x, self.y, self.z, self.dens and self.temp (optional) 
         """

        import h5py

        utils.print_(f'Reading data from: {filename} | Format: {source}')
        if not os.path.exists(filename): 
            raise FileNotFoundError('Input SPH file does not exist')

        if source.lower() == 'sphng-bin':
            self.sph = utils.read_sph(filename, remove_sink=True, cgs=True)
            self.x = self.sph[:, 2]
            self.y = self.sph[:, 3]
            self.z = self.sph[:, 4]
            self.dens = self.sph[:, 10] / self.g2d
            self.npoints = len(self.dens)
            if self.add_temp:
                self.temp = self.sph[:, 11]
            else:
                self.temp = np.zeros(self.dens.shape)

        elif source.lower() == 'gizmo':
            self.sph = h5py.File(filename, 'r')['PartType0']
            code_length = u.pc.to(u.cm)
            code_dens = 1.3816326620099363e-23
            coords = np.array(self.sph['Coordinates'])
            self.x = coords[:, 0] * code_length
            self.y = coords[:, 1] * code_length
            self.z = coords[:, 2] * code_length
            self.dens = np.array(self.sph['Density']) * code_dens / self.g2d
            self.npoints = len(self.dens)
            if self.add_temp: 
                self.temp = np.array(self.sph['KromeTemperature'])
            else:
                self.temp = np.zeros(self.dens.shape)

            # Recenter the particles based on the center of mass
            self.x -= np.average(self.x, weights=self.dens)
            self.y -= np.average(self.y, weights=self.dens)
            self.z -= np.average(self.z, weights=self.dens)

        elif source.lower() == 'gadget':
            utils.not_implemented()

        elif source.lower() == 'arepo':
            utils.not_implemented()

        elif source.lower() == 'gradsph':
            utils.not_implemented()
            
        elif source.lower() == 'gandalf':
            utils.not_implemented()
            
        else:
            raise ValueError(f'Source = {source} is currently not supported')

    def read_amr(self, filename, sourle='athena++'):
        """ Read SPH data """

        import h5py

        utils.print_(f'Reading data from: {filename} | Format: {source}')
        if not os.path.exists(filename): 
            raise FileNotFoundError('Input AMR file does not exist')

        if source.lower() == 'athena++':
            utils.not_implemented()

        elif source.lower() == 'zeustw':
            utils.not_implemented()

        elif source.lower() == 'flash':
            utils.not_implemented()

        elif source.lower() == 'enzo':
            utils.not_implemented()
            
        elif source.lower() == 'ramses':
            utils.not_implemented()
            
        else:
            raise ValueError(f'Source = {source} is currently not supported')

    def trim_box(self, bbox=None, rout=None):
        """ 
        Trim the original grid to a given size, can be rectangular or spherical. 
        """

        # Emtpy dynamic array to store the indices of the particles to remove
        to_remove = []

        if bbox is not None:

            self.bbox = bbox

            # Create a matrix of vertices if bbox is provided as a single number
            if np.isscalar(bbox):
                bbox = [[-bbox, bbox]] * 3
            
            # Store the vertices
            x1 = bbox[0][0]
            x2 = bbox[0][1]
            y1 = bbox[1][0]
            y2 = bbox[1][1]
            z1 = bbox[2][0]
            z2 = bbox[2][1]
            utils.print_(f'Deleting particles outside a box ' +
                f'half-length of {self.bbox * u.cm.to(u.au)} au')

            # Iterate over particles and delete upon reject
            for i in range(self.x.size):
                if self.x[i] < x1 or self.x[i] > x2:
                    to_remove.append(i)

            for j in range(self.y.size):
                if self.y[j] < y1 or self.y[j] > y2:
                    to_remove.append(j)

            for k in range(self.z.size):
                if self.z[k] < z1 or self.z[k] > z2:
                    to_remove.append(k)

        if rout is not None and bbox is None:
            # Override any previous value of rout
            self.rout = rout
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
        utils.print_(f'{self.npoints - self.x.size} ' +
            'particles were not included in the grid')

    def interpolate_points(self, field, method='linear', fill='min'):
        """
            Interpolate a set of points in cartesian coordinates along with their
            values into a rectangular grid.
        """

        # Construct the rectangular grid
        utils.print_('Creating a box of ' +
            f'{self.ncells} x {self.ncells} x {self.ncells} cells')

        rmin = np.min([self.x.min(), self.y.min(), self.z.min()])
        rmax = np.max([self.x.max(), self.y.max(), self.z.max()])
        self.xc = np.linspace(rmin, rmax, self.ncells)
        self.yc = np.linspace(rmin, rmax, self.ncells)
        self.zc = np.linspace(rmin, rmax, self.ncells)
        self.X, self.Y, self.Z = np.meshgrid(self.xc, self.yc, self.zc)

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
        fill = np.min(values) if fill == 'min' else fill
        xyz = np.vstack([self.x, self.y, self.z]).T
        interp = griddata(
            points=xyz, 
            values=values, 
            xi=(self.X, self.Y, self.Z), 
            method=method, 
            fill_value=fill
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
            # Coordinate system: cartesian
            f.write('1\n')                     
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
            xw = self.xc + dx
            yw = self.yc + dy
            zw = self.zc + dz

            # Add the missing wall at the beginning of each axis
            xw = np.insert(xw, 0, self.xc[0])
            yw = np.insert(yw, 0, self.yc[0])
            zw = np.insert(zw, 0, self.zc[0])

            # Write the cell walls
            for i in xw:
                f.write(f'{i:13.6e}\n')      
            for j in yw:
                f.write(f'{j:13.6e}\n')      
            for k in zw:
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

    def write_vector_field(self, vector_field):
        """ Create a vector field for dust alignment """

        utils.print_('Writing grain alignment direction file')

        x = self.X
        y = self.Y
        z = self.Z
        self.vector_field = vector_field.lower()
        field = np.zeros((self.ncells, self.ncells, self.ncells, 3))
 
        if self.vector_field == 'x':
            field[..., 0] = np.ones(x.shape)
 
        elif self.vector_field == 'y':
            field[..., 1] = np.ones(y.shape)
 
        elif self.vector_field == 'z':
            field[..., 2] = np.ones(z.shape)
 
        elif self.vector_field in ['t', 'toroidal']:
            r = np.sqrt(x**2 + y**2)
            field[..., 0] = y / r
            field[..., 1] = -x / r
 
        elif self.vector_field in ['r', 'radial']:
            r = np.sqrt(x**2 + y**2 + z**2)
            field[..., 0] = x / r
            field[..., 1] = y / r
            field[..., 2] = z / r
 
        elif self.vector_field in ['h', 'hourglass']:
            a = 0.1
            factor = np.sqrt(
                1 + (a*x*z)**2*np.exp(-2*a*z*z) + (a*y*z)**2*np.exp(-2*a*z*z))
            field[..., 0] = a * x * z * np.exp(-a * z*z) / factor
            field[..., 1] = a * y * z * np.exp(-a * z*z) / factor
            field[..., 2] = np.ones(z.shape)

        elif self.vector_field in ['hel', 'helicoidal']:
            # Helicoidal = Superpoistion of Toroidal & Helicoidal
            r = np.sqrt(x**2 + y**2)
            toro = np.array([y/r, -x/r, np.zeros(z.shape)])
            a = 0.1
            factor = np.sqrt(
                1 + (a*x*z)**2*np.exp(-2*a*z*z) + (a*y*z)**2*np.exp(-2*a*z*z))

            hour = np.array([
                a * x * z * np.exp(-a * z*z) / factor, 
                a * y * z * np.exp(-a * z*z) / factor,
                np.ones(z.shape)
            ])
            heli = (toro + hour) / np.linalg.norm(toro + hour)
            field[..., 0] = heli[0]
            field[..., 1] = heli[1]
            field[..., 2] = heli[2]

        elif self.vector_field in ['d', 'dipole']:
            r5 = np.sqrt(x**2 + y**2 + z**2)**5
            field[..., 0] = (3 * x * z) / r5
            field[..., 1] = (3 * y * z) / r5
            field[..., 2] = (2*z*z - x*x - y*y) / r5
 
        elif self.vector_field in ['q', 'quadrupole']:
            r7 = 2 * np.sqrt(x**2 + y**2 + z**2)**7 
            field[..., 0] = (-3*x*(x**2 + y**2 - 4*z**2)) / r7
            field[..., 1] = (-3*y*(x**2 + y**2 - 4*z**2)) / r7
            field[..., 2] = (3*z*(-3*x**2 - 3*y**2 + 2*z**2)) / r7
        
        # Normalize the field
        l = np.sqrt(field[..., 0]**2 + field[..., 1]**2 + field[..., 2]**2)
        field[..., 0] = np.squeeze(field[..., 0]) / l
        field[..., 1] = np.squeeze(field[..., 1]) / l
        field[..., 2] = np.squeeze(field[..., 2]) / l

        # Assume perfect alignment (a_eff = 1). We can change it in the future
        a_eff = 1
        field[..., 0] *= a_eff
        field[..., 1] *= a_eff
        field[..., 2] *= a_eff
        
        # Write the vector field 
        with open('grainalign_dir.inp','w+') as f:
            f.write('1\n')
            f.write(f'{int(self.ncells**3)}\n')
            for iz in range(self.ncells):
                for iy in range(self.ncells):
                    for ix in range(self.ncells):
                        f.write(f'{field[ix, iy, iz, 0]:13.6e} ' +\
                                f'{field[ix, iy, iz, 1]:13.6e} ' +\
                                f'{field[ix, iy, iz, 2]:13.6e}\n')

    def plot_dens_midplane(self):
        """ Plot the density midplane at z=0 using Matplotlib """
        try:
            from matplotlib.colors import LogNorm
            utils.print_(f'Plotting the density grid midplane at z = 0')
            plt.rcParams['text.usetex'] = True
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
                extent = [-self.bbox*u.cm.to(u.au), 
                        self.bbox*u.cm.to(u.au)] * 2

            plt.imshow(self.interp_dens[:,:,self.ncells//2-1].T, 
                norm=LogNorm(vmin=None, vmax=None),
                cmap='BuPu',
                extent=extent,
            )
            plt.colorbar()
            plt.xlabel('AU')
            plt.ylabel('AU')
            plt.title(r'Dust Density Midplane at $z=0$ (g cm$^{-3}$)')
            plt.show()

        except Exception as e:
            utils.print_('Unable to show the 2D grid slice.',  red=True)
            utils.print_(e, bold=True)

    def plot_temp_midplane(self):
        """ Plot the temperature midplane at z=0 using Matplotlib """
        try:
            from matplotlib.colors import LogNorm
            utils.print_(f'Plotting the temperature grid midplane at z = 0')
            plt.rcParams['text.usetex'] = True
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
                extent = [-self.bbox*u.cm.to(u.au), 
                        self.bbox*u.cm.to(u.au)] * 2

            plt.imshow(self.interp_temp[:,:,self.ncells//2-1].T, 
                norm=LogNorm(vmin=None, vmax=None),
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
        """ Render the interpolated 3D field using Mayavi """
        try:
            utils.print_('Visualizing the interpolated field ...')
            from mayavi import mlab
            mlab.contour3d(self.interp_dens, contours=20, opacity=0.2)
            mlab.show()

        except Exception as e:
            utils.print_('Unable to show the 3D grid.',  red=True)
            utils.print_(e, bold=True)


    def plot_temp_3d(self): 
        """ Render the interpolated 3D field using Mayavi """
        try:
            utils.print_('Visualizing the interpolated field ...')
            from mayavi import mlab
            mlab.contour3d(self.interp_temp, contours=20, opacity=0.2)
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
