import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel

from synthesizer import utils

class SynImage:
    """ Object meant to add intrument related effects to ideal-intensity images.
        Equations are taken from Zamponi et al. (2022).
        Ref: https://ui.adsabs.harvard.edu/abs/2022Ap%26SS.367...65Z/abstract
    """

    def __init__(self, imagename):
        self.imagename = imagename
        self.img, self.hdr = fits.getdata(imagename, header=True)
        self.pixsize = self.hdr['CDELT1'] * u.deg.to(u.arcsec)

    def convolve(self, res, pa=0):
        """ Convolve an image with a gaussian beam given a resolution (") """

        utils.print_(f'Convolving image {self.imagename} with a beam of {res}"')
        self.resolution = res

        # Get the resolution (FWHM) in terms of image pixels
        res_pix = res / self.pixsize

        # Convert the kernel FWHM into a standard deviation
        std_x = res_pix / np.sqrt(8 * np.log(2))

        # Create a Gaussian kernel and convolve the image
        kernel = Gaussian2DKernel(std_x, std_x, pa)
        self.img = convolve_fft(self.img, kernel)

        # Rescale the flux from Jy/pixel to Jy/beam
        self.img = np.pi / 4 / np.log(2) * res**2 / self.pixsize**2 * self.img
        
        # Update the header
        self.hdr['BUNIT'] = 'Jy/beam'
        self.hdr['BMAJ'] = res * u.arcsec.to(u.deg)
        self.hdr['BMIN'] = res * u.arcsec.to(u.deg)
        self.hdr['BPA'] = pa

    def add_noise(self, obstime, bandwidth=8e9, pwv=10, elevation=45):
        """ Add thermal noise to an image based on an observing time 
            in hours and a frequency bandwidth in Hz.
        """
        self.obstime = obstime * u.h.to(u.s)
        
        # Estimate the receiver's system temperature
        self.T_sys = self._get_T_sys('apex', pwv, elevation) 

        # Estimate the thermal noise from the radiometer equation
        self.T_rms = self.T_sys / np.sqrt(bandwidth * obstime)

        # Create noise as a Gaussian distribution centered at T_rms 
        self.noise = np.random.normal(0, self.T_rms, self.img.shape)

        # Add noise to the image
        self.img = self.img + self.noise
        self.hdr['NOISE'] = f'{self.T_rms}K'
        utils.print_(
            f'Adding thermal (gaussian) noise centered at {self.T_rms:.2} K')

    def _get_T_sys(self, telescope, pwv, elevation):
        """ Find tabulated system temperatures for different telescope receivers """
        if telescope == 'apex':
            return 500

        elif telescope == 'muse':
            return 500

    def write_fits(self, fitsimage):
        """ Write the resulting image to FITS file """
        utils.write_fits(
            fitsimage, data=self.img, header=self.hdr, overwrite=True)
