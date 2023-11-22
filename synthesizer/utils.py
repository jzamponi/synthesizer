"""
    Collection of useful functions for my thesis.
"""
import os
import sys
import time
import warnings
import subprocess
from glob import glob
from functools import wraps
from pathlib import Path, PosixPath

import numpy as np

import matplotlib.pyplot as plt

from astropy.io import ascii, fits
from astropy import units as u
from astropy import constants as c

home = Path.home()
pwd = Path(os.getcwd())


class NotInstalled(Exception):
    def __init__(self, message=''):
        message = f'{color.red}{message}{color.none}'
        super().__init__(message)

class color:
    red = "\033[91m"
    green = '\033[92m'
    warn = '\033[93m'
    blue = '\033[94m'
    header = '\033[95m'
    cyan = '\033[96m'
    none = "\033[0m"
    bold = "\033[1m"
    gray = "\033[2m"
    it = "\033[3m"
    ul = '\033[4m'

def print_(string, verbose=True, bold=False, red=False, blue=False, green=False,
    ul=False, fname=None, *args, **kwargs):

    # Get the name of the calling function by tracing one level up in the stack
    fname = sys._getframe(1).f_code.co_name if fname is None else fname

    # Check if verbosity state is defined as a global variable
    if verbose is None:
        if "VERBOSE" in globals() and VERBOSE:
            verbose = True

    if verbose:
        if bold:
            print(f"{color.bold}[{fname}] {string} {color.none}", flush=True, 
                *args, **kwargs)
        elif red:
            print(f"{color.red}[{fname}] {string} {color.none}", flush=True, 
                *args, **kwargs)
        elif blue:
            print(f"{color.blue}[{fname}] {string} {color.none}", flush=True, 
                *args, **kwargs)
        elif green:
            print(f"{color.green}[{fname}] {string} {color.none}", flush=True, 
                *args, **kwargs)
        elif ul:
            print(f"{color.ul}[{fname}] {string} {color.none}", flush=True, 
                *args, **kwargs)
        else:
            print(f"[{fname}] {string}", flush=True, 
                *args, **kwargs)

def basename(path, dot=True):
    """ Trim the trailing path from full path to leave the filename alone """
    if '/' in path:
        path = path.split('/')[-1]
        if dot and '.' in path:
            path = path.split('.')[0]

    return path

def not_implemented(msg=''): 
    raise NotImplementedError(
        f"{color.blue}I'm sorry, this feature is not yet implemented. " +\
        f"{color.it}{msg}{color.none}")


def write_fits(filename, data, header=None, overwrite=True, verbose=False):
    # Get the name of the calling function by tracing one level up in the stack
    caller = sys._getframe(1).f_code.co_name

    if filename != "":
        if overwrite and os.path.exists(filename):
            os.remove(filename)

        fits.HDUList(fits.PrimaryHDU(data=data, header=header)).writeto(filename)
        print_(f"Written file {filename}", verbose=verbose, fname=caller)


def elapsed_time(caller):
    """ Decorator designed to print the time taken by a functon. """
    # To do: elapsed_time() is not forwarding the error and exceptions
    # all the way up to its origin and makes it harder to debug. 
    # It only returns the following message from wrapper(): 
    # UnboundLocalError: local variable 'f' referenced before assignment

    # Forward docstrings to the caller function
    @wraps(caller)

    def wrapper(*args, **kwargs):
        # Measure time before it runs
        start = time.time()

        # Execute the caller function
        f = caller(*args, **kwargs)

        # Measure time difference after it finishes
        runtime = time.time() - start

        # Print the elapsed time nicely formatted
        tformat = "%H:%M:%S"
        print_(
            f'Elapsed time: {time.strftime(tformat, time.gmtime(runtime))}',
            verbose = True, 
            fname = caller.__name__
        )
        return f
    return wrapper

def latest_file(filename):
    """ Return the name of the most recent file given the file pattern 
        filename. It supports wildcards. 
        It basically emulates the bash command $ ls -t filename
    """
    return max(glob(filename), key=lambda f: os.path.getctime(f))

def file_exists(filename, raise_=True, msg=''):
    """ Raise an error if a file doesnt exist. Supports linux wildcards. """

    msg = f'{color.red}{filename} not found. {msg}{color.none}'
    
    if '*' in filename:
        if len(glob(filename)) == 0:
            if raise_:
                raise FileNotFoundError(msg)
            else:
                return False
        else:
            return True
        
    else:
        if not os.path.exists(filename): 
            if raise_:
                raise FileNotFoundError(msg)
            else:
                return False
        else:
            return True

def download_file(url, msg=None, verbose=True, *args, **kwargs):
    """ Perform an HTTP GET request to fetch files from internet """ 

    if verbose:
        print_(f'Downloading file from {url}' if msg is None else msg, 
            *args, **kwargs)

    if 'github' in url and 'raw' not in url:
        raise ValueError("URLs of files from GitHub must come in raw format."+\
            ' Consider adding ?raw=true at the end of the url.')
    
    import requests

    # Strip the filename from the base url
    filename = url.split('/')[-1]

    try:
        # Perform an HTTP GET request
        req = requests.get(url)

        # Raise the HTTP Error if existent
        req.raise_for_status()

        # Download the file
        download = Path(filename).write_bytes(req.content)

    except requests.ConnectionError:
        print_(f'No internet connection. Unable to download file.', red=True)
        return False

def which(program, msg=''):
    """ Call the which unix command to check whether a given program is
        in the PATH, i.e., executable, or not. 
        which returns 0 if the program is found, and 1 otherwise.
        In Python 0 and 1 are False and True, so we catch the latter only.
    """
    
    not_found = subprocess.run(['which', f'{program}'], stdout=subprocess.PIPE)

    if not_found.returncode:
        raise NotInstalled(
            f'{program} is not installed on this machine, or not in the PATH.'+\
            f' {msg}')


def plot_checkout(fig, show, savefig, path="", block=True):
    """
    Final step in every plotting routine:
        - Check if the figure should be showed.
        - Check if the figure should be saved.
        - Return the figure, for further editing.
    """
    # Turn path into a Path object for flexibility
    path = Path(path)

    # Save the figure if required
    savefig = "" if savefig is None else savefig
    if savefig != "":
        # Check if savefig is a global path
        plt.savefig(f"{savefig}" if '/' in savefig else f"{path}/{savefig}") 

    # Show the figure if required
    if show:
        plt.show(block=block)

    return fig


def radmc3d_casafits(fitsfile='radmc3d_I.fits', radmc3dimage='image.out',
        stokes='I', tau=False, dpc=140, verbose=False):
    """ Read in an image.out file created by RADMC3D and generate a
        FITS file with a CASA-compatible header, ready for a 
        synthetic observation.
    """

    # Read the header
    with open(radmc3dimage, 'r') as f:
        iformat = int(f.readline())
        line = f.readline().split()
        nx = int(line[0])
        ny = int(line[1])
        nlam = int(f.readline())
        line = f.readline().split()
        pixsize_x = float(line[0])
        pixsize_y = float(line[1])
        lam = float(f.readline())
        
    # Read the data
    img = np.loadtxt(radmc3dimage, skiprows=5)

    # Select Stokes map
    if iformat == 3:
        img = img[:, {'I': 0, 'Q': 1, 'U': 2, 'V': 3}[stokes]]
        
    # Make a squared map
    img = img.reshape(nx, ny)

    if not tau:
        # Rescale to Jy/sr
        img = img * (
            u.erg * u.s ** -1 * u.cm ** -2 * u.Hz ** -1 * u.sr ** -1).to(
            u.Jy * u.sr ** -1
        )

        # Convert sr into pixels (Jy/sr --> Jy/pixel)
        img = img * (pixsize_x**2 / (dpc*u.pc.to(u.cm))**2)

        # Convert physical pixel size to angular size for the header
        cdelt1 = (pixsize_x / (dpc*u.pc.to(u.cm))) * u.rad.to(u.deg)
        cdelt2 = (pixsize_y / (dpc*u.pc.to(u.cm))) * u.rad.to(u.deg)
        cunit1 = 'deg'
        cunit2 = 'deg'
        ctype1 = 'RA---SIN'
        ctype2 = 'DEC--SIN'

    else:
        cdelt1 = pixsize_x * u.cm.to(u.au)
        cdelt2 = pixsize_y * u.cm.to(u.au)
        cunit1 = 'AU'
        cunit2 = 'AU'
        ctype1 = 'param'
        ctype2 = 'param'

    # Set a minimal header
    header = fits.Header({
        'CRPIX1': 1 + nx / 2,
        'CDELT1': cdelt1,
        'CRVAL1': 0, 
        'CUNIT1': cunit1,
        'CTYPE1': ctype1,
        'CRPIX2': 1 + ny / 2,
        'CDELT2': cdelt2,
        'CRVAL2': 0, 
        'CUNIT2': cunit2,
        'CTYPE2': ctype2,
        'LAMBDA': f'{lam}um',
        'RESTFRQ': c.c.cgs.value / (lam*u.micron.to(u.cm)),
        'BUNIT': 'Jy/pixel' if not tau else '',
        'BTYPE': 'Intensity' if not tau else 'op. depth', 
        'BZERO': 1.0, 
        'BSCALE': 'BSCALE', 
        'LONPOLE': 180.0, 
        'DISTANCE': f'{dpc}pc',
    })
    write_fits(fitsfile, img, header, True, verbose)

def stats(data, verbose=False, slice=None):
    """
    Compute basic statistics of a array or a fits file.
    The functions used here ignore NaN values in the data.
    """

    # Read data
    if isinstance(data, (str,PosixPath)):
        data, hdr = fits.getdata(data, header=True)

        if isinstance(slice, int):
            data = data[slice]
        elif isinstance(slice, list) and len(slice) == 2:
            data = data[slice[0], slice[1]]
    else:
        data = np.array(data)

    # Set the relevant quantities
    stat = {
        "max": np.nanmax(data),
        "mean": np.nanmean(data),
        "min": np.nanmin(data),
        "maxpos": maxpos(data),
        "minpos": minpos(data),
        "std": np.nanstd(data),
        "S/N": np.nanmax(data) / np.nanstd(data),
    }

    # Print statistics if verbose enabled
    for label, value in stat.items():
        print_(f"{label}: {value}", verbose=verbose)

    return stat


def maxpos(data, axis=None):
    """
    Return a tuple with the coordinates of a N-dimensional array.
    """
    # Read data from fits file if data is string
    if isinstance(data, (str,PosixPath)):
        data = fits.getdata(data)

    # Remove empty axes
    data = np.squeeze(data)

    return np.unravel_index(np.nanargmax(data, axis=axis), data.shape)


def minpos(data, axis=None):
    """
    Return a tuple with the coordinates of a N-dimensional array.
    """
    # Read data from fits file if data is string
    if isinstance(data, (str,PosixPath)):
        data = fits.getdata(data)

    # Remove empty axes
    data = np.squeeze(data)

    return np.unravel_index(np.nanargmin(data, axis=None), data.shape)


def add_comment(filename, comment):
    """
    Read in a fits file and add a new keyword to the header.
    """
    data, header = fits.getdata(filename, header=True)

    header["NOTE"] = comment

    write_fits(filename, data=data, header=header, overwrite=True)


def edit_header(filename, key, value, verbose=False):
    """
    Read in a fits file and change the value of a given keyword.
    """
    data, header = fits.getdata(filename, header=True)

    # Check if the key is already in the header
    value_ = header.get(key, default=None)

    if value_ is None:
        print_(f"Adding keyword {key} = {value}", verbose=verbose)
        header[key] = value

    elif value == 'del' and value_ is not None:
        print_(f'Deleting from header: {key} = {value_}', verbose=verbose)
        del header[key]

    else:
        print_(f"Keyword {key} already exists.", verbose=verbose)
        print_(f"Updating from {value_} to {value}.", verbose=verbose)
        header[key] = value

    write_fits(filename, data=data, header=header, overwrite=True, verbose=verbose)


def fix_header_axes(file):
    """ Removes a third axis when NAXIS is 2, to avoid APLPy errors """
    
    header = fits.getheader(file)
    
    # Clean extra keywords from the header to avoid APLPy errors 
    if 'CDELT3' in header and header['NAXIS'] == 2:
        edit_header(file, 'CDELT3', 'del', False)
        edit_header(file, 'CRVAL3', 'del', False)
        edit_header(file, 'CUNIT3', 'del', False)
        edit_header(file, 'CTYPE3', 'del', False)
        edit_header(file, 'CRPIX3', 'del', False)

    if 'PC3_3' in header and header['NAXIS'] == 2:
        edit_header(file, 'PC3_1', 'del', False)
        edit_header(file, 'PC3_2', 'del', False)
        edit_header(file, 'PC1_3', 'del', False)
        edit_header(file, 'PC2_3', 'del', False)
        edit_header(file, 'PC3_3', 'del', False)

def get_beam(filename, verbose=False):
    """ Print or return the info from the header associated to the beam """
    
    data, hdr = fits.getdata(filename, header=True)

    beam = {
        'bmaj': hdr.get('BMAJ', default=0) * u.deg.to(u.arcsec), 
        'bmin': hdr.get('BMIN', default=0) * u.deg.to(u.arcsec), 
        'bpa': hdr.get('BPA', default=0), 
    }
    
    print_(f"Bmaj: {beam['bmaj']:.2f} arcsec", verbose=verbose)
    print_(f"Bmin: {beam['bmin']:.2f} arcsec", verbose=verbose)
    print_(f"Bpa: {beam['bpa']:.2f} deg", verbose=verbose)

    return beam

def plot_map(
    filename,
    header=None,
    rescale=None,
    transpose=False,
    rot90=False, 
    fliplr=False, 
    flipud=False, 
    cblabel=None,
    scalebar=None,
    cmap="magma",
    stretch='linear', 
    verbose=True,
    bright_temp=True,
    figsize=None,
    vmin=None, 
    vmax=None, 
    contours=False, 
    show=True, 
    savefig=None, 
    checkout=True, 
    block=True, 
    distance=140*u.pc,
    *args,
    **kwargs,
):
    """
    Plot a fits file using the APLPy library.
    """
    import aplpy

    # Supress multiple info/warnings from APLPy about header keyword fixes
    aplpy.core.log.setLevel('ERROR')

    # Temporal fits file to write and plot if data is modified
    tempfile = '.temp_file.fits'

    # Read from FITS file if input is a filename
    if isinstance(filename, (str,PosixPath)):
        data, hdr = fits.getdata(filename, header=True)

    elif isinstance(filename, (np.ndarray, list)) and header is not None:
        data = filename
        hdr = header

    # Flip the 2D array if required
    if rot90:
        print_('Rotating the image by 90 degrees', verbose)
        data = np.rot90(data)
    if transpose:
        print_('Transposing the image', verbose)
        data = np.transpose(data)
    if flipud:
        print_('Flipping up to down', verbose)
        data = np.flipud(data)
    if fliplr:
        print_('Flipping left to right', verbose)
        data = np.fliplr(data)
    if rescale is not None:
        print_(f'Rescaling data by factor of: {rescale}', verbose)
        data *= rescale
    if bright_temp:
        try:
            # Convert Jy/beam into Kelvin
            data = Tb(
                data=data,
                freq=hdr.get("RESTFRQ") * u.Hz.to(u.GHz),
                bmin=hdr.get("bmin") * u.deg.to(u.arcsec),
                bmaj=hdr.get("bmaj") * u.deg.to(u.arcsec),
            )
            hdr['BUNIT'] = 'K'

        except Exception as e:
            print_(f'{e}', verbose)
            print_('Beam or frequency keywords not available. ' +\
            'Impossible to convert into T_b.', verbose, bold=True)
    
    # If data is modified, write to a temporal file
    if any(
        [rot90, transpose, flipud, fliplr, rescale is not None, bright_temp]
    ):
        write_fits(tempfile, data.squeeze(), hdr, True, verbose)
        filename = tempfile

    # Initialize the figure
    fig = aplpy.FITSFigure(str(filename), figsize=figsize, *args, **kwargs)
    fig.set_auto_refresh(True)
    fig.show_colorscale(cmap=cmap, vmax=vmax, vmin=vmin, stretch=stretch)
    
    # Add contours if requested
    if contours == True:
        conts = fig.show_contour(
            colors='white', 
            levels=8, 
            returnlevels=True, 
            alpha=0.5
        )
        print_(f'contours: {conts}', verbose)

    elif isinstance(contours, (str, PosixPath)):
        conts = fig.show_contour(
            str(contours), 
            colors='white', 
            levels=8, 
            returnlevels=True, 
            alpha=0.5
        )
        print_(f'Setting contours from external file: {conts}', verbose)

    # Auto set the colorbar label if not provided
    if cblabel is None and bright_temp:
        cblabel = r"$T_{\rm b}$ (K)"

    elif cblabel is None and rescale == 1e3:
        cblabel = "mJy/beam"

    elif cblabel is None and rescale == 1e6:
        cblabel = r"$\mu$Jy/beam"

    elif cblabel is None:
        try:
            hdr = fits.getheader(filename)
            cblabel = hdr.get("BUNIT")
        except Exception as e:
            print_(e, verbose)
            cblabel = ""

    # Colorbar
    fig.add_colorbar()
    fig.colorbar.set_location("top")
    fig.colorbar.set_axis_label_text(cblabel)
    fig.colorbar.set_axis_label_font(size=20, weight=15)
    fig.colorbar.set_font(size=20, weight=15)

    # Frame and ticks
    fig.frame.set_color("black")
    fig.frame.set_linewidth(1.2)
    fig.ticks.set_color('black' if cmap=='magma' else 'black')
    fig.ticks.set_linewidth(1.2)
    fig.ticks.set_length(6)
    fig.ticks.set_minor_frequency(5)

    # Hide ticks and labels if FITS file is not a real obs.
    if "alma" not in str(filename) or "vla" not in str(filename):
        print_(f'File: {filename}. Hiding axis ticks and labels', verbose)
        fig.axis_labels.hide()

    # Beam
    if 'BMAJ' in hdr and 'BMIN' in hdr and 'BPA' in hdr:
        fig.add_beam(facecolor='none', edgecolor='white', linewidth=1)
        bmaj = hdr.get('BMAJ') * u.deg.to(u.arcsec)
        bmin = hdr.get('BMIN') * u.deg.to(u.arcsec)
        #fig.add_label(0.28, 0.07, f'{bmaj:.1f}"x {bmin:.1f}"', 
        #    relative=True, color='white', size=13)

    # Scalebar
    if scalebar is not None:
        if scalebar.unit in ['au', 'pc']:
            try:
                D = distance * u.pc
                print_(
                    f'Physical scalebar created for a distance of: {D}', 
                    verbose=False
                )
                skaska = (scalebar.to(u.cm) / D.to(u.cm)) * u.rad.to(u.arcsec)
                fig.add_scalebar(skaska * u.arcsec)
                unit = f' {scalebar.unit}'
            except Exception as e:
                print_(f'Not able to add scale bar. Error: {e}', verbose=True, red=True)

        elif scalebar.unit in ['arcsec', 'deg']:
            fig.add_scalebar(scalebar)
            unit = f'"' if scalebar.unit == 'arcsec' else "'"

        fig.scalebar.set_color("white")
        fig.scalebar.set_corner("bottom right")
        fig.scalebar.set_font(size=23)
        fig.scalebar.set_linewidth(3)
        fig.scalebar.set_label(f"{int(scalebar.value)}{unit}")

    if filename == tempfile: os.remove(tempfile)

    return plot_checkout(fig, show, savefig, block=block) if checkout else fig


def polarization_map(
    source = 'radmc3d', 
    render="intensity",
    polarization="linear", 
    stokes_I = None, 
    stokes_Q = None, 
    stokes_U = None, 
    wcs="deg",
    rotate=0,
    step=20,
    scale=50,
    scalebar=None, 
    mapsize=None, 
    cmap="magma", 
    vector_color="white",
    vector_width=1, 
    const_pfrac=False,
    min_pfrac=0, 
    rms_I=None, 
    rms_Q=None, 
    bright_temp=False, 
    rescale=None,
    savefig=None,
    show=True,
    block=True, 
    verbose=True,
    *args,
    **kwargs,
):
    """
    Extract I, Q, U and V from the polaris output
    and create maps of Pfrac and Pangle using APLpy.
    """

    # Enable verbosity
    global VERBOSE
    VERBOSE = verbose

    # Store the current path
    pwd = os.getcwd()

    # Read the Stokes components from a data cube
    if source in ['alma', 'vla', 'radmc3d', 'synobs']:
        hdr = fits.getheader(f'{source}_I.fits')

        I = fits.getdata(f'{source}_I.fits').squeeze()
        Q = fits.getdata(f'{source}_Q.fits').squeeze()
        U = fits.getdata(f'{source}_U.fits').squeeze()
        V = np.zeros(U.shape)
        tau = np.zeros(I.shape)
        
    else:
        # Assume the name of the source files for I, Q and U is given manually
        source = stokes_I.split('_')[0]

        print_('Reading files from an external source ...', True)
        hdr = fits.getheader('obs_I.fits' if stokes_I is None else stokes_I)
        I = fits.getdata(stokes_I).squeeze()
        Q = fits.getdata(stokes_Q).squeeze()
        U = fits.getdata(stokes_U).squeeze()
        V = np.zeros(U.shape)
        tau = np.zeros(I.shape)

    # Compute the polarizatoin angle
    pangle = 0.5 * np.arctan2(U, Q)
    pangle = pangle * u.rad.to(u.deg)
	
    # Set rms of stokes I  
    if rms_I is None and 'RMS_I' in hdr:
        rms_I = hdr.get('RMS_I')
        min_I = 5 * rms_I 
    elif rms_I is not None and rms_I > 0:
        min_I = 5 * rms_I
    else:
        rms_I = 0
        min_I = np.nanmin(I)

    # Set rms of stokes Q  
    if rms_Q is None and 'RMS_Q' in hdr:
        rms_Q = hdr.get('RMS_Q')
        min_Q = 3 * rms_Q 
    elif rms_Q is not None and rms_Q > 0:
        min_Q = 2 * rms_Q
    else:
        rms_Q = 0
        min_Q = np.nanmin(Q)
        
    if source in ['alma', 'vla']:
        print_(f'rms_I: {rms_I}', verbose, bold=True)
        print_(f'rms_Q: {rms_Q}', verbose, bold=True)

    # Compute the polarized intensity
    # Apply debias correction (Viallancourt et al. 2006)
    pi = np.sqrt(U**2 + Q**2 - rms_Q**2)

    # Compute the polarization fraction 
    if polarization in ['linear', 'l']:
        pfrac = np.divide(pi, I, where=I != 0)
    elif polarization in ['circular', 'c']:
        pfrac = V / I

    # Mask the polarization vectors for a given threshold in pol. fraction
    if min_pfrac > 0:
        pangle[pfrac < min_pfrac] = np.NaN

    # Mask the polarization vectors emission with stokes I under a given SNR
    pangle[I < min_I] = np.NaN
    pfrac[I < min_I] = np.NaN

    # Mask the polarization vectors emission with pol. intensity under a given SNR
    pangle[pi < min_Q] = np.NaN
    pfrac[pi < min_Q] = np.NaN
    pi[pi < min_Q] = np.NaN
    
    # Set the polarization fraction to 100% to plot vectors of constant length
    if const_pfrac: 
        if render.lower() in ["pf", "pfrac"]:
            print_(f'Ignoring const_pfrac = {const_pfrac}', verbose)
        else:
            pfrac = np.ones(pfrac.shape) 

    # Write quantities into fits files
    quantities = {
        "I": I, 
        "Q": Q, 
        "U": U, 
        #"V": V, 
        #"tau": tau, 
        "pi": pi, 
        "pf": pfrac, 
        "pa": pangle
    }
    for q, d in quantities.items():
        write_fits(f'{source}_{q}.fits', data=d, header=hdr, overwrite=True)

    # Define the unit for the plot
    if bright_temp:
        unit = '(K)'
    elif 'BUNIT' in hdr:
        unit = '(' + hdr.get('BUNIT') + ')'
    else:
        unit = r'($\mu$Jy/pixel)' if rescale is None else '(Jy/pixel)'
        unit = '(Jy/beam)' if source in ['alma','vla'] else unit

    # Select the quantity to plot
    if render.lower() in ["i", "intensity"]:
        figname = f"{source}_I.fits"
        cblabel = f"Stokes I {unit}"
        # Rescale to micro Jy/px
        rescale = 1e6 if rescale is None and source == 'polaris' else rescale

    elif render.lower() in ["q"]:
        figname = f"{source}_Q.fits"
        cblabel = f"Stokes Q {unit}"
        # Rescale to micro Jy/px
        rescale = 1e6 if rescale is None and source == 'polaris' else rescale

    elif render.lower() in ["u"]:
        figname = f"{source}_U.fits"
        cblabel = f"Stokes U {unit}"
        # Rescale to micro Jy/px
        rescale = 1e6 if rescale is None and source == 'polaris' else rescale

    elif render.lower() in ["tau", "optical depth"]:
        figname = f"{source}_tau.fits"
        cblabel = "Optical depth"
        rescale = 1 if rescale is None else rescale

    elif render.lower() in ["pi", "poli", "polarized intensity"]:
        figname = f"{source}_pi.fits"
        cblabel = f"Polarized intensity {unit}"
        # Rescale to micro uJy/px
        rescale = 1e6 if rescale is None and source == 'polaris' else rescale

    elif render.lower() in ["pf", "pfrac", "polarization fraction"]:
        figname = f"{source}_pf.fits"
        cblabel = "Polarization fraction"
        rescale = 1 if rescale is None else rescale
        bright_temp = False

    elif render.lower() in ["pa", "pangle", "polarization angle"]:
        figname = f"{source}_pa.fits"
        cblabel = "Polarization angle (deg)"
        rescale = 1 if rescale is None else rescale
        bright_temp = False

    else:
        rescale = 1
        raise ValueError("Wrong value for render. Must be i, q, u, pf, pi, pa or tau.")

    # Plot the render quantity a colormap
    fig = plot_map(
        figname, 
        rescale=rescale, 
        cblabel=cblabel, 
        bright_temp=bright_temp, 
        scalebar=scalebar, 
        verbose=verbose, 
        block=False,
        cmap=cmap, 
        *args,
        **kwargs,
    )
    
    # Set the image size
    if mapsize is not None:
        D = 141*u.pc
        if mapsize.unit == 'au':
            img_size = ((mapsize / 2).to(u.pc) / D) * u.rad.to(u.deg)

        elif mapsize.unit == 'pc':
            img_size = (mapsize / 2 / D) * u.rad.to(u.deg)

        elif mapsize.unit in ['arcsec']:
            img_size = (mapsize / 2)*u.arcsec.to(u.deg)

        else:
            img_size = mapsize / 2 

        fig.recenter(hdr.get('CRVAL1'), hdr.get('CRVAL2'), radius=img_size)

    rotate = rotate if source == 'obs' else int(rotate) - 90

    # Add polarization vectors
    fig.show_vectors(
        f"{source}_pf.fits",
        f"{source}_pa.fits",
        step=step,
        scale=scale,
        rotate=rotate,
        color=vector_color,
        linewidth=vector_width, 
        units='degrees', 
        layer="pol_vectors",
    )
    fig.refresh()

    # Plot the tau = 1 contour when optical depth is plotted    
    if render.lower() in ["tau", "optical depth"]:
        fig.show_contour(
            f'{source}_tau.fits', 
            levels=[1],
            colors='green',
        )

    return plot_checkout(fig, show, savefig, block=block)

def Tb(data, outfile="", freq=0, bmin=0, bmaj=0, overwrite=False, verbose=False):
    """
    Convert intensities [Jy/beam] into brightness temperatures [K].
    Frequencies must be in GHz and bmin and bmaj in arcseconds.
    Frequencies and beams are read from header if possible. 
    """

    # Detects whether data flux comes from a file or an array
    if isinstance(data, (str,PosixPath)):
        data, hdr = fits.getdata(data, header=True)
    else:
        hdr = {}

    # Drop empty axes
    flux = np.squeeze(data)

    # Get the frequency from the header if not provided
    if freq == 0:
        freq = hdr.get("RESTFRQ", default=freq) * u.Hz
        freq = freq.to(u.GHz)
    else:
        freq *= u.GHz

    # Get the beam minor and major axis from the header if not provided
    if bmin == 0:
        bmin = hdr.get("BMIN", default=bmin) * u.deg
        bmin = bmin.to(u.arcsec)
    else:
        bmin *= u.arcsec

    if bmaj == 0:
        bmaj = hdr.get("BMAJ", default=bmaj) * u.deg
        bmaj = bmaj.to(u.arcsec)
    else:
        bmaj *= u.arcsec

    print_(
        f"Reading BMIN and BMAJ from header: "
        + f'{bmin.to(u.arcsec):1.3f}" x '
        + f'{bmaj.to(u.arcsec):1.3f}"',
        verbose=verbose,
    )

    # Convert the beam gaussian stddev into a FWHM and obtain the beam area
    fwhm_to_sigma = 1 / np.sqrt(8 * np.log(2))
    beam = 2 * np.pi * bmaj * bmin * fwhm_to_sigma ** 2

    # Convert surface brightness (jy/beam) into brightness temperature
    to_Tb = u.brightness_temperature(freq)
    temp = flux * (u.Jy / beam).to(u.K, equivalencies=to_Tb)

    # Write flux to fits file if required
    write_fits(outfile, temp, hdr, overwrite, verbose)

    return temp.value



