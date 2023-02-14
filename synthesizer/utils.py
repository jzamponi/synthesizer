"""
    Collection of useful functions for my thesis.
"""
import os
import sys
import time
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

def print_(string, verbose=True, bold=False, red=False, fname=None, blue=False, 
    ul=False, *args, **kwargs):

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
        elif ul:
            print(f"{color.ul}[{fname}] {string} {color.ul}", flush=True, 
                *args, **kwargs)
        else:
            print(f"[{fname}] {string}", flush=True, 
                *args, **kwargs)


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


def file_exists(filename, raise_=True):
    """ Raise an error if a file doesnt exist. Supports linux wildcards. """

    msg = f'{color.red}{filename} not found.{color.none}'
    
    if '*' in filename:
        if len(glob(filename)) == 0:
            if raise_:
                raise FileNotFoundError(msg)
            else:
                return False
        else:
            if not raise_:
                return True
        
    else:
        if not os.path.exists(filename): 
            if raise_:
                raise FileNotFoundError(msg)
            else:
                return False
        else:
            if not raise_:
                return True

def download_file(url, msg=None, verbose=True, *args, **kwargs):
    """ Perform an HTTP GET request to fetch files from internet """ 

    if verbose:
        print_(f'Downloading file from {url}' if msg is None else msg, 
            *args, **kwargs)

    if 'github' in url and 'raw' not in url:
        raise ValueError("URLs of files from Github must come in raw format.")
    
    import requests

    # Strip the filename from the base url
    filename = url.split('/')[-1]

    # Perform an HTTP GET request
    req = requests.get(url)

    # Raise the HTTP Error if existent
    req.raise_for_status()

    # Download the file
    download = Path(filename).write_bytes(req.content)


def ring(soundfile=None):
    """ Play a sound from system. Useful to notify when a function finishes."""
    if not isinstance(soundfile, (str,PosixPath)):
        soundfile = "/usr/share/sounds/freedesktop/stereo/service-login.oga"

    os.system(f"paplay {soundfile} >/dev/null 2>&1")


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


def parse(s, delimiter="%", d=None):
    """
    Parse a string containing a given delimiter and return a dictionary
    containing the key:value pairs.
    """
    # Set the delimiter character
    delimiter = d if isinstance(d, (str,PosixPath)) else delimiter

    # Store all existing global and local variables
    g = globals()
    l = locals()

    string = s.replace(d, "{")
    string = s.replace("{_", "}_")

    # TO DO: This function is incomplete.
    return string


def set_hdr_to_iras16293B(
    hdr, 
    wcs="deg", 
    keep_wcs=False,
    spec_axis=False, 
    stokes_axis=False, 
    for_casa=False, 
    verbose=False, 
):
    """
    Adapt the header to match that of the ALMA observation of IRAS16293-2422B.
    Data from Maureira et al. (2020).
    """

    # Set the sky WCS to be in deg by default
    # and delete the extra WCSs
    if all([spec_axis, stokes_axis]):
        hdr["NAXIS"] = 4
    elif any([spec_axis, stokes_axis]):
        hdr["NAXIS"] = 3
    else:
        hdr["NAXIS"] = 3 if for_casa else 2

    keys = ["NAXIS", "CDELT", "CUNIT", "CRPIX", "CRVAL", "CTYPE"]

    WCS = {"deg": "A", "AU": "B", "pc": "C"}

    # TO DO: tell it to copy cdelt1A to cdelt1 only if more than one wcs exists.
    # Because, if no cdelt1A then it will set cdelt1 = None

    for n in [1, 2]:
        for k in keys[1:]:
            hdr[f"{k}{n}"] = hdr.get(f"{k}{n}{WCS[wcs]}", hdr.get(f"{k}{n}"))
            for a in WCS.values():
                hdr.remove(f"{k}{n}{a}", ignore_missing=True)

    for n in [3, 4]:
        for key in keys:
            hdr.remove(f"{key}{n}", ignore_missing=True)

    # Remove extra keywords PC3_* & PC*_3 added by CASA tasks and associated to a 3rd dim.
    if not any([spec_axis, stokes_axis, for_casa]):
        for k in ["PC1_3", "PC2_3", "PC3_3", "PC3_1", \
                    "PC3_2", "PC4_2", "PC4_3", "PC2_4", \
                    "PC3_4", "PC4_4", "PC4_1", "PC1_4"]:
            hdr.remove(k, True) 
            hdr.remove(k.replace('PC', 'PC0').replace('_', '_0'), True) 

    # Adjust the header to match obs. from IRAS16293-2422B
    if not keep_wcs:
        hdr["CUNIT1"] = "deg"
        hdr["CTYPE1"] = "RA---SIN"
        hdr["CRPIX1"] = 1 + hdr.get("NAXIS1") / 2
        hdr["CDELT1"] = hdr.get("CDELT1")
        hdr["CRVAL1"] = np.float64(248.0942916667)
        hdr["CUNIT2"] = "deg"
        hdr["CTYPE2"] = "DEC--SIN"
        hdr["CRPIX2"] = 1 + hdr.get("NAXIS2") / 2
        hdr["CDELT2"] = hdr.get("CDELT2")
        hdr["CRVAL2"] = np.float64(-24.47550000000)

    # Add spectral axis if required
    if spec_axis:
        # Convert the observing wavelength from the header into frequency
        wls = {
            "0.0013": {
                "freq": np.float64(230609583076.92307),
                "freq_res": np.float64(3.515082631882e10),
            },
            "0.003": {
                "freq": np.float64(99988140037.24495),
                "freq_res": np.float64(2.000144770049e09),
            },
            "0.007": {
                "freq": np.float64(42827493999.99999),
                "freq_res": np.float64(3.515082631882e10),
            },
            "0.0086": {
                "freq": np.float64(35e9),
                "freq_res": np.float64(1.0),
            },
            "0.0092": {
                "freq": np.float64(32.58614e9),
                "freq_res": np.float64(1.0),
            },
        }

        wl_from_hdr = str(hdr.get("HIERARCH WAVELENGTH1"))
        hdr["NAXIS3"] = 1
        hdr["CTYPE3"] = "FREQ"
        hdr["CRVAL3"] = wls[wl_from_hdr]["freq"]
        hdr["CRPIX3"] = np.float64(0.0)
        hdr["CDELT3"] = wls[wl_from_hdr]["freq_res"]
        hdr["CUNIT3"] = "Hz"
        hdr["RESTFRQ"] = hdr.get("CRVAL3")
        hdr["SPECSYS"] = "LSRK"

    # Add stokes axis if required
    if stokes_axis:
        hdr["NAXIS4"] = 1
        hdr["CTYPE4"] = "STOKES"
        hdr["CRVAL4"] = np.float32(1)
        hdr["CRPIX4"] = np.float32(0)
        hdr["CDELT4"] = np.float32(1)
        hdr["CUNIT4"] = ""

    # Add missing keywords (src: http://www.alma.inaf.it/images/ArchiveKeyworkds.pdf)
    hdr["BTYPE"] = "Intensity"
    hdr["BUNIT"] = "Jy/beam"
    hdr["BZERO"] = 0.0
    hdr["RADESYS"] = "ICRS"

    return hdr


def read_sph(snapshot="snap_541.dat", write_hdf5=False, remove_sink=True, cgs=True, verbose=False):
    """
    Notes:

    Assumes your binary file is formatted with a header stating the quantities,
    assumed to be f4 floats. May not be widely applicable.

    For these snapshots, the header is of the form...

    # id t x y z vx vy vz mass hsml rho T u

    where:

    id = particle ID number
    t = simulation time [years]
    x, y, z = cartesian particle position [au]
    vx,vy,vz = cartesian particle velocity [need to check units]
    mass = particle mass [ignore]
    hmsl = particle smoothing length [ignore]
    rho = particle density [g/cm3]
    T = particle temperature [K]
    u = particle internal energy [ignore]
    
    outlier_index = 31330
    """
    import h5py

    # Read file in binary format
    with open(snapshot, "rb") as f:
        print_(f'Reading file: {snapshot}', verbose)
        names = f.readline()[1:].split()
        data = np.frombuffer(f.read()).reshape(-1, len(names))
        data = data.astype("f4")

    # Turn data into CGS 
    if cgs:
        data[:, 2:5] *= u.au.to(u.cm)
        data[:, 8] *= u.M_sun.to(u.g)

    # Remove the cell data of the outlier, likely associated to a sink particle
    if remove_sink:
        sink_id = 31330
        data[sink_id, 5] = 0
        data[sink_id, 6] = 0
        data[sink_id, 7] = 0
        data[sink_id, 8] = data[:,8].min()
        data[sink_id, 9] = data[:,9].min()
        data[sink_id, 10] = 3e-11 
        data[sink_id, 11] = 900
    
    return data


def convert_opacity_file(
    infile='dust_mixture_001.dat', 
    outfile='dustkappa_polaris.inp', 
    verbose=True,
    show=True, 
):
    """
    Convert dust opacity files from POLARIS to RADMC3D format.
    """

    # Read polaris file with dust info
    print_(f'Reading in polaris dust file: {infile} in SI units', verbose)
    d = ascii.read(infile, data_start=10)
    
    # Store the wavelenght, absorption and scattering opacities and assymetry g
    lam = d['col1'] * u.m.to(u.micron)
    kabs = d['col18'] * (u.m**2/u.kg).to(u.cm**2/u.g)
    ksca = d['col20'] * (u.m**2/u.kg).to(u.cm**2/u.g)
    g_HG = d['col9']

    print_(f'Writing out radmc3d opacity file: {outfile} in CGS', verbose)
    with open(outfile, 'w+') as f:
        f.write('3\n')
        f.write(f'{len(lam)}\n')
        for i,l in enumerate(lam):
            f.write(f'{l:.6e}\t{kabs[i]:.6e}\t{ksca[i]:.6e}\t{g_HG[i]:.6e}\n')
    
    if show:
        print_(f'Plotting opacities ...', verbose)
        plt.loglog(lam, kabs, '--', c='black', label=r'$\kappa_{\rm abs}$')
        plt.loglog(lam, ksca, ':', c='black', label=r'$\kappa_{\rm sca}$')
        plt.loglog(lam, kabs+ksca, '-', c='black', label=r'$\kappa_{\rm ext}$')
        plt.legend()
        plt.xlabel('Wavelength (microns)')
        plt.ylabel(r'Dust opacity $\kappa$ (cm$^2$ g$^{-1}$)')
        plt.xlim(1e-1, 1e5)
        plt.ylim(1e-2, 1e4)
        plt.tight_layout()


def radmc3d_casafits(fitsfile='radmc3d_I.fits', radmc3dimage='image.out',
        stokes='I', dpc=141, verbose=False):
    """ Read in an image.out file created by RADMC3D and generate a
        FITS file with a CASA-compatible header, ready for a 
        synthetic observation.
    """

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
        
    img = np.loadtxt(radmc3dimage, skiprows=5)

    # Select Stokes map
    if iformat == 3:
        img = img[:, {'I': 0, 'Q': 1, 'U': 2, 'V': 3}[stokes]]
        
    # Make a squared map
    img = img.reshape(nx, ny)

    # Rescale to Jy/sr
    img = img * (
        u.erg * u.s ** -1 * u.cm ** -2 * u.Hz ** -1 * u.sr ** -1).to(
        u.Jy * u.sr ** -1
    )

    # Convert sr into pixels (Jy/sr --> Jy/pixel)
    img = img * (pixsize_x**2 / (dpc*u.pc.to(u.cm))**2)

    # Set a minimal header
    header = fits.Header()
    header.update({
        'CRPIX1': f'{nx // 2}',
        'CDELT1': f'',
        'CRVAL1': f'', 
        'CUNIT1': f'',
        'CTYPE1': f'',
        'CRPIX2': f'{nx // 2}',
        'CDELT2': f'',
        'CRVAL2': f'', 
        'CUNIT2': f'DEG',
        'CTYPE2': f'DEC--SIN',
        'RESTFRQ': f'{c.c.cgs.value / (lam*u.micron.to(u.cm))}',
        'BUNIT': 'Jy/pixel',
        'BTYPE': 'Intensity', 
        'BZERO': '1.0', 
        'BSCALE': 'BSCALE', 
        'LONPOLE': '180.0', 
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


def get_beam(filename, verbose=True):
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


def edit_header(filename, key, value, verbose=True):
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


@elapsed_time
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
    *args,
    **kwargs,
):
    """
    Plot a fits file using the APLPy library.
    """
    from aplpy import FITSFigure

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
        except Exception as e:
            print_(f'{e}', verbose)
            print_('Beam or frequency keywords not available. ' +\
            'Impossible to convert into T_b.', verbose, bold=True)
    
    write_fits(filename, data.squeeze(), hdr, True, verbose)

    # Remove non-celestial WCS
    filename_ = filename
    if hdr.get("NAXIS") > 2:
        tempfile = '.temp_file.fits'
        # <patch>: Use the header from the observation of IRAS16293B
        # but keep the phasecenter of the original input file
        print_('Setting the header to that from IRAS16293B', verbose, bold=True)
        hdr_ = set_hdr_to_iras16293B(hdr)
        write_fits(
            tempfile, 
            data=data.squeeze(), 
            header=hdr_, 
            overwrite=True
        )
        filename = tempfile    

    # Initialize the figure
    fig = FITSFigure(str(filename), figsize=figsize, *args, **kwargs)
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
                D = 141 * u.pc
                print_(f'Physical scalebar created for a distance of: {D}', verbose=False)
                scalebar_ = (scalebar.to(u.cm) / D.to(u.cm)) * u.rad.to(u.arcsec)
                fig.add_scalebar(scalebar_ * u.arcsec)
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

    return plot_checkout(fig, show, savefig, block=block) if checkout else fig


def pol_angle(stokes_q, stokes_u):
    """Calculates the polarization angle from Q and U Stokes component.

    Args:
        stokes_q (float): Q-Stokes component [Jy].
        stokes_u (float): U-Stokes component [Jy].

    Returns:
        float: Polarization angle.
    
    Disclaimer: This function is directly copied from polaris-tools.
    """
    # Polarization angle from Stokes Q component
    q_angle = 0.
    if stokes_q >= 0:
        q_angle = np.pi / 2.
    # Polarization angle from Stokes U component
    u_angle = 0.
    if stokes_u >= 0:
        u_angle = np.pi / 4.
    elif stokes_u < 0:
        if stokes_q >= 0:
            u_angle = np.pi * 3. / 4.
        elif stokes_q < 0:
            u_angle = -np.pi / 4.
    # x vector components from both angles
    x = abs(stokes_q) * np.sin(q_angle)
    x += abs(stokes_u) * np.sin(u_angle)
    # y vector components from both angles
    y = abs(stokes_q) * np.cos(q_angle)
    y += abs(stokes_u) * np.cos(u_angle)
    # Define a global direction of the polarization vector 
    # since polarization vectors are ambiguous in both directions.
    if x < 0:
        x *= -1.0
        y *= -1.0
    # Polarization angle calculated from Q and U components
    pol_angle = np.arctan2(y, x)

    return pol_angle, x, y


@elapsed_time
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
    from aplpy import FITSFigure

    # Enable verbosity
    global VERBOSE
    VERBOSE = verbose

    # Store the current path
    pwd = os.getcwd()

    # Read the Stokes components from a data cube
    if source in ['alma', 'vla', 'radmc3d']:
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

@elapsed_time
def tau_surface(
    densfile='dust_density_3d.fits.gz', 
    tempfile='dust_temperature_3d.fits.gz', 
    prefix='', 
    tau=1, 
    los=0, 
    bin_factor=[1,1,1], 
    render='temperature', 
    plot_tau=True, 
    amax='100um', 
    convolve_map=True, 
    plot2D=False, 
    plot3D=True, 
    savefig=None, 
    verbose=True
):
    """ 
        Compute and plot the surface with optical depth = 1 within a 3D 
        density array from a FITS file.
    """

    # Prepend the prefix to the filenames
    densfile = Path(prefix/Path(densfile))
    tempfile = Path(prefix/Path(tempfile))

    from astropy.nddata.blocks import block_reduce

    print_(f'Reading density from FITS file: {densfile}', verbose)
    print_(f'Reading temperature from FITS file: {tempfile}', verbose)
    rho = fits.getdata(densfile).squeeze() * (u.kg/u.m**3).to(u.g/u.cm**3)
    temp = fits.getdata(tempfile).squeeze()
    hdr = fits.getheader(densfile)

    # Read the delta length of a given axis from the header
    dl = hdr[f'cdelt{[3, 2, 1][los]}'] * (u.m).to(u.cm)

    # Bin the array down before plotting, if required
    if bin_factor not in [1, [1,1,1]]:
        if isinstance(bin_factor, (int, float)):
            bin_factor = [bin_factor, bin_factor, bin_factor]

        print_(f'Original array shape: {temp.shape}', verbose)

        print_(f'Binning density grid ...', verbose)
        rho = block_reduce(rho, bin_factor, func=np.nanmean)

        print_(f'Binning temperature grid ...', verbose)
        temp = block_reduce(temp, bin_factor, func=np.nanmean)

        # Rescale also the delta length by the binning factor
        dl *= bin_factor[0]

        print_(f'Binned array shape: {temp.shape}', verbose)

    # Dust opacities for a mixture of silicates and graphites in units of cm2/g
    if amax == '10um':
        # Extinction opacity at 1.3 and 3 mm for amax = 10um
        kappa_1mm = 1.50
        kappa_3mm = 0.60
        kappa_7mm = 0.23
        kappa_18mm = 0.07
    elif amax == '100um':
        # Extinction opacity at 1.3 and 3 mm for amax = 1000um
        kappa_1mm = 2.30
        kappa_3mm = 0.75
        kappa_7mm = 0.31
        kappa_18mm = 0.12
    elif amax == '1000um':
        # Extinction opacity at 1.3 and 3 mm for amax = 1000um
        kappa_1mm = 12.88
        kappa_3mm = 6.120
        kappa_7mm = 1.27
        kappa_18mm = 0.09

    # In the case of grain growth, combine the optical depth of the 2 dust pops
    if amax != '100-1000um':
        print_(f'Plotting for amax = {amax}', verbose)
        sigma_3d_1mm = (rho * kappa_1mm * dl)
        sigma_3d_3mm = (rho * kappa_3mm * dl)
        sigma_3d_7mm = (rho * kappa_7mm * dl)
        sigma_3d_18mm = (rho * kappa_18mm * dl)

    else:
        print_(f'Plotting for combined amax = {amax}', verbose)
        # The following opacities are for amax100um including organics 
        # and amax1000um without organics (i.e., sublimated).
        sigma_3d_1mm = np.where(temp > 300, rho * 12.88 * dl, rho * 1.80 * dl)
        sigma_3d_3mm = np.where(temp > 300, rho * 6.120 * dl, rho * 0.55 * dl)

    # Integrate the (density * opacity) product to calculate the optical depth
    op_depth_1mm = np.cumsum(sigma_3d_1mm, axis=los)
    op_depth_3mm = np.cumsum(sigma_3d_3mm, axis=los)
    op_depth_7mm = np.cumsum(sigma_3d_7mm, axis=los)
    op_depth_18mm = np.cumsum(sigma_3d_18mm, axis=los)

    if plot2D and not plot3D:
        from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel

        # Set all tau < 1 regions to a high number
        op_thick_1mm = np.where(op_depth_1mm < 1, op_depth_1mm.max(), op_depth_1mm)
        op_thick_3mm = np.where(op_depth_3mm < 1, op_depth_3mm.max(), op_depth_3mm)
        op_thick_7mm = np.where(op_depth_7mm < 1, op_depth_7mm.max(), op_depth_7mm)
        op_thick_18mm = np.where(op_depth_18mm < 1, op_depth_18mm.max(), op_depth_18mm)

        # Find the position of the minimum for tau > 1
        min_tau_pos_1mm = np.apply_along_axis(np.argmin, 0, op_thick_1mm).squeeze()
        min_tau_pos_3mm = np.apply_along_axis(np.argmin, 0, op_thick_3mm).squeeze()
        min_tau_pos_7mm = np.apply_along_axis(np.argmin, 0, op_thick_7mm).squeeze()
        min_tau_pos_18mm = np.apply_along_axis(np.argmin, 0, op_thick_18mm).squeeze()
    
        Td_tau1_1mm = np.zeros(temp[0].shape)
        Td_tau1_3mm = np.zeros(temp[0].shape)
        Td_tau1_7mm = np.zeros(temp[0].shape)
        Td_tau1_18mm = np.zeros(temp[0].shape)

        # Fill the 2D arrays with the temp. at the position of tau=1
        for i in range(min_tau_pos_1mm.shape[0]):
            for j in range(min_tau_pos_1mm.shape[1]):
                Td_tau1_1mm[i,j] = temp[min_tau_pos_1mm[i,j], i, j]
                Td_tau1_3mm[i,j] = temp[min_tau_pos_3mm[i,j], i, j]
                Td_tau1_7mm[i,j] = temp[min_tau_pos_7mm[i,j], i, j]
                Td_tau1_18mm[i,j] = temp[min_tau_pos_18mm[i,j], i, j]
        
        # Convolve the array with the beam from the observations at 1.3 and 3 mm
        def fwhm_to_std(obs):
            scale = dl * u.cm.to(u.pc)
            bmaj = obs.header['bmaj']*u.deg.to(u.rad) * (141 / scale) 
            bmin = obs.header['bmin']*u.deg.to(u.rad) * (141 / scale) 
            bpa = obs.header['bpa']
            std_x = bmaj / np.sqrt(8 * np.log(2))
            std_y = bmin / np.sqrt(8 * np.log(2))
            return std_x, std_y, bpa
            
        if convolve_map:
            print_('Convolving 2D temperature maps', verbose=True)
            std_x, std_y, bpa = fwhm_to_std(Observation('1.3mm'))
            Td_tau1_1mm = convolve_fft(Td_tau1_1mm, Gaussian2DKernel(std_x, std_y, bpa))

            std_x, std_y, bpa = fwhm_to_std(Observation('3mm'))
            Td_tau1_3mm = convolve_fft(Td_tau1_3mm, Gaussian2DKernel(std_x, std_y, bpa))

            std_x, std_y, bpa = fwhm_to_std(Observation('7mm'))
            Td_tau1_7mm = convolve_fft(Td_tau1_7mm, Gaussian2DKernel(std_x, std_y, bpa))
        
            std_x, std_y, bpa = fwhm_to_std(Observation('18mm'))
            Td_tau1_18mm = convolve_fft(Td_tau1_18mm, Gaussian2DKernel(std_x, std_y, bpa))
        return Td_tau1_1mm.T, Td_tau1_3mm.T, Td_tau1_7mm.T, Td_tau1_18mm.T

        
    if plot3D:
        from mayavi import mlab
        from mayavi.api import Engine
        from mayavi.sources.parametric_surface import ParametricSurface
        from mayavi.modules.text import Text

        # Initialaze the Mayavi scene
        engine = Engine()
        engine.start()
        fig = mlab.figure(size=(1500,1200), bgcolor=(1,1,1), fgcolor=(0.5,0.5,0.5))

        # Select the quantity to render: density or temperature
        if render in ['d', 'dens', 'density']:
            render_quantity = rho 
            plot_label = r'log(Dust density (kg m^-3))' 
        elif render in ['t', 'temp', 'temperature']:
            render_quantity = temp 
            plot_label = r'Dust Temperature (K)' 

        # Filter the optical depth lying outside of a given temperature isosurface, 
        # e.g., at T > 100 K.
        op_depth_1mm[temp < 150] = 0
        op_depth_3mm[temp < 150] = 0
        op_depth_7mm[temp < 150] = 0
        op_depth_18mm[temp < 150] = 0

        # Plot the temperature
        rendplot = mlab.contour3d(
            render_quantity, 
            colormap='inferno', 
            opacity=0.5, 
            vmax=400, 
            contours=10, 
        )
        figcb = mlab.colorbar(
            rendplot, 
            orientation='vertical', 
            title='',
        )

        # Add the axes and outline of the box
        mlab.axes(ranges=[-100, 100] * 3, 
            xlabel='AU', ylabel='AU', zlabel='AU', nb_labels=5)

        # Plot the temperature
        densplot = mlab.contour3d(
            rho, 
            colormap='BuPu', 
            opacity=0.5, 
            contours=5, 
        )
        denscb = mlab.colorbar(
            densplot, 
            orientation='vertical', 
            title='Dust Density (g/cm^3)',
        )
        if plot_tau:
            # Plot optical depth at 1mm
            tauplot_1mm = mlab.contour3d(
                op_depth_1mm, 
                contours=[tau], 
                color=(0, 1, 0), 
                opacity=0.5, 
            )
           # Plot optical depth at 3mm
            tauplot_3mm = mlab.contour3d(
                op_depth_3mm,  
                contours=[tau], 
                color=(0, 0, 1), 
                opacity=0.7, 
            )
           # Plot optical depth at 7mm
            tauplot_7mm = mlab.contour3d(
                op_depth_7mm,  
                contours=[tau], 
                color=(0.59, 0.41, 0.27), 
                opacity=0.7, 
            )
           # Plot optical depth at 7mm
            tauplot_18mm = mlab.contour3d(
                op_depth_18mm,  
                contours=[tau], 
                color=(0.75, 0.34, 0.79), 
                opacity=0.7, 
            )

        # The following commands are meant to customize the scene and were 
        # generated with the recording option of the interactive Mayavi GUI.

        # Adjust the viewing angle for an edge-on projection
        scene = engine.scenes[0]
        scene.scene.camera.position = [114.123, -161.129, -192.886]
        scene.scene.camera.focal_point = [123.410, 130.583, 115.488]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [-0.999, -0.017, 0.014]
        scene.scene.camera.clipping_range = [90.347, 860.724]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.render()

        # Adjust the light source of the scene to illuminate the disk from the viewer's POV 
        #camera_light1 = engine.scenes[0].scene.light_manager.lights[0]
        #camera_light1 = scene.scene.light_manager.lights[0]
        #camera_light1.elevation = 90.0
        #camera_light1.intensity = 1.0
        ##camera_light2 = engine.scenes[0].scene_light.manager.lights[1]
        #camera_light2 = scene.scene_light.manager.lights[1]
        #camera_light2.elevation = 90.0
        #camera_light2.elevation = 0.7

        # Customize the iso-surfaces
        module_manager = engine.scenes[0].children[0].children[0]
        temp_surface = engine.scenes[0].children[0].children[0].children[0]
#        temp_surface.contour.minimum_contour = 70.0
#        temp_surface.contour.maximum_contour = 400.0
        temp_surface.actor.property.representation = 'surface'
        temp_surface.actor.property.line_width = 3.0
        if plot_tau:
            tau1_surface = engine.scenes[0].children[1].children[0].children[0]
            tau3_surface = engine.scenes[0].children[2].children[0].children[0]
            tau7_surface = engine.scenes[0].children[1].children[0].children[0]
            tau18_surface = engine.scenes[0].children[2].children[0].children[0]
            tau1_surface.actor.property.representation = 'wireframe'
            tau3_surface.actor.property.representation = 'wireframe'
            tau7_surface.actor.property.representation = 'wireframe'
            tau18_surface.actor.property.representation = 'wireframe'
            tau1_surface.actor.property.line_width = 3.0
            tau3_surface.actor.property.line_width = 4.0
            tau7_surface.actor.property.line_width = 3.0
            tau18_surface.actor.property.line_width = 4.0

        # Adjust the colorbar
        lut = module_manager.scalar_lut_manager
        lut.scalar_bar_representation.position = np.array([0.02, 0.2])
        lut.scalar_bar_representation.position2 = np.array([0.1, 0.63])
        lut.label_text_property.bold = False
        lut.label_text_property.italic = False
        lut.label_text_property.font_family = 'arial'
        lut.data_range = np.array([70., 400.])

        # Add labels as text objects to the scene
        parametric_surface = ParametricSurface()
        engine.add_source(parametric_surface, scene)        

        label1 = Text()
        engine.add_filter(label1, parametric_surface)
        label1.text = plot_label
        label1.property.font_family = 'arial'
        label1.property.shadow = True
        label1.property.color = (0.86, 0.72, 0.21)
        label1.actor.position = np.array([0.02, 0.85])
        label1.actor.width = 0.30

        if plot_tau:
            label2 = Text()
            engine.add_filter(label2, parametric_surface)
            label2.text = 'Optically thick surface at 1.3mm'
            label1.property.font_family = 'arial'
            label2.property.color = (0.31, 0.60, 0.02)
            label2.actor.position = np.array([0.02, 0.95])
            label2.actor.width = 0.38

            label3 = Text()
            engine.add_filter(label3, parametric_surface)
            label3.text = 'Optically thick surface at 3mm'
            label2.property.font_family = 'arial'
            label3.property.color = (0.20, 0.40, 0.64)
            label3.actor.position = [0.02, 0.915]
            label3.actor.width = 0.355

            label4 = Text()
            engine.add_filter(label4, parametric_surface)
            label4.text = 'Line of Sight'
            label4.property.font_family = 'times'
            label4.property.color = (0.8, 0.0, 0.0)
            label4.actor.position = np.array([0.63, 0.90])
            label4.actor.width = 0.20

        if savefig is not None:
            scene.scene.save(savefig)

        if plot_tau:
            return render_quantity, op_depth_1mm, op_depth_3mm, op_depth_7mm, op_depth_18mm
        else:
            return render_quantity

