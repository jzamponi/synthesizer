#!/usr/bin/env python3
"""
    Pipeline script to calculate synthetic ALMA/JVLA images from an SPH model,
    using RADMC3D for the raytracing and CASA for the synthetic observation. 

    Example:

    $ synthesizer --sphfile snap_001.dat --ncells 100 --bbox 50
        --show-grid-2d --show-grid-3d --raytrace --lam 3000 --amax 10 
        --polarization --opacity --material p --amin 0.1 --amax 10 --na 100 
        --show-rt --synobs --show-synobs

    For details, run:
    $ synthesizer --help

    Requisites:
        Software:   python3, CASA, RADMC3D, 
                    Mayavi (optional), ParaView (optional)

        Modules:    python3-aplpy, python3-scipy, python3-numpy, python3-h5py
                    python3-matplotlib, python3-astropy, python3-mayavi,
                    python3-radmc3dPy

"""

from synthesizer import parser

__version__ = "1.0.0"

if __name__ == "__main__":
    parser.parser()
