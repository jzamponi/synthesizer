# Synthesizer
## Generate synthetic observations from 3D numerical simulations

Synthesizer is a program to calculate synthetic ALMA/JVLA images from an SPH model directly from the command-line. 
It interpolates SPH particle positions into a rectangular grid and then uses RADMC3D to do the Monte-Carlo and raytracing. It can use CASA to generate a final synthetic observation. It can also include the effects of polarization either by scattering or grain alignment.
Additionally, Synthesizer includes Dustmixer. This is a tool to generate dust opacity tables, including full scattering matrices, for a given dust composition (via optical constants) and allows to experiment with the mixing of different materials.  

## Instalation

    $ cd synthesizer
    $ python3 -m pip install -e .

## Example:


    $ synthesizer --grid --model ppdisk --ncells 100 --bbox 300
      --show-grid-2d --show-grid-3d --raytrace --polarization
     --opacity --synobs --show-rt --show-synobs


        
For details, run:

    $ synthesizer --help


Requisites:

    Software:   python3, CASA, RADMC3D, 
                Mayavi (optional), ParaView (optional)
        
    Modules:    python3-aplpy, python3-scipy, python3-numpy, python3-h5py
                python3-matplotlib, python3-astropy, python3-mayavi,

## Feedback

If you have any feedback, please reach out at jzamponi@mpe.mpg.de

