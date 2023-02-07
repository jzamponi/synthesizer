# Synthesizer
## Generate synthetic observations from 3D numerical simulations

Synthesizer is a program to calculate synthetic ALMA/JVLA images from an SPH model directly from the command-line. 
It interpolates SPH particle positions into a rectangular grid and then uses RADMC3D to do the Monte-Carlo and raytracing. It can use CASA to generate a final synthetic observation. It can also include the effects of polarization either by scattering or grain alignment.
Additionally, Synthesizer includes Dustmixer. This is a tool to generate dust opacity tables, including full scattering matrices, for a given dust composition (via optical constants) and allows to experiment with the mixing of different materials.  

Example:


    $ synthesizer --sphfile snap_001.dat --ncells 100 --bbox 50
      --show-grid-2d --show-grid-3d --render --raytrace --lam 3000  
      --polarization --opacity --material s --amin 0.1 --amax 10 --na 100 
      --q 3.5 --synobs --show-rt --show-synobs


        
For details, run:

    $ synthesizer --help


Requisites:

    Software:   python3, CASA, RADMC3D, 
                Mayavi (optional), ParaView (optional)
        
    Modules:    python3-aplpy, python3-scipy, python3-numpy, python3-h5py
                python3-matplotlib, python3-astropy, python3-mayavi,
                python3-radmc3dPy

