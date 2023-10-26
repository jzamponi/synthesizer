<p align="center">
<img width="40%" src="https://raw.githubusercontent.com/jzamponi/synthesizer/main/synthesizer/img/logo_and_title_white.jpg" />
</p>

## Generate synthetic observations from 3D numerical simulations

Synthesizer is a program to calculate synthetic ALMA/JVLA images from an SPH model directly from the command-line. 
It interpolates SPH particle positions into a rectangular grid and then uses RADMC3D to do the Monte-Carlo and raytracing. It can use CASA to generate a final synthetic observation. It can also include the effects of polarization either by scattering or grain alignment.
Additionally, Synthesizer includes Dustmixer. This is a tool to generate dust opacity tables, including full scattering matrices, for a given dust composition (via optical constants) and allows to experiment with the mixing of different materials.  

## Installation
    $ pip install astro-synthesizer
    

## Run the code:

Synthesizer requires at least one of its five main options to run:



    $ synthesizer
    [synthesizer] Nothing to do. Main options: --grid, --opacity, --monte-carlo, --raytrace or --synobs
    [synthesizer] Run synthesizer --help for details.



#### Example
Create a protoplanetary disk model, let radmc3d generate an image and then casa observe it.

    $ synthesizer --grid --model ppdisk --show-grid-2d 
      --temperature --show-grid-3d --raytrace --synobs
      --opacity --show-rt --show-synobs

Given the --show-* flags, synthesizer will plot the results of every step. You can also read in snapshots from an SPH simulation:



    $ synthesizer --grid --sphfile snapshot_001.h5 --source gizmo  
     --show-grid-2d --show-grid-3d --raytrace --synobs 
     --opacity --show-rt --show-synobs


Previous results can also be shown without having to re-run a full step, with commands like


    $ synthesizer --show-rt --show-synobs --show-opac --show-grid-2d



        
For details, run:

    $ synthesizer --help


#### Compatibility with hydrodynamical codes 
Currently supported (M)HD codes:

    - GIZMO
    - AREPO
    - GADGET
    - ZeusTW

Synthesizer is very young and at the moment only snapshots from the above listed codes are supported. Creating interfaces to new codes is easy to implement but it takes time. For AMR codes, grid and density (and optionally temperature) information is needed to generate the input for RADMC3D. For SPH codes, all synthesizer needs are point coordinates x, y, z and density (and optionally temperature), all in cgs units. It does the gridding by interpolating point coordinates into a regular cartesian mesh. 
If you're interested in using synthesizer and your hydro code is not yet supported, feel free to get in contact. The implementation should be quite strightforward.  

#### Requisites:

    Software:   python3, CASA, RADMC3D, ParaView (optional)
        
### Feedback

If you have any feedback, please reach out at jzamponi@mpe.mpg.de.


