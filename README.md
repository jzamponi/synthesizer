<p align="center">
<img width="40%" src="https://raw.githubusercontent.com/jzamponi/synthesizer/main/synthesizer/img/logo_and_title_white.jpg" />
</p>

## Generate synthetic observations from 3D numerical simulations

Synthesizer is a program to calculate synthetic images from either analytical 
models or hydrodynamical simulations (both grid and particle based). 
You can use Synthesizer via API and CLI. 
Command-line interaction offers a fast way to test your modelling setup, without 
the need to write any scripts. 
API interaction comes in handy during the production phase. Its modular form 
allows to easily script a full modelling pipeline by simply calling the relevant 
modules of Synthesizer. Pipelines templates are included within the examples/ 
source directory.

For the post-processing of particle based simulations, Synthesizer interpolates 
particle positions into a rectangular grid and then performs Monte-Carlo heating 
and raytracing, using the RADMC3D radiative transfer code. 
It can also optionally use the CASA software to generate a final synthetic 
observation of your model. 

Polarization effects, either by scattering or grain alignment, can easily be 
added to your setup. 

Additionally, Synthesizer includes the DustMixer module. This is a tool to 
easily generate dust opacity tables for a given dust composition, supporting 
full scattering matrices for polarization observations.   
This module serves as a testing tool to experiment mixing different materials 
and different laboratory measurements of optical constants.   

## Installation
    $ pip install astro-synthesizer
    

## Run the code:

Synthesizer requires at least one of its five main options to run:



    $ synthesizer
    [synthesizer] Nothing to do. Main options: --grid, --opacity, --monte-carlo, --raytrace or --synobs
    [synthesizer] Run synthesizer --help for details.



#### Example
Create a protoplanetary disk model, containing dust grains of DSHARP composition, 
let RADMC3D generate an image and then let CASA observe it (the order of command-
line arguments is irrelevant).

    $ synthesizer --grid --model ppdisk --temperature --raytrace --synobs 
      --opacity --material dsharp --show-grid-2d --show-opac --show-rt --show-synobs

Given the optional --show-* flags, synthesizer will plot the results of every step. 
You can also read in snapshots from an SPH simulation:



    $ synthesizer --grid --sphfile snapshot_001.h5 --source gizmo  
     --show-grid-2d --show-grid-3d --raytrace --synobs 
     --opacity --show-rt --show-synobs


Already existing results can also be displayed without having to re-run a full step, 
using commands like


    $ synthesizer --show-rt --show-synobs --show-opac --show-grid-2d --show-mc-3d


Example API implementations and modelling pipelines are distributed within the 
source code and can be found in the examples/ directory.

        
For details, run:

    $ synthesizer --help


#### Note on 3D visualization
Synthesizer uses the Mayavi library for visualization and rendering of 3D 
data. Mayavi's current latest version presents [issues](https://github.com/enthought/mayavi/issues/1284) 
on installation via pip and therefore breaks Synthesizer's installation. 
If you intend to use the --show-grid-3d or --show-mc-3d options, please install 
mayavi manually from the following source:


    $ pip install https://github.com/enthought/mayavi/zipball/master


#### Compatibility with hydrodynamical codes 
The coupling of hydrodynamical codes from different origins is now implemented in
Synthesizer using the snapshot readers from the YT Python project.
If your code is supported by YT, then it works with Synthesizer (beta).

Synthesizer does currently only work with Cartesian grids. 
Implementation of spherical grids is currently under development. 

For particularly customized hydro readers, use the --source option (or source 
function argument in you call to the create_grid module).

If your code is not currently supported and you are interested in using this tool, 
feel free to get in touch. 

### Requisites:

    Software:   python3, RADMC3D, CASA, Mayavi/ParaView (optional)
        
### Feedback

If you find any bugs or have any feedback, please feel free to reach out at joaquin.zamponi@gmail.com.


