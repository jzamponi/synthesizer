#/usr/bin/env python3
"""
    Example pipeline script to generate several 
    Protoplanetary disk models using the Synthesizer. 
    This is the general radiative transfer moddelling approach.

    The Synthesizer does not include any internal looper for a 
    range of model or observational parameters. 
    Parameter space exploration must be done by explicitly calling 
    the program with values of interest. 
    
    For details on default values, formats and units, read the help:
    $ synthesizer --help

    In case variable types are still unclear, check the 
    parser.add_argument() section inside the parser.py file.
 
"""

import os
from pathlib import Path
from synthesizer import utils
from synthesizer.pipeline import Pipeline

cwd = Path.home()/'testdir'

for m, mass in zip([1e-3, 1e-5], ['hm', 'lm']):
    for flare in [0, 0.5, 1]:
        for s, settling in zip([5, 0.5], ['ns', 's']):
            for incl in [75, 90]:

                path_out = cwd/f'{mass}/a1/b{flare}/{settling}/trad/L2/sg/a100um/{incl}deg'
                os.makedirs(path_out, exist_ok=True)
                os.chdir(path_out)
            
                pipeline = Pipeline(
                    overwrite=True,
                    nthreads = 4,
                    amax = 10, 
                    material = 'sg',
                )

                pipeline.create_grid(
                    model = 'ppdisk',
                    mdisk = m,
                    rin = 1, 
                    rout = 200,
                    rc = 140, 
                    h0 = s, 
                    flare = flare,
                    bbox = 200,
                    ncells = 100,
                    show_2d=True,
                )

                pipeline.monte_carlo(
                    nphot = 1e5,
                    star = [0, 0, 0, 2.88, 1, 4000]
                )

                pipeline.raytrace(
                    lam = 1300,
                    incl = incl,
                    npix = 200,
                    sizeau = 400,
                    distance = 140,
                    show=True,
                )
