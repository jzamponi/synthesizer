### Synthesizer Workshop 23.11.23
#
# The following script serves as an example usage of Synthesizer via its API
#
# The following example will create a mixture of four dust materials downloaded
# from the internet. The example tries to reproduce the DSHARP dust composition
# (Birnstiel et al. 2018).
#
# For details on every option available, uncomment the line below. Long output!
#!synthesizer --help

from synthesizer.pipeline import Pipeline

# Initialize the pipeline
pipeline = Pipeline(
    nthreads = 1,
    overwrite = True,
)

# First create opacities using a single material ('s': silicate)
pipeline.dustmixer(
    material = 's',
    amax = 10, 
    show_nk = True,
    show_opac = True
)

# Let us now use a predefined mix of silicate and graphite ('sg')
pipeline.dustmixer(
    material = 'sg',
    amax = 10, 
    show_nk = True,
    show_opac = True
)

# Let us now use another well known predefined mix
pipeline.dustmixer(
    material = 'dsharp',
    amax = 10, 
    show_nk = True,
    show_opac = True
)

# Now call the DustMixer and explicitly mix external optical constants
repo = 'https://raw.githubusercontent.com/jzamponi/synthesizer/main/synthesizer/dustmixer/nk'
pipeline.dustmixer(
    material = [
        f'{repo}/h2o-w-Warren2008.lnk',
        f'{repo}/astrosil-Draine2003.lnk',
        f'{repo}/fes-Henning1996.lnk',
        f'{repo}/c-org-Henning1996.lnk',
        ],
    mfrac = [0.2, 0.3291, 0.0743, 0.3966],
    mixing = 'b', 
    amin = 0.1, 
    amax = 10, 
    q = -3.5, 
    na = 100, 
    polarization = False, 
    show_nk = True,
    show_opac = True
)

