### Synthesizer Workshop 23.11.23
#
# The following script serves as an example usage of Synthesizer via its API
#
# For details on every option available, uncomment the line below. Long output!
#!synthesizer --help

from synthesizer.pipeline import Pipeline

# Initialize the pipeline
pipeline = Pipeline(
    nthreads = 1,
    amax = 10,
    overwrite = True,
)

# Create a protoplanetary disk
pipeline.create_grid(
    model = 'ppdisk',
    mdisk = 0.3,
    rin = 1,
    rout = 200,
    h0 = 5,
    flare = 0.3,
    ncells = 200,
    bbox = 200,
    temperature = True,
    show_2d = True,
    show_3d = True,
)

# Generate a temperature
pipeline.monte_carlo(
    nphot = 1e5,
    star = [0, 0, 0, 2.88, 1, 4000],
)

pipeline.raytrace(
    lam = 3000,
    incl = 40,
    phi = 60
    npix = 200,
    sizeau = 400,
    distance = 140,
    tau = True,
    show = True,
)

pipeline.synthetic_observation(
    script = None,
    simobserve = True,
    clean = True,
    exportfits = True,
    obstime = 1,
    resolution = 0.1,
    obsmode = 'int',
    use_template = True,
    telescope = 'ALMA',
    show = True,
    verbose = False,
)
