### Synthesizer Workshop 23.11.23
#
# The following script serves as an example usage of Synthesizer via its CLI
#
# For details on every option available, uncomment the line below. Long output!
#!synthesizer --help

synthesizer --grid --raytrace --monte-carlo --synobs 
    --model ppdisk
    --mdisk 0.1
    --rin 1
    --rout 200
    --h0 5
    --flare 0.3
    --ncells 200
    --bbox 200
    --show-grid-2d
    --show-grid-3d
    --nphot 1e5
    --star 0 0 0 2.88 1 4000
    --lam 3000
    --incl 40
    --npix 200
    --sizeau 400
    --distance 140
    --tau
    --show-rt
    --obstime 1 
    --resolution 0.1
    --obsmode sd
    --telescope ALMA
    --show-synobs
