### Synthesizer Workshop 23.11.23
#
# The following script serves as an example usage of Synthesizer via its CLI
#
# The following example will create a mixture of four dust materials downloaded
# from the internet. The example tries to reproduce the DSHARP dust composition
# (Birnstiel et al. 2018).
#
# For details on every option available, uncomment the line below. Long output!
#!synthesizer --help


# First create opacities using a single material ('s': silicate)
synthesizer --opacity --material s

# Let us now use a predefined mix of silicate and graphite ('sg')
synthesizer -op --material sg --amax 1  --show-opac

# Let us now use another well known predefined mix
synthesizer -op --material dsharp --amax 10 --na 50 --show-nk --show-opac

# Now call the DustMixer and explicitly mix external optical constants
synthesizer 
    -op 
    --amax 10 
    --na 50 
    --show-nk 
    --show-opac 
    --material  h2o-w-Warren2008.lnk 
                astrosil-Draine2003.lnk 
                fes-Henning1996.lnk 
                c-org-Henning1996.lnk 
    --mfrac 0.2 0.3291 0.0743 0.3966 
    --mixing b 
    --polarization

