#!/bin/bash -l

# Example implementation of a radiative transfer modelling pipeline 
# using the command line interface, written in Bash. 
#
# Range of values for physical quantities are generated in the form of bash
# arrays and passed to the synthesizer extenally as command line arguments. 
#

# Define the base directory for all outputs. It is later appended by working_dir
basedir="$HOME/testdir/"

# Define some parameter grids
m=( "1e-3" "1e-5" )
mass=( "hm" "lm" )
flare=( "0" "0.5" "1" )
s=( "0.5" "5" )
sett=( "s" "ns" )
incl=( "75" "90" )

i=0
j=0
k=0
l=0

for mi in "${mass[@]}"
do
    for bi in "${flare[@]}"
    do
        for si in "${sett[@]}"
        do
            for ii in "${incl[@]}"
            do
                # Create and enter to the output folder
                working_dir=$basedir/${mi}/a1/b${bi}/${si}/trad/L2/sg/a10um/1.3mm/${ii}deg
                mkdir --parents --verbose $working_dir
                cd $working_dir
                echo -e "\n\n\nEntering $PWD"
            
                # Run the Synthesizer
                synthesizer 
                    --grid 
                    --model ppdisk 
                    --mdisk ${m[i]} 
                    --flare ${flare[j]} 
                    --h0 ${s[k]} 
                    --ncells 100 
                    --bbox 200 
                    --monte-carlo 
                    --nphot 1e5 
                    --raytrace 
                    --npix 400 
                    --sizeau 400 
                    --lam 1300 
                    --amax 10 
                    --na 50
                    --material sg 
                    --incl ${incl[l]} 
                    --use-template 
                    --nthreads 40 
                    --overwrite 
                (( l++ ))
            done
            (( k++ ))
        done
        (( j++ ))
    done
    (( i++ ))
done

exit 0
