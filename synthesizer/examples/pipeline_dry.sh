#!/bin/bash -l


# Define the base directory for all outputs. It is later appended by working_dir
basedir="$HOME/class0disks/results"

# Define the parameter grids
m=( "1e-3" "5e-5" )
mass=( "hm" "lm" )
flare=( "0" "0.5" "1" )
s=( "5" "0.5" )
sett=( "ns" "s" )
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
                working_dir=$basedir/${mi}/a1/b${bi}/${si}/trad/L2/sg/a100um/1.3mm/${ii}deg
                mkdir --parents --verbose $working_dir
                cd $working_dir
                echo -e "\n\n\nEntering $PWD"
            
                # Run the Synthesizer
#                echo """
#                synthesizer 
#                    --grid 
#                    --model ppdisk 
#                    --mdisk ${m[i]} 
#                    --flare ${flare[i]} 
#                    --h0 ${s[i]} 
#                    --ncells 400 
#                    --bbox 200 
#                    --monte-carlo 
#                    --nphot 1e7 
#                    --raytrace 
#                    --npix 400 
#                    --sizeau 400 
#                    --lam 1300 
#                    --amax 100 
#                    --material sg 
#                    --incl ${incl[i]} 
#                    --use-template 
#                    --nthreads 40 
#                    --overwrite 
#                
#
#                """
                (( l++ ))
            done
            (( k++ ))
        done
        (( j++ ))
    done
    (( i++ ))
done

exit 0
