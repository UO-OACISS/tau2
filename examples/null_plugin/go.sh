#!/bin/bash -e

export TAU_PROFILE_FORMAT=merged
#export TAU_SAMPLING=1
#export TAU_MEASURE_TAU=1
nprocs=4

# launch the writer without plugin
echo "Without plugin"
T="$(date +%s)"
mpirun -np ${nprocs} ./matmult
A="$(($(date +%s)-T))"
mv tauprofile.xml tauprofile-without-plugin.xml

# set up environment
source ./sourceme.sh

# Launch the writer with null plugin
echo "Null plugin"
T="$(date +%s)"
mpirun -np ${nprocs} ./matmult
B="$(($(date +%s)-T))"
mv tauprofile.xml tauprofile-with-plugin.xml

printf "Time to run tau: %02d hours %02d minutes %02d seconds.\n" "$((A/3600))" "$((A/60%60))" "$((A%60))"
printf "Time to run tau+plugin: %02d hours %02d minutes %02d seconds.\n" "$((B/3600))" "$((B/60%60))" "$((B%60))"

