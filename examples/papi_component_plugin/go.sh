#!/bin/bash -e

export TAU_PLUGINS=libTAU-papi-components-plugin.so
export TAU_PLUGINS_PATH=../../x86_64/lib/shared-papi-mpi-pthread

# launch the program with the plugin
echo "With PAPI component plugin"
T="$(date +%s)"
mpirun -np 4 ./matmult
#gdb --args ./matmult
A="$(($(date +%s)-T))"
printf "Time to run tau: %02d hours %02d minutes %02d seconds.\n" "$((A/3600))" "$((A/60%60))" "$((A%60))"


