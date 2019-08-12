#!/bin/bash -e

# launch the writer without plugin
echo "Without plugin"
T="$(date +%s)"
#mpirun -np 8 ./matmult >& matmult.log
A="$(($(date +%s)-T))"
printf "Time to run tau: %02d hours %02d minutes %02d seconds.\n" "$((A/3600))" "$((A/60%60))" "$((A%60))"

# set up environment
source ./sourceme.sh

# Launch the writer with plugin - BP writer, no reader
export TAU_ADIOS2_ENGINE=BP
echo "BP plugin"
T="$(date +%s)"
#mpirun -np 8 ./matmult >& matmult.log
B="$(($(date +%s)-T))"
printf "Time to run tau+bp: %02d hours %02d minutes %02d seconds.\n" "$((B/3600))" "$((B/60%60))" "$((B%60))"

# cleanup
rm -rf *.bp *.bp.dir

# Launch the writer with plugin - SST writer and reader
# launch the reader - it will sleep 2 seconds and wait for files
export TAU_ADIOS2_FILENAME=tau-metrics
python3 ./reader.py &
unset TAU_ADIOS2_FILENAME

export TAU_ADIOS2_ENGINE=SST
echo "SST plugin"
T="$(date +%s)"
mpirun -np 8 ./matmult >& matmult.log
C="$(($(date +%s)-T))"
printf "Time to run tau+sst: %02d hours %02d minutes %02d seconds.\n" "$((C/3600))" "$((C/60%60))" "$((C%60))"

