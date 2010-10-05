#!/bin/bash

#qsub -l size=768 ./jobscript.sh
#qsub -l size=1536 ./jobscript_mpi.sh
#qsub -l size=3072 ./jobscript_mpi.sh # Not enough memory
#qsub -l size=4104 ./jobscript_mpi.sh
qsub -l size=6144 ./jobscript_mpi.sh
qsub -l size=8196 ./jobscript_mpi.sh
qsub -l size=12288 ./jobscript_mpi.sh
qsub -l size=16392 ./jobscript_mpi.sh
qsub -l size=24576 ./jobscript_mpi.sh
