#!/bin/bash 

if [ "x$PE_MPICH_GTL_DIR_amd_gfx908" != "x" ]; then 
  MPI_EXTRA_OPTS="${PE_MPICH_GTL_DIR_amd_gfx908} ${PE_MPICH_GTL_LIBS_amd_gfx908} "
fi

MPIOPTS=`mpicc -show | awk '{$1=""}1'  `
echo ${MPIOPTS}  ${MPI_EXTRA_OPTS}
  
