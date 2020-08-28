#!/bin/bash
# Begin LSF Directives
#BSUB -P GEN010TOOLS
#BSUB -W 10
#BSUB -nnodes 1
#BSUB -J adios2-tau-test
#BSUB -o adios2-tau-test%J
#BSUB -e adios2-tau-test%J

module unload darshan-runtime
module load gcc/6.4.0
module load python/3.7.0-anaconda3-5.3.0

date
hostname

#export TAU_ADIOS2_PERIODIC=1
#export TAU_ADIOS2_PERIOD=5000000
#export TAU_ADIOS2_ONE_FILE=0
#export TAU_ADIOS2_ENGINE=BP
#export TAU_PLUGINS=libTAU-adios2-trace-plugin.so
#export TAU_PLUGINS_PATH=${MEMBERWORK}/gen010/tau2_install/ibm64linux/lib/shared-gnu-mpi-pthread-pdt-adios2

#ADIOS_PATH=${MEMBERWORK}/gen010/adios2_install
#export PYTHONPATH=${ADIOS_PATH}/lib64/python3.7/site-packages:${PYTHONPATH}
#export LD_LIBRARY_PATH=${ADIOS_PATH}/lib64:${LD_LIBRARY_PATH}
export TAU_PROFILE_FORMAT=merged

NSETS=1
NMPIS=20
NCORES=40
NSETSNODE=1
 
T="$(date +%s)"
jsrun -n ${NSETS} -a ${NMPIS} -c ${NCORES} -g 0 -r ${NSETSNODE} \
-b none ./matmult >& matmult.log
#-d packed -l cpu-cpu -b packed:1 ./matmult >& matmult.log
A="$(($(date +%s)-T))"

jslist -R

mv tauprofile.xml tauprofile-tau.xml
printf "Time to run clean: %02d hours %02d minutes %02d seconds.\n" "$((A/3600))" "$((A/60%60))" "$((A%60))"
