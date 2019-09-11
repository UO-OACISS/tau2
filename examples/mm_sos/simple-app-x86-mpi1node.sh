#!/bin/bash

#Load SOS enviromental variables
source sosd.env.sourceme


set -x
export SOS_CMD_PORT=22501
export TAU_SOS_PERIODIC=1
export TAU_PLUGINS=libTAU-sos-plugin.so
export TAU_PLUGINS_PATH=../../x86_64/lib/shared-mpi-pthread-pdt-sos
#export TAU_PLUGINS_PATH=../../x86_64/lib/shared-icpc-mpi-pthread-pdt-sos


export SOS_IN_MEMORY_DATABASE=TRUE
export SOS_EXPORT_DB_AT_EXIT=VERBOSE




#export TAU_VERBOSE=1
unset TAU_VERBOSE
#export TAU_METRICS=TIME,LIKWID_L1D_REPLACEMENT:PMC0

#Execute tau with sos and use ebs to get code performance data
#tau_exec -ebs -T likwid,mpi,pthread,sos,pdt -sos ./matmult



mpirun \
	  -np 1 env SOS_CMD_PORT=22501 tau_exec -T mpi,pthread,sos,pdt -sos ./matmult \
	: -np 1 env SOS_CMD_PORT=22502 tau_exec -T mpi,pthread,sos,pdt  -sos ./matmult &

sleep 2

env SOS_CMD_PORT=20690 ./report
