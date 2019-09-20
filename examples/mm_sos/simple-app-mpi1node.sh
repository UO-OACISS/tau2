#!/bin/bash

set -x
export TAU_SOS_PERIODIC=1
#export TAU_PLUGINS=libTAU-sos-plugin.so
#export TAU_PLUGINS_PATH=../../x86_64/lib/shared-mpi-pthread-pdt-sos

export SOS_IN_MEMORY_DATABASE=TRUE
export SOS_EXPORT_DB_AT_EXIT=VERBOSE

unset TAU_VERBOSE
mpirun \
	  -np 1 env SOS_CMD_PORT=22501 tau_exec -T mpi,pthread,sos,pdt -sos ./matmult \
	: -np 1 env SOS_CMD_PORT=22502 tau_exec -T mpi,pthread,sos,pdt  -sos ./matmult &

env SOS_CMD_PORT=20690 ./report
