#!/bin/bash

set -x
#Make TAU report periodically.
export TAU_SOS_PERIODIC=1

#Plugin paths for TAU, -sos loads them
#export TAU_PLUGINS=libTAU-sos-plugin.so
#export TAU_PLUGINS_PATH=../../x86_64/lib/shared-mpi-pthread-pdt-sos

#Do not use TAU_VERBOSE to reduce logs, only needed if TAU fails or want to see
#information related to how TAU executes
unset TAU_VERBOSE
#export TAU_VERBOSE=1

#Execute the example application with two processes, each process will communicate to
#a different SOS daemon, usefull in the case the user wants to add a hostfile and 
#have one daemon and one application in one node and another daemon and app in
#another node.
mpirun \
	  -np 1 env SOS_CMD_PORT=22501 tau_exec -T mpi,pthread,sos,pdt -sos ./matmult \
	: -np 1 env SOS_CMD_PORT=22502 tau_exec -T mpi,pthread,sos,pdt  -sos ./matmult &


#Connect to the aggregator and report the selected metrics, the metrics can be changed 
#in report.c
env SOS_CMD_PORT=20690 ./report



#Check database with
#sqlite3 sosd.00000.db "SELECT frame, value_name, comm_rank,value from viewCombined WHERE (frame>0 )) ORDER BY value_name, frame, comm_rank ;"
#SOS should be stopped before the command is executed
