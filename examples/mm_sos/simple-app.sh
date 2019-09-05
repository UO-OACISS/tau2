#!/bin/bash

#Load SOS enviromental variables
source sosd.env.sourceme


set -x
export SOS_CMD_PORT=22501
export TAU_SOS_PERIODIC=1
export TAU_PLUGINS=libTAU-sos-plugin.so
export TAU_PLUGINS_PATH=../../x86_64/lib/shared-mpi-pthread-pdt-sos

#export TAU_VERBOSE=1
unset TAU_VERBOSE

#Execute tau with sos and use ebs to get code performance data
mpirun -np 1 tau_exec -ebs -T mpi,pthread,sos,pdt -sos ./matmult

#Wait a bit for the data to be saved to disk
sleep 2

#Stop SOS daemon--Only one process
mpirun -np 1 sosd_stop

sleep 2

#Check the code related performance data
sqlite3 sosd.00000.db "SELECT frame,value_name,value FROM viewCombined WHERE value_name LIKE '%[SAMPLE]%' ORDER by frame, value_name"
