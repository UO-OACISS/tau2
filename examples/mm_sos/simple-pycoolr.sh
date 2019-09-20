#!/bin/bash
set -x

DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#export TAU_PLUGINS=libTAU-sos-plugin.so
#export TAU_PLUGINS_PATH=../../x86_64/lib/shared-mpi-pthread-pdt-sos

export SOS_IN_MEMORY_DATABASE=FALSE
export SOS_EXPORT_DB_AT_EXIT=FALSE

start_sos_daemon()
{
    # start the SOS daemon
    echo "Work directory is: $SOS_WORK"
    rm -rf sosd.00000.* profile.* dump.*
    daemon="mpirun -np 1 sosd -l 0 -a 1 -k 0 -r aggregator -w ${SOS_WORK}"
    echo ${daemon}
    ${daemon} >& sosd.log &
    sleep 1
}

stop_sos_daemon()
{
    # shut down the daemon.
    if pgrep -x "sosd" > /dev/null; then
        #sosd_stop
    mpirun -np 1 sosd_stop 
    fi
    sleep 5
}


stop_sos_daemon
make clean-sos
start_sos_daemon

unset TAU_VERBOSE
export TAU_SOS_PERIODIC=1

mpirun -np 2 tau_exec -T pdt,mpi,sos,pthread -sos ./matmult &

echo "sleep 1"
sleep 1

echo "Launch PyCOOLR"
../../tools/src//pycoolr/bin/pycoolr -tool=sos 

stop_sos_daemon

if pgrep -x "matmult" > /dev/null; then
        pkill mpirun
fi



