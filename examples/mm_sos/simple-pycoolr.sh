#!/bin/bash
set -x

DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#Plugin paths for TAU, -sos loads them
#export TAU_PLUGINS=libTAU-sos-plugin.so
#export TAU_PLUGINS_PATH=../../x86_64/lib/shared-mpi-pthread-pdt-sos

#Stores the database in disk so pycoolr has access to the database file
export SOS_IN_MEMORY_DATABASE=FALSE
export SOS_EXPORT_DB_AT_EXIT=FALSE


#Starts sos damon
#First remove old profile files
#executes the daemons and redirects the output to sosd.log
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

#Stops the SOS daemon
stop_sos_daemon()
{
    # shut down the daemon.
    if pgrep -x "sosd" > /dev/null; then
    	mpirun -np 1 sosd_stop 
    fi
    sleep 5
}


stop_sos_daemon
#Cleans old files
make clean-sos
start_sos_daemon

#TAU without additional console output
unset TAU_VERBOSE
#Makes TAU report periodically
export TAU_SOS_PERIODIC=1

#Executes the application with two processes in the same node
#and connects them to the same SOS daemon
mpirun -np 2 tau_exec -T pdt,mpi,sos,pthread -sos ./matmult &

#Gives enough time for TAU to reports all the available metrics
echo "sleep 5"
sleep 5

#Execute pycoolr from tools directory
echo "Launch PyCOOLR"
pycoolr -tool=sos 

#If pycoolr is close before the application finished, stop 
#SOS daemon and the application
stop_sos_daemon


if pgrep -x "matmult" > /dev/null; then
        pkill mpirun
fi



