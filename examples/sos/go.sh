#!/bin/bash -x

DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#if [ -f ../../x86_64/sos/sos_flow/build-linux/bin/sosd ] ; then
#    PATH=$PATH:$DIR/../../x86_64/sos/sos_flow/build-linux/bin/
#fi

if [ -f ../../x86_64/sos/sos_flow/inst/bin/sosd ] ; then
    PATH=$PATH:$DIR/../../x86_64/sos/sos_flow/inst/bin/
elif [ -f $HOME/src/sos_flow/build/bin/sosd ] ; then
    PATH=$PATH:$HOME/src/sos_flow/build/bin
fi

#export SOS_CMD_PORT=22500
export SOS_CMD_PORT=22501
export SOS_WORK=${DIR}
export SOS_EVPATH_MEETUP=${DIR}
#export SOS_IN_MEMORY_DATABASE=1
# to use periodic, enable this variable, and comment out the
# TAU_SOS_send_data() call in matmult.c.
#export TAU_SOS_PERIODIC=1

#export TAU_PLUGINS=libTAU-sos-plugin.so
#export TAU_PLUGINS_PATH=/home/khuck/src/tau2/x86_64/lib/shared-papi-mpi-pthread-pdt-sos-adios
export TAU_PLUGINS=libTAU-sos-plugin.so
export TAU_PLUGINS_PATH=/home/users/aurelem/tau/tau2/x86_64/lib/shared-mpi-pthread-pdt-sos-mpit
#export TAU_PLUGINS_PATH=/home/users/aurelem/tau/tau2/x86_64/lib/shared-mpi-pthread-pdt-sos

export TAU_VERBOSE=1
export TAU_PROFILE=1
export TAU_TRACK_MPI_T_PVARS=1

PLATFORM=delphi.nic.uoregon.edu

start_sos_daemon()
{
    # start the SOS daemon

    echo $SOS_WORK
    daemon="sosd -l 0 -a 1 -k 0 -r aggregator -w ${SOS_WORK}"
    echo ${daemon}
    ${daemon} &
    sleep 1
}

stop_sos_daemon()
{
    # shut down the daemon.
    if pgrep -x "sosd" > /dev/null; then
        sosd_stop
    fi
    sleep 1
}

# start clean
pkill -9 sosd
pkill -9 pycoolr
pkill -9 python
#stop_sos_daemon
rm -rf sosd.00000.* profile.* dump.*
start_sos_daemon
export TAU_METRICS=TIME:PAPI_FP_OPS
ulimit -c unlimited
mpirun -np 4 ./matmult &
#gdb ./matmult 
#mpirun -np 2 ./matmult 
#./matmult 
sleep 1
echo "Launch PyCOOLR"
cd ../../x86_64/bin
./pycoolr -tool=sos -platform=$PLATFORM
stop_sos_daemon
showdb

