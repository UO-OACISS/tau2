#!/bin/bash -x

#In order to run this example, please change the lines below that say UPDATE

DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TAU=/home/users/sramesh/MPI_T/TAU_INSTALLATION # UPDATE

export PATH=$TAU/86_64/bin:$TAU/x86_64/sos/sos_flow/build-linux/bin:$PATH #UPDATE

export SOS_CMD_PORT=22501
export SOS_WORK=`pwd`
export SOS_EVPATH_MEETUP=${DIR}
#export SOS_IN_MEMORY_DATABASE=1
# to use periodic, enable this variable, and comment out the
# TAU_SOS_send_data() call in matmult.c.
#export TAU_SOS_PERIODIC=1

export TAU_PLUGINS=libTAU-sos-plugin.so
export TAU_PLUGINS_PATH=$TAU/x86_64/lib/shared-mpi-pthread-sos-mpit # UPDATE

export TAU_VERBOSE=0
export TAU_PROFILE=1
export TAU_TRACK_MPI_T_PVARS=1

PLATFORM=`hostname`'.nic.uoregon.edu' # UPDATE

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

rm -rf sosd.00000.* profile.* dump.*
start_sos_daemon

ulimit -c unlimited

mpirun -np 4 tau_exec -T mpi,mpit,sos ./matmult &
sleep 1
echo "Launch PyCOOLR"

cd $TAU/x86_64/bin
./pycoolr -tool=sos -platform=$PLATFORM
#stop_sos_daemon
#showdb
