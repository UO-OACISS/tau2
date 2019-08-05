#!/bin/bash
set -x

DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SOS_CMD_PORT=22505
export SOS_WORK=${DIR}
export SOS_EVPATH_MEETUP=${DIR}

export TAU_PLUGINS=libTAU-sos-plugin.so
export TAU_PLUGINS_PATH=../../x86_64/lib/shared-mpi-pthread-pdt-sos

start_sos_daemon()
{
    # start the SOS daemon

    echo "Work directory is: $SOS_WORK"
    rm -rf sosd.00000.* profile.* dump.*
    #export SOS_IN_MEMORY_DATABASE=1
    daemon="../../../sos_flow/build/bin/sosd -l 0 -a 1 -k 0 -r aggregator -w ${SOS_WORK}"
    echo ${daemon}
    ${daemon} >& sosd.log &
    sleep 1
}

stop_sos_daemon()
{
    # shut down the daemon.
    if pgrep -x "sosd" > /dev/null; then
        ../../../sos_flow/build/bin/sosd_stop
    fi
    sleep 1
}

stop_sos_daemon
start_sos_daemon

# to use periodic, enable this variable, and comment out the
# TAU_SOS_send_data() call in matmult.c.
export TAU_VERBOSE=1
export TAU_SOS_PERIODIC=1
#unset TAU_SOS_PERIODIC
mpirun -np 4 ./matmult
#gdb --args ./matmult
sleep 1

stop_sos_daemon
../../../sos_flow/build/bin/showdb

