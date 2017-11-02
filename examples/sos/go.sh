#!/bin/bash -x

DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f ../../x86_64/sos/sos_flow/build-linux/bin/sosd ] ; then
    PATH=$PATH:$DIR/../../x86_64/sos/sos_flow/build-linux/bin/
fi

export SOS_CMD_PORT=22500
export SOS_WORK=${DIR}
export SOS_EVPATH_MEETUP=${DIR}
# to use periodic, enable this variable, and comment out the
# TAU_SOS_send_data() call in matmult.c.
# export TAU_SOS_PERIODIC=1
export TAU_SOS_HIGH_RESOLUTION=1
export TAU_SOS=1

start_sos_daemon()
{
    # start the SOS daemon

    daemon="sosd -l 0 -a 1 -k 0 -r aggregator -w ${SOS_WORK}"
    echo ${daemon}
    ${daemon} &
    sleep 3
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
stop_sos_daemon
rm -rf sosd.00000.*
start_sos_daemon
mpirun -np 4 ./matmult
stop_sos_daemon
showdb

