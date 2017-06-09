#!/bin/bash -x

DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SOS_CMD_PORT=22500
export SOS_DB_PORT=22503
export SOS_ROOT=/home3/khuck/src/sos_flow
export SOS_WORK=${DIR}/sos_flow_working
rm -rf ${SOS_WORK}
mkdir -p ${SOS_WORK}

start_sos_daemon()
{
    # start the SOS daemon

    if [ -z $1 ]; then echo "   >>> BATCH MODE!"; fi;
    if [ -z $1 ]; then echo "   >>> Starting the sosd daemons..."; fi;
    ${SOS_ROOT}/src/mpi.cleanall
    if [ -z $1 ]; then echo "   >>> Launching the sosd daemons..."; fi;
    daemon0="-np 1 ${SOS_ROOT}/bin/sosd --role SOS_ROLE_DAEMON --port ${SOS_CMD_PORT} --buffer_len 8388608 --listen_backlog 10 --work_dir ${SOS_WORK}"
    daemon1="-np 1 ${SOS_ROOT}/bin/sosd --role SOS_ROLE_DB     --port ${SOS_DB_PORT} --buffer_len 8388608 --listen_backlog 10 --work_dir ${SOS_WORK}"
    echo ${daemon0}
    echo ${daemon1}
    mpirun ${daemon0} : ${daemon1} &
    sleep 3
}

stop_sos_daemon()
{
    # shut down the daemon.
    ${SOS_ROOT}/bin/sosd_stop
    sleep 1
}

start_sos_daemon
mpirun -np 1 ./matmult
stop_sos_daemon
showdb

