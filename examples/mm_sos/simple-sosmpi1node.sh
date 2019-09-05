#!/bin/bash
#Load SOS enviromental variables
source sosd.env.sourceme

set -x

DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SOS_CMD_PORT=22501
export SOS_WORK=${DIR}
export SOS_EVPATH_MEETUP=${DIR}

export SOS_DISCOVERY_DIR=${DIR}



export SOS_IN_MEMORY_DATABASE=TRUE
export SOS_EXPORT_DB_AT_EXIT=VERBOSE


export lis=2
export agg=1



start_sos_daemon()
{
    # start the SOS daemon

    echo "Work directory is: $SOS_WORK"
    rm -rf sosd.00000.* profile.* dump.*
    mpirun  \
	    -np 1 env SOS_CMD_PORT=20690 sosd -r aggregator -l $lis -a $agg -k 0  -w $(pwd) \
	  : -np 1 env SOS_CMD_PORT=22501 sosd -r listener   -l $lis -a $agg -k 1  -w $(pwd) \
	  : -np 1 env SOS_CMD_PORT=22502 sosd -r listener   -l $lis -a $agg -k 2  -w $(pwd) 
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

#stop_sos_daemon
./clean.sh
start_sos_daemon
