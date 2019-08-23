#!/bin/bash
#Load SOS enviromental variables
source sosd.env.sourceme

set -x

DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SOS_CMD_PORT=22501
export SOS_WORK=${DIR}
export SOS_EVPATH_MEETUP=${DIR}

export SOS_DISCOVERY_DIR=${DIR}


start_sos_daemon()
{
    # start the SOS daemon

    echo "Work directory is: $SOS_WORK"
    rm -rf sosd.00000.* profile.* dump.*
    #export SOS_IN_MEMORY_DATABASE=1
    daemon="sosd -l 0 -a 1 -k 0 -r aggregator -w ${SOS_WORK}"
    echo ${daemon}
    ${daemon} #>& sosd.log &
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

stop_sos_daemon
./clean.sh
start_sos_daemon
