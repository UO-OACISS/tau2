#!/bin/bash

set -x

#Store database in memory as it is not needed in disk at runtime for this example
#Save the database to disk at the end and report it.
export SOS_IN_MEMORY_DATABASE=TRUE
export SOS_EXPORT_DB_AT_EXIT=VERBOSE

#Declare the number of listeners and aggregators
export lis=2
export agg=1



start_sos_daemon()
{
    # start the SOS daemon
    echo "Work directory is: $SOS_WORK"
    rm -rf sosd.00000.* profile.* dump.*
    #Execute three sos DAEMONS, one aggregator and two listeners,
    #In this way it is possible to have one listeners tied to a process 
    #in differents machines using hostfiles with mpi
    mpirun  \
	    -np 1 env SOS_CMD_PORT=20690 sosd -r aggregator -l $lis -a $agg -k 0  -w $(pwd) \
	  : -np 1 env SOS_CMD_PORT=22501 sosd -r listener   -l $lis -a $agg -k 1  -w $(pwd) \
	  : -np 1 env SOS_CMD_PORT=22502 sosd -r listener   -l $lis -a $agg -k 2  -w $(pwd) 
    sleep 1
}

#Clean old files
make clean-sos
#Start the daemons
start_sos_daemon
