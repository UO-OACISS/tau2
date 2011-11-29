#!/bin/bash

PS="/bin/ps"
GREP="/bin/grep"

PATHS="trunk"
COMPONENTS="utility otf2 opari2"
COMPILERS="gcc intel pgi studio"
KINDS_OF_LIB="static shared"
BITTEN_DIR="/home/roessel/bitten"

rm -f nohup.out

# for path in $PATHS; do
#     for component in $COMPONENTS; do
# 	for compiler in $COMPILERS; do
# 	    for kind_of_lib in $KINDS_OF_LIB; do
# 		CONFIG="scorep_${component}_${path}_${compiler}_${kind_of_lib}_${MACHINE}"
#                 MODULES="silc-autotools ${compiler}"
# 		LOGFILE="/tmp/bitten-${CONFIG}.log"
# 		MAIL_BODY="/tmp/bitten-${CONFIG}.mail"
# 		MASTER_URL="https://silc.zih.tu-dresden.de/trac-${component}/builds"
# 		BITTEN_CMD="bitten-slave -v --log=${LOGFILE} --config=${BITTEN_DIR}/config/${CONFIG}.ini ${MASTER_URL}"

# 		slave_still_running=`$PS ax | $GREP -v grep | $GREP "${BITTEN_CMD}"`
# 		if [ -n "$slave_still_running" ]; then
#                     # process still running, nothing to do here
# 		    rm -f ${LOGFILE} ${MAIL_BODY}
# 		else
# 		    echo "(re)starting bitten-slave ..." >  ${MAIL_BODY}
# 		    echo "${BITTEN_CMD}"                 >> ${MAIL_BODY}
# 		    echo "... in 10 seconds"             >> ${MAIL_BODY}
# 		    mail -s "deb32: Cron <roessel@atlantis>: (re)starting bitten-slave ${CONFIG}" c.roessel@fz-juelich.de < ${MAIL_BODY}
# 		    sleep 10
# 		    rm -f ${LOGFILE} ${MAIL_BODY}

# 		    # fork a subprocess that loads modules, logs the environment and starts the slave
# 		    ${BITTEN_DIR}/bin/start_bitten_slave.sh "$MODULES" $BITTEN_DIR $CONFIG "$BITTEN_CMD"
# 		fi
# 	    done
# 	done
#     done
# done



#########################################
# Component silc, i.e. with MPI
#########################################

MPIS="mpich2 openmpi impi"
component="silc"

for path in $PATHS; do
    for compiler in $COMPILERS; do
	for mpi in $MPIS; do
	    if [ "${mpi}" == "openmpi" ] && [ "${compiler}" == "studio" ]; then 
		continue 
	    fi
	    if [ "${mpi}" == "impi" ] && [ "${compiler}" == "studio" ]; then 
		continue 
	    fi
	    if [ "${mpi}" == "impi" ] && [ "${compiler}" == "pgi" ]; then 
		continue 
	    fi

	    for kind_of_lib in $KINDS_OF_LIB; do
		CONFIG="scorep_${component}_${path}_${compiler}_${mpi}_${kind_of_lib}_${MACHINE}"
		MODULES="silc-dev ${compiler} ${mpi}"
		LOGFILE="/tmp/bitten-${CONFIG}.log"
		MAIL_BODY="/tmp/bitten-${CONFIG}.mail"
		MASTER_URL="https://silc.zih.tu-dresden.de/trac-${component}/builds"
		BITTEN_CMD="bitten-slave -v --log=${LOGFILE} --config=${BITTEN_DIR}/config/${CONFIG}.ini ${MASTER_URL}"

		slave_still_running=`$PS ax | $GREP -v grep | $GREP "${BITTEN_CMD}"`
		if [ -n "$slave_still_running" ]; then
		    # process still running, nothing to do here
		    rm -f ${LOGFILE} ${MAIL_BODY}
		else
		    echo "(re)starting bitten-slave ..." >  ${MAIL_BODY}
		    echo "${BITTEN_CMD}"                 >> ${MAIL_BODY}
		    echo "... in 10 seconds"             >> ${MAIL_BODY}
		    mail -s "deb32: Cron <roessel@atlantis>: (re)starting bitten-slave ${CONFIG}" c.roessel@fz-juelich.de < ${MAIL_BODY}
		    sleep 10
		    rm -f ${LOGFILE} ${MAIL_BODY}

		    # fork a subprocess that loads modules, logs the environment and starts the slave
		    ${BITTEN_DIR}/bin/start_bitten_slave.sh "$MODULES" $BITTEN_DIR $CONFIG "$BITTEN_CMD"
		fi
	    done
	done
    done
done

# optional argument to bitten-slave:
# --log=${LOGFILE}
# --keep-files
# --dry-run
# -v