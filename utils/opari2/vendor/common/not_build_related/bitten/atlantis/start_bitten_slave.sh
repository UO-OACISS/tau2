#!/bin/bash

MODULES="$1"
BITTEN_DIR="$2"
CONFIG="$3"
BITTEN_CMD="$4"

#echo $MODULES
#echo $BITTEN_DIR
#echo $CONFIG
#echo $BITTEN_CMD

# activate support for "environmental modules"
eval `tclsh /usr/local/module/modulecmd.tcl sh autoinit`
module load ${MODULES}

# remember the environment, for debugging purposes
mkdir -p ${BITTEN_DIR}/environment
env > ${BITTEN_DIR}/environment/cron_env_${CONFIG}
cat /etc/issue >> ${BITTEN_DIR}/environment/cron_env_${CONFIG}
uname -a >> ${BITTEN_DIR}/environment/cron_env_${CONFIG}

nohup nice -n 19 $BITTEN_CMD &
