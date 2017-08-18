#!/bin/bash

currdir=${PWD}
echo ${PWD}

export BEACON_TOPOLOGY_SERVER_ADDR=128.223.202.147
export BEACON_ROOT_FS=/dev/shm/aurelem
export LD_LIBRARY_PATH=/home/users/aurelem/beacon/mix/BEACON_inst/lib:$LD_LIBRARY_PATH
export PATH=/home/users/aurelem/beacon/mix/BEACON_inst/bin:/home/users/aurelem/beacon/mix/BEACON_inst/sbin:$PATH
export PYCOOLR_NODE=cerberus.nic.uoregon.edu
export PYCOOLR_LIBPATH=/home/users/aurelem/beacon/mix/BEACON_inst/lib
alias psbeacon="ps -aux | grep beacon"
killall -9 beacon_topology_setup_server
killall -9 global_beacon
beacon_topology_setup_server > ./setup_server.log 2>&1 &
sleep 1
global_beacon > ./global_beacon.log 2>&1 &

./src/pycoolr/pycoolr-plot/coolr.py src/pycoolr/pycoolr-plot/configs/beaconcerberus.cfg
