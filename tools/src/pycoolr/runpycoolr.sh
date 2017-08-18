#!/bin/bash

currdir=${PWD}
echo ${PWD}

PYCOOLR_PLATFORM="cerberus.nic.uoregon.edu"
PYCOOLR_TOOL="beacon"

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

if [[ $PYCOOLR_TOOL = "beacon" ]];
then

  if [[ $PYCOOLR_PLATFORM = "cerberus.nic.uoregon.edu" ]];
  then
  echo "tool: beacon, platform: cerberus.nic.uoregon.edu"
  ./src/pycoolr/pycoolr-plot/coolr.py src/pycoolr/pycoolr-plot/configs/beaconcerberus.cfg

  elif [[ $PYCOOLR_PLATFORM = "godzilla.nic.uoregon.edu" ]];
  then
  echo "tool: beacon, platform: godzilla.nic.uoregon.edu"
  ./src/pycoolr/pycoolr-plot/coolr.py src/pycoolr/pycoolr-plot/configs/beacongodzilla.cfg

  elif [[ $PYCOOLR_PLATFORM = "delphi.nic.uoregon.edu" ]];
  then
  echo "tool: beacon, platform: delphi.nic.uoregon.edu"
  ./src/pycoolr/pycoolr-plot/coolr.py src/pycoolr/pycoolr-plot/configs/beacondelphi.cfg

  fi

elif [[ $PYCOOLR_TOOL = "sos" ]];
  then
  if [[ $PYCOOLR_PLATFORM = "cerberus.nic.uoregon.edu" ]];
  then
  echo "tool: sos, platform: cerberus.nic.uoregon.edu"
  ./src/pycoolr/pycoolr-plot/coolr.py src/pycoolr/pycoolr-plot/configs/soscerberus.cfg

  elif [[ $PYCOOLR_PLATFORM = "godzilla.nic.uoregon.edu" ]];
  then
  echo "tool: sos, platform: godzilla.nic.uoregon.edu"
  ./src/pycoolr/pycoolr-plot/coolr.py src/pycoolr/pycoolr-plot/configs/sosgodzilla.cfg

  elif [[ $PYCOOLR_PLATFORM = "delphi.nic.uoregon.edu" ]];
  then
  echo "tool: sos, platform: delphi.nic.uoregon.edu"
  ./src/pycoolr/pycoolr-plot/coolr.py src/pycoolr/pycoolr-plot/configs/sosdelphi.cfg

  fi
fi
