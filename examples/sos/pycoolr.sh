#!/bin/bash

#echo "${TAUROOT}"
TAUROOT=../..
ARCH=x86_64
PLATFORM=cerberus.nic.uoregon.edu
#echo "'${TAUROOT}/${MACHINE}/lib/CubeReader.jar'"
#sosroot=
pkill -f sosd
echo "Start SOS daemons"
cd $TAUROOT
cd $ARCH
cd sos/sos_flow
source hosts/linux/setenv.sh
cd scripts 
./scripts/evp.start.2 &

sleep 2
cd ../inst/bin
demo_app -i 1 -p 100 -m 20000

#source hosts/linux/setenv.sh 
#cp sosd.00000.db /dev/shm

sleep 2
cd ../../..
cd bin
echo "current directory: `pwd`"
echo "Launch PyCOOLR"
./pycoolr -tool=sos -platform=$PLATFORM

