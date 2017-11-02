#!/bin/bash

#echo "${TAUROOT}"
TAUROOT=../..
ARCH=x86_64
PLATFORM=cerberus.nic.uoregon.edu
#echo "'${TAUROOT}/${MACHINE}/lib/CubeReader.jar'"
#sosroot=
pkill -f sosd
echo "Generate evpath database"
cd $TAUROOT
cd $ARCH
cd sos/sos_flow
source hosts/linux/setenv.sh 
cd scripts
./evp.start.2 &
cp sosd.00000.db /dev/shm

sleep 5
cd ../../..
cd bin
echo "current directory: `pwd`"
echo "Launch PyCOOLR"
./pycoolr -tool=sos -platform=$PLATFORM

