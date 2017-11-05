#!/bin/bash -x

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
echo `pwd`
source hosts/linux/setenv.sh
cd scripts 
rm -rf sosd.0000*
./evp.start.2 &

sleep 2
cd ../inst/bin
export SOS_BIN_DIR=`pwd`
demo_app_silent -i 1 -p 100 -m 20000

#source hosts/linux/setenv.sh 
#cp sosd.00000.db /dev/shm

sleep 2
cd ../../../..
echo "current directory: `pwd`"
cd bin
echo "current directory: `pwd`"
echo "Launch PyCOOLR"
./pycoolr -tool=sos -platform=$PLATFORM

