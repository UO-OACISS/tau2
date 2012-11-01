#!/bin/bash

# set JAVA_HOME
# export JAVA_HOME=/home/khuck/jdk1.6.0_32
# export PATH=$PATH:$JAVA_HOME/bin

# check that javac exists
JAVAC=`which javac`
if [ $? -ne "0" ] ; then
  echo "No javac found in path."
  exit
fi

if [ "${JAVAC}" == "" ] ; then
  echo "No javac found in path."
  exit
fi

# check the version
JAVA_FIVE=`java -version 2>&1 | grep version | /usr/bin/awk '{ print $3; }'| sed -e s/\"//g | sed -e s/java\ version//g | sed -e s/1\.[01234]\..*// | wc -c`

# if we have less than 5 characters (i.e. 1.5.0) then quit
if [ "${JAVA_FIVE}" -le "5" ] ; then
  echo "javac version less than 1.5 - TAU tools require 1.5 or newer."
  exit
fi

myarch=`./utils/archfind`
if [ ! -d "./$myarch" ] ; then
# configure WITHOUT prefix, so the java jars are in the right place
# for compiling
  ./configure
fi
cd tools/src/common
make clean ; make
cd ../vis
make clean ; make
cd ../perfdmf
make clean ; make override
cd ../paraprof
make clean ; make override
cd ../perfexplorer
./configure
make clean ; make
cd ../../..
