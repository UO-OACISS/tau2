#!/bin/bash -e

# where is this script?
if [ -z ${scriptdir} ] ; then
    scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

# Set the TAU root directory (relative to where this script lives)
tauroot="$( dirname "$( dirname "${scriptdir}" )" )"
export tauroot

# Get the hostname
myhost=`hostname`
# In case there are extra qualifiers on the hostname
myhost=`basename -s .nic.uoregon.edu ${myhost}`
osname=`uname`

# Source that host's settings file to load modules and set paths
source ${tauroot}/tests/configs/${myhost}.settings

cd ${tauroot}

# Usage message
if [ $# -eq 0 ] ; then
    echo "Usage: $0 [configure|build|test|clean]"
    kill -INT $$
fi

# Do configure step, using all arguments except this script name and configure
if [ "$1" == "configure" ] ; then
    echo "./configure ${@:2} ..."
    ./configure ${@:2}

# Do build step
elif [ "$1" == "build" ] ; then
    nprocs=2
    if [ ${osname} == "Darwin" ]; then
        nprocs=`sysctl -n hw.ncpu`
        ntcores=`sysctl -n hw.ncpu`
    else
        nprocs=`nproc --all`
        # Get the true number of total cores, not threads.
        ncores=`lscpu | grep -E '^Core' | awk '{print $NF}'`
        nsockets=`lscpu | grep -E '^Socket' | awk '{print $NF}'`
        let ntcores=$ncores*$nsockets
    fi

    echo "building with ${ntcores} cores..."
    make -j $ntcores -l $ntcores install

# Do test step
elif [ "$1" == "test" ] ; then
    echo "Running tests..."
    cd ${tauroot}/tests/programs
    make

# Do clean step
elif [ "$1" == "clean" ] ; then
    echo "Cleaning tests..."
    cd ${tauroot}/tests/programs
    make cleanall
fi

