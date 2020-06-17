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

# Usage message
if [ $# -lt 2 ] ; then
    echo "Usage: $0 compiler [configure|build|test|clean]"
    echo "Exmple: $0 gcc configure"
    echo "Exmple: $0 icc build"
    echo "Exmple: $0 pgcc test"
    kill -INT $$
fi

compiler=$1

if [ -f ${tauroot}/tests/configs/${myhost}.${compiler}.settings ] ; then
    # Source that host's settings file to load modules and set paths
    source ${tauroot}/tests/configs/${myhost}.${compiler}.settings
else
    echo "${tauroot}/tests/configs/${myhost}.${compiler}.settings not found."
    kill -INT $$
fi

cd ${tauroot}

compilers="-cc=${CC} -c++=${CXX} -fortran=${FC}"

# Do configure step, using all arguments except this script name and configure
if [ "$2" == "configure" ] ; then
    echo "./configure ${compilers} ${@:3}"
    ./configure ${compilers} ${@:3}

# Do build step
elif [ "$2" == "build" ] ; then
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
elif [ "$2" == "test" ] ; then
    echo "Running tests..."
    cd ${tauroot}/tests/programs
    make clean
    make

# Do clean step
elif [ "$2" == "clean" ] ; then
    echo "Cleaning tests..."
    cd ${tauroot}/tests/programs
    make cleanall
fi

