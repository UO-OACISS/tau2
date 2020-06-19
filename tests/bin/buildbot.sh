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
    echo "Usage: $0 compiler [configure|build|test|clean] config"
    echo "Exmple: $0 gcc configure base"
    echo "Exmple: $0 icc build mpi"
    echo "Exmple: $0 pgcc test cuda"
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
    config=$3
    if [ "${config}" == "vanilla" ] ; then
        config=""
    elif [ "${config}" == "base" ] ; then
        config=${base_support}
    elif [ "${config}" == "pthread" ] ; then
        config=${pthread_config}
    elif [ "${config}" == "opari" ] ; then
        config=${opari_config}
    elif [ "${config}" == "ompt" ] ; then
        config=${ompt_config}
    elif [ "${config}" == "mpi" ] ; then
        config=${mpi_config}
    elif [ "${config}" == "papi" ] ; then
        config=${papi_config}
    elif [ "${config}" == "python" ] ; then
        config=${python_config}
    elif [ "${config}" == "cuda" ] ; then
        config=${cuda_config}
    fi
    echo "./configure ${compilers} ${config}"
    ./configure ${compilers} ${config}

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
    config_arch=`${tauroot}/utils/archfind`
    export PPROF_CMD="${tauroot}/${config_arch}/bin/pprof -a"
    config=$3
    if [ "${config}" == "vanilla" ] ; then
        export PROGRAMS=${basic_test_programs}
    elif [ "${config}" == "base" ] ; then
        export PROGRAMS=${basic_test_programs}
    elif [ "${config}" == "pthread" ] ; then
        export PROGRAMS=${basic_test_programs}
    elif [ "${config}" == "opari" ] ; then
        export PROGRAMS=${basic_test_programs}
    elif [ "${config}" == "ompt" ] ; then
        export PROGRAMS=${basic_test_programs}
    elif [ "${config}" == "mpi" ] ; then
        export PROGRAMS=${mpi_test_programs}
        export MPIRUN="${mpirun_command}"
    elif [ "${config}" == "papi" ] ; then
        export PROGRAMS=${basic_test_programs}
        export TAU_METRICS=TIME:PAPI_TOT_INS
        export PPROF_CMD="cd MULTI__PAPI_TOT_INS && ${tauroot}/${config_arch}/bin/pprof -a"
    elif [ "${config}" == "python" ] ; then
        export PROGRAMS=${python_test_programs}
    elif [ "${config}" == "cuda" ] ; then
        export PROGRAMS=${cuda_test_programs}
    fi
    echo "Running tests..."
    cd ${tauroot}/tests/programs
    make clean
    make
    unset PROGRAMS
    unset MPIRUN
    unset TAU_METRICS
    unset PPROF_CMD

# Do clean step
elif [ "$2" == "clean" ] ; then
    echo "Cleaning tests..."
    cd ${tauroot}/tests/programs
    make cleanall
fi

