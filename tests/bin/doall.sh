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
archtype=`arch`

myname=`whoami`
mkdir -p /tmp/${myname}
config_log=/tmp/${myname}/config.log
build_log=/tmp/${myname}/build.log
test_log=/tmp/${myname}/test.log

three_steps() {
    echo "Configuring... ${compiler} ${configs} (see ${config_log})"
    ${scriptdir}/buildbot.sh ${compiler} configure ${configs} >& ${config_log}
    echo "Building... (see ${build_log})"
    ${scriptdir}/buildbot.sh ${compiler} build >& ${build_log}
    echo "Testing ${PROGRAMS}... (see ${test_log})"
    ${scriptdir}/buildbot.sh ${compiler} test >& ${test_log}
}

test_vanilla() {
    export PROGRAMS=${basic_test_programs}
    configs=""
    # Total vanilla build
    three_steps

    # Base support build
    configs="${base_support}"
    three_steps
}

test_threads() {
    export PROGRAMS=${basic_test_programs}

    # Test threading options
    for option in "${thread_options[@]}" ; do
        configs="${base_support} ${option}"
        three_steps
    done
}

test_mpi() {
    export PROGRAMS=${mpi_test_programs}
    export MPIRUN="mpirun -np 2"

    # Test mpi options
    for option in "${mpi_options[@]}" ; do
        configs="${base_support} ${option}"
        three_steps
    done

    unset MPIRUN
}

test_papi() {
    export PROGRAMS=${basic_test_programs}
    export TAU_METRICS=TIME:PAPI_TOT_INS

    # Test papi options
    for option in "${papi_options[@]}" ; do
        configs="${base_support} ${option}"
        three_steps
    done

    unset TAU_METRICS
}

test_python() {
    export PROGRAMS=${python_test_protrams}

    # Test python options
    for option in "${python_options[@]}" ; do
        configs="${base_support} ${option}"
        three_steps
    done
}

test_cuda() {
    export PROGRAMS=${cuda_test_protrams}

    # Test cuda options
    for option in "${cuda_options[@]}" ; do
        configs="${base_support} ${option}"
        three_steps
    done
}

declare -a compilers=("gcc" "pgi" "intel")

for compiler in "${compilers[@]}" ; do
    # Source that host's settings file to load modules and set paths
    source ${tauroot}/tests/configs/${myhost}.${compiler}.settings

    cd ${tauroot}

    # Start with a clean build
    if [ "${archtype}" == "x86_64" ] ; then
        if [ -d "${tauroot}/${archtype}" ] ; then
            rm -rf ${tauroot}/${archtype}
        fi
    fi

    test_vanilla
    test_threads
    test_mpi
    test_papi
    test_python
    test_cuda
done

echo "Success!"
