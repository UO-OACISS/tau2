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

# Source that host's settings file to load modules and set paths
source ${tauroot}/tests/configs/${myhost}.settings

cd ${tauroot}

# Start with a clean build
if [ "${archtype}" == "x86_64" ] ; then
    if [ -d "${tauroot}/${archtype}" ] ; then
        rm -rf ${tauroot}/${archtype}
    fi
fi

myname=`whoami`
mkdir -p /tmp/${myname}
config_log=/tmp/${myname}/config.log
build_log=/tmp/${myname}/build.log
test_log=/tmp/${myname}/test.log

export PROGRAMS="mm"

test_vanilla() {
    # Total vanilla build
    echo "Configuring vanilla (see ${config_log})"
    ${scriptdir}/buildbot.sh configure >& ${config_log}
    echo "Building... (see ${build_log})"
    ${scriptdir}/buildbot.sh build >& ${build_log}
    echo "Testing ${PROGRAMS}... (see ${test_log})"
    ${scriptdir}/buildbot.sh test >& ${test_log}

    # Base support build
    configs="${base_support}"
    echo "Configuring with ${configs}... (see ${config_log})"
    ${scriptdir}/buildbot.sh configure ${configs} >& ${config_log}
    echo "Building... (see ${build_log})"
    ${scriptdir}/buildbot.sh build >& ${build_log}
    echo "Testing ${PROGRAMS}... (see ${test_log})"
    ${scriptdir}/buildbot.sh test >& ${test_log}
}

test_threads() {
    # Test threading options
    for option in "${thread_options[@]}" ; do
        configs="${base_support} ${option}"
        echo "Configuring with ${configs}... (see ${config_log})"
        ${scriptdir}/buildbot.sh configure ${configs} >& ${config_log}
        echo "Building... (see ${build_log})"
        ${scriptdir}/buildbot.sh build >& ${build_log}
        echo "Testing ${PROGRAMS}... (see ${test_log})"
        ${scriptdir}/buildbot.sh test >& ${test_log}
    done
}

test_mpi() {
    # Test mpi options
    for option in "${mpi_options[@]}" ; do
        configs="${base_support} ${option}"
        echo "Configuring with ${configs}... (see ${config_log})"
        ${scriptdir}/buildbot.sh configure ${configs} >& ${config_log}
        echo "Building... (see ${build_log})"
        ${scriptdir}/buildbot.sh build >& ${build_log}
        echo "Testing ${PROGRAMS}... (see ${test_log})"
        ${scriptdir}/buildbot.sh test >& ${test_log}
    done
}

test_papi() {
    export TAU_METRICS=TIME:PAPI_TOT_INS

    # Test papi options
    for option in "${papi_options[@]}" ; do
        configs="${base_support} ${option}"
        echo "Configuring with ${configs}... (see ${config_log})"
        ${scriptdir}/buildbot.sh configure ${configs} >& ${config_log}
        echo "Building... (see ${build_log})"
        ${scriptdir}/buildbot.sh build >& ${build_log}
        echo "Testing ${PROGRAMS}... (see ${test_log})"
        ${scriptdir}/buildbot.sh test >& ${test_log}
    done

    unset TAU_METRICS
}

test_python() {
    export PROGRAMS="python"

    # Test python options
    for option in "${python_options[@]}" ; do
        configs="${base_support} ${option}"
        echo "Configuring with ${configs}... (see ${config_log})"
        ${scriptdir}/buildbot.sh configure ${configs} >& ${config_log}
        echo "Building... (see ${build_log})"
        ${scriptdir}/buildbot.sh build >& ${build_log}
        echo "Testing ${PROGRAMS}... (see ${test_log})"
        ${scriptdir}/buildbot.sh test >& ${test_log}
    done
}

test_cuda() {
    export PROGRAMS="cuda openacc"

    # Test cuda options
    for option in "${cuda_options[@]}" ; do
        configs="${base_support} ${option}"
        echo "Configuring with ${configs}... (see ${config_log})"
        ${scriptdir}/buildbot.sh configure ${configs} >& ${config_log}
        echo "Building... (see ${build_log})"
        ${scriptdir}/buildbot.sh build >& ${build_log}
        echo "Testing ${PROGRAMS}... (see ${test_log})"
        ${scriptdir}/buildbot.sh test >& ${test_log}
    done
}

test_vanilla
test_threads
test_mpi
test_papi
test_python
test_cuda

echo "Success!"
