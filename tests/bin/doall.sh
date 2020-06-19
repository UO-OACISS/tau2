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
    cmd="${scriptdir}/buildbot.sh ${compiler} configure ${config}"
    echo "Configuring... ${compiler} ${config} (see ${config_log})"
    echo ${cmd}
    ${cmd} >& ${config_log}
    echo "Building... (see ${build_log})"
    cmd="${scriptdir}/buildbot.sh ${compiler} build ${config}"
    echo ${cmd}
    ${cmd} >& ${build_log}
    echo "Testing ${PROGRAMS}... (see ${test_log})"
    cmd="${scriptdir}/buildbot.sh ${compiler} test ${config}"
    echo ${cmd}
    ${cmd} >& ${test_log}
}

test_vanilla() {
    config="vanilla"
    # Total vanilla build
    three_steps

    # Base support build
    config="base"
    three_steps
}

test_threads() {
    # Test threading options
    if [ "${pthread_config}" != "" ] ; then
        config="pthread"
        three_steps
    fi
    if [ "${opari_config}" != "" ] ; then
        config="opari"
        three_steps
    fi
    if [ "${ompt_config}" != "" ] ; then
        config="ompt"
        three_steps
    fi
}

test_mpi() {
    # Test mpi options
    if [ "${mpi_config}" != "" ] ; then
        config="mpi"
        three_steps
    fi
}

test_papi() {
    # Test papi options
    if [ "${papi_config}" != "" ] ; then
        config="papi"
        three_steps
    fi
}

test_python() {
    # Test python options
    if [ "${python_config}" != "" ] ; then
        config="python"
        three_steps
    fi
}

test_cuda() {
    # Test cuda options
    if [ "${cuda_config}" != "" ] ; then
        config="cuda"
        three_steps
    fi
}

#declare -a compilers=("gcc" "pgi" "intel" "xlc")
declare -a compilers=("xlc")

for compiler in "${compilers[@]}" ; do
    if [ ! -f ${tauroot}/tests/configs/${myhost}.${compiler}.settings ] ; then
        continue
    fi
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
