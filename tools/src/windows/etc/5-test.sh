#!/bin/bash

PATH=/c/Program\ Files/Microsoft\ Visual\ Studio\ 9.0/VC/bin:$PATH
DATE=`date +%Y-%m-%d`

rebuild()
{
    uninstaller='/c/Program Files/Tau cvs/uninstall.exe'
    
    if [ -r "$uninstaller" ] ; then
	echo "Uninstalling old version"
	"$uninstaller" /S
    fi
    rm -rf "/c/Program Files/Tau cvs"
    
    cd /c/tau
    ./build-windows-release.sh cvs
    cd zip
    ./tau-cvs.exe /S
}

runtests()
{
    declare -i numpass
    declare -i numfail
    numpass=0
    numfail=0
    cd /c/Program\ Files/Tau\ cvs/examples/mpi/
    nmake
    if [ -r "mpi.exe" ] ; then
	echo "PASS: MPI test successfully built"
	numpass=$numpass+1
    else
	echo "FAIL: MPI test failed to build"
	numfail=$numfail+1
    fi
    
    cd /c/Program\ Files/Tau\ cvs/examples/threads/
    nmake
    ./profile.exe
    
    if [ -r "profile.0.0.0" ] ; then
	echo "PASS: Threads example successfully ran"
	numpass=$numpass+1
    else
	echo "FAIL: Threads example failed to run"
	numfail=$numfail+1
    fi

    if [ "$numfail" = 0 ] ; then
	echo "PASS: All tests passed!"
    else 
	echo "FAIL $numfail tests failed, $numpass tests passed"
    fi
}

go()
{
rebuild
runtests
}

(time go 2>&1) 2>&1 | tee test.log
