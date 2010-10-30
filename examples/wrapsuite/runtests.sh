#!/bin/bash 

declare -i numpass
declare -i numfail
declare -i numrun

declare testfail
numpass=0;
numfail=0;
numrun=0;

CC=gcc
CXX=g++

if [ `uname -s` = Darwin ] ; then
  extralib=-ldl
  export TAU_OPTIONS="-optCompInst -optShared -optRevert"
# Under Mac OS X, it gets many other malloc events if it doesn't use -optShared
else
  extralib=-lrt
  export TAU_OPTIONS="-optCompInst -optRevert"
fi

cleanup()
{
    /bin/rm -rf simple profile.* *.o
}

check_output()
{
#    pprof
#    cp profile.0.0.0 $i.profile
#    cat profile.0.0.0
    testfail=no

    cat $i.check | while read line; do
#	echo "checking for $line"
	grep -c "$line" profile.0.0.0 &> /dev/null
	if [ $? != 0 ] ; then
	    exit 1
	fi
    done

    testfail=$?

    if [ $testfail = 0 ] ; then
	echo "pass"
	numpass=$numpass+1
    else
	echo "fail"
	numfail=$numfail+1
    fi
    numrun=$numrun+1
    
}


export TAU_METRICS=TIME
unset TAU_VERBOSE


for i in test_*.c ; do
#for i in test_multiple.c ; do
    echo "Testing $i..."

    echo -n "  default...  "
    cleanup
    $CC -o simple $i
    tau_exec -memory -T serial ./simple
    check_output


    echo -n "  link with $extralib -lpthread...  "
    cleanup
    $CC -o simple $i $extralib -lpthread
    tau_exec -memory -T serial ./simple
    check_output


    echo -n "  built with TAU...  "
    cleanup
    tau_cc.sh -g -o simple $i &> /dev/null
    tau_exec -memory -T serial ./simple
    check_output

    echo -n "  built with TAU and linked with $extralib -lpthread...  "
    cleanup
    tau_cc.sh -g -o simple $i $extralib -lpthread  &> /dev/null
    tau_exec -memory -T serial ./simple
    check_output


done


cleanup

echo ""
echo "$numpass passed, $numfail failed, $numrun total"
