#!/bin/bash

# don't want the user to see glibc errors (on Franklin)
export MALLOC_CHECK_=0

# check that we're on linux
if [ $(uname) != "Linux" ]; then
    echo "failed"
    exit
fi

# first, check that this is sun java
ver=`java -version 2>&1 | tail -1 | awk '{print $1 $2}'`

if [ "x$ver" != "xJavaHotSpot(TM)" ] ; then
    echo "failed"
    exit
fi

memtotal=`cat /proc/meminfo | head -1 | awk '{print $2}'`

if [ $(uname -m) == "i686" ]; then
    if [ $memtotal -gt 2300 ] ; then
	memtotal=2300
    fi
fi


trymem=$(($memtotal/1000*7/8))
while [ $trymem -gt 250 ] ; do

    check=`java -Xmx${trymem}m foobar 2>&1 | head -2 | tail -1`
    if [ "x$check" != "xCould not reserve enough space for object heap" ] ; then
	if [ "x$check" != "xThe specified size exceeds the maximum representable size." ] ; then
	    echo "$trymem"
	    exit
	fi
    fi

    trymem=$(($trymem*3/4))
done

echo "failed"
