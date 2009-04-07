#!/bin/bash

if [ $# -ne 1 ] ; then
    echo "-------------"
    echo "Usage: build-windows-release.sh <version> (e.g. 2.14.5)"
    echo "-------------"
    exit
fi

./1-get.sh
./2-build.sh
./3-install.sh
./4-package.sh

