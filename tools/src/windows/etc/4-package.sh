#!/bin/bash

ROOT=/cygdrive/c/tau

if [ $# -ne 1 ] ; then
    echo "-------------"
    echo "Usage: 4-package.sh <version> (e.g. 2.14.5)"
    echo "-------------"
    exit
fi

cd $ROOT

# rm -rf zip
# mkdir -p zip/tau-$1
# cp -r $ROOT/tau-windows/* $ROOT/zip/tau-$1
# cd $ROOT/zip
# zip -r tau-$1.zip tau-$1

echo "-------------"
echo "Finished"
echo "File is $ROOT\zip\tau-$1.zip"
echo "-------------"

cd $ROOT

PATH="$PATH:/c/Program Files/NSIS"

makensis.exe /DVERSION=$1 /DOUTFILE=C:/tau/zip/tau-$1.exe C:/tau/tau2/tools/src/windows/tau.nsi

echo "Done!"
