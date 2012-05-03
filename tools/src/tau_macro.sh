#!/bin/sh
if [ $# = 1 ]
then
  echo "Usage: $0 <filename> -D<variables> "
  echo "generates taucpp.<filename> as output where macros are expanded but header files are not"
  exit 1
fi

filename=$1
shift
cpp $filename $* -dM > $filename.tau.inc
sed -e 's@#include@//TAU_INCLUDE#include@g' $filename > $filename.tau.tmp
#echo "cpp $filename.tau.tmp  $* -P -traditional-cpp -w -include $filename.tau.inc"
cpp $filename.tau.tmp  $* -P -traditional-cpp -w -include $filename.tau.inc  | sed -e 's@//TAU_INCLUDE#include@#include@g' > taucpp.$filename

/bin/rm -f $filename.tau.tmp $filename.tau.inc



