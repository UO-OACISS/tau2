#!/bin/sh
if [ $# = 1 ]
then
  echo "Usage: $0 <filename> -D<variables> "
  echo "generates <base>.pp.<suffix> as output where macros are expanded but header files are not"
  exit 1
fi

filename=$1
base=`echo $filename | sed -e 's/\.[^\.]*$//' -e's/.*\///'`
# this transforms /this/file\ name/has/spaces/ver1.0.2/foo.pp.F90 to foo.pp for base
suf=`echo $filename | sed -e 's/.*\./\./' `

shift
cpp $filename $* -dM  > $filename.tau.inc
sed -e 's@#include@//TAU_INCLUDE#include@g' $filename > $filename.tau.tmp
#echo "cpp $filename.tau.tmp  $* -P -traditional-cpp -w -include $filename.tau.inc"
# omit -traditional-cpp for C/C++
cpp $filename.tau.tmp  $* -CC -P -w -include $filename.tau.inc  | sed -e 's@//TAU_INCLUDE#include@#include@g' > $base.pp$suf


# -CC preserves the comments
# -P removes any line number cluter 
# -traditional-cpp preserves whitespaces

#/bin/rm -f $filename.tau.tmp $filename.tau.inc



