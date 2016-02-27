#!/bin/bash 
if [ $# = 1 ]
then
  echo "Usage: $0 [options] <infile> [-o outfile]"
  echo "generates <base>.pp.<suffix> as output where macros are expanded but header files are not"
  echo "if '-o outfile' is specified it will be ignored"
  exit 1
fi

if [ "${@: -2:1}" == "-o" ] ; then
  filename="${@: -3:1}"
  argc=$(($#-3))
else
  filename="${@: -1:1}"
  argc=$(($#-1))
fi
argv=${@:1:$argc}

#echo "filename: $filename"
#echo "argv: ${argv[@]}"

base=`echo $filename | sed -e 's/\.[^\.]*$//' -e's/.*\///'`
# this transforms /this/file\ name/has/spaces/ver1.0.2/foo.pp.F90 to foo.pp for base
suf=`echo $filename | sed -e 's/.*\./\./' `

shift

# Remove the temporary file so we're appending to an empty file
rm -f $filename.tau.inc

# Put compiler-specific macros in the output file
# Needed for CMake compiler family discovery
if [ -n "$TAU_MAKEFILE" ] ; then
  FULL_CC=`grep ^FULL_CC "$TAU_MAKEFILE" | sed 's/FULL_CC=//'`
  $FULL_CC -E -dM - < /dev/null 2>/dev/null > $filename.tau.inc
  # IBM and oracle compilers use -qshowmacros instead of -dM
  # Pretty much everyone uses -dM
  if [ $? -ne 0 ] ; then
    $FULL_CC -E -qshowmacros - < /dev/null 2>/dev/null > $filename.tau.inc
    if [ $? -ne 0 ] ; then
      echo "WARNING: TAU was unable to prepend compiler-specific macros to the preprocessed files."
    fi
  fi
fi
grep -v __glibcxx_function_requires $filename.tau.inc > $filename.tau.inc~

cpp $filename ${argv[@]} -dM  >> $filename.tau.inc~
grep -v __glibcxx_function_requires $filename.tau.inc~ > $filename.tau.inc

sed -e 's@#include@//TAU_INCLUDE#include@g' $filename > $filename.tau.tmp
cpp $filename.tau.tmp  ${argv[@]} -CC -P -w -include $filename.tau.inc  | sed -e 's@//TAU_INCLUDE#include@#include@g' > $base.pp$suf


# -CC preserves the comments
# -P removes any line number cluter 
# -traditional-cpp preserves whitespaces

/bin/rm -f $filename.tau.tmp $filename.tau.inc

