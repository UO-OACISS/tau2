#!/bin/sh

if [ $# = 0 ] 
then
  echo "This tool generates a selective instrumentation file (called throttle.tau)"
  echo "from a program output that has TAU<id>: Throttle: Disabling ... messages."
  echo "The throttle.tau file can be sent to re-instrument a program using "
  echo "-optTauSelectFile=throttle.tau as an option to tau_compiler.sh/tau_f90.sh, etc."
  echo "Usage: tau_throttle.sh <output_file(s)> "
  exit 1
fi

echo "BEGIN_EXCLUDE_LIST" > throttle.tau
cat $*  | awk '/TAU<[-]*[0-9]*>:\ Throttle:/ {print $4 " " $5 $6 $7 $8 $9;}' | sort | uniq | sed -e s/[.\>A-Z0-9:\ ]*Throttle:// -e s/\\[.*//  >>throttle.tau
echo "END_EXCLUDE_LIST" >> throttle.tau
