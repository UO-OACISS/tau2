#!/bin/sh
# This script was written by Felix Wolf [UTK, FZJ]. It extracts the list of 
# those routines that call an MPI routine directly or indirectly and generates
# a selective instrumentation file that can be fed to the tau_instrumentor for
# re-instrumenting the program. USAGE: Just invoke this in the dir where 
# profile.* are present and it'll write the selective instrumentation file to 
# the stdout.

echo "BEGIN_INCLUDE_LIST"
pprof . | \
grep '=>' | \
grep 'MPI' | \
grep '()' | \
sed s/'[ ][0-9]\+[.,:]*[0-9]*[.,:]*[0-9]*'//g | \
sed s/'=>'//g | \
tr ' ' '\n' | \
tr -s '\n' | \
sort | \
uniq | \
grep -v '()'
echo "END_INCLUDE_LIST"



 
