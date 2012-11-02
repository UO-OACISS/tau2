#!/bin/bash
procs=1
if [ $# -gt 0 ] 
then
  procs=$1
fi

aprun -n $procs hostname > topolist.txt

sort topolist.txt  -uo topolist.txt

while read p
do 
if [[ $p == nid* ]]
then
  collect=$collect,$p; 
fi
done < topolist.txt 
topolcoords -n $collect > topolist.txt
#tail -n+2 topolist.txt > topolist.txt
