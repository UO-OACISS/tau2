#!/bin/bash
input="tauprofile.xml"
key="tau_anonymized_key.xml"
output="original.xml"

usage() {
  echo "$0 is a tool to join a tauprofile.xml file generated using TAU_ANONYMIZE=1"
  echo "with the key file (mapping event names) to generate an output file that" 
  echo "may be viewed with original function and event names (decrypted) in paraprof."
  echo " "
  echo "Usage: $0 <tauprofile.xml> [<tau_anonymized_key.xml> -o <original.xml>]"
  exit
}

if [ $# = 0  -o "x$1" == "x--help" ] ; then
  usage
  exit
fi

if [ $# = 1 ]; then
  input=$1; 
  echo "$0 $input $key -o $output"
fi

if [ $# = 2 ]; then 
  input=$1; 
  key=$2;
  echo "$0 $input $key -o $output"
fi

if [ $# = 4 -a "x$3" == "x-o" ]; then
  input=$1; 
  key=$2;
  output=$4; 
fi

if [ ! -r $input ]; then
  echo "$0: ERROR: $input not readable."
  usage
  exit
fi

if [ ! -r $key ]; then
  echo "$0: ERROR: $key not readable."
  usage
  exit
fi

cp $key $output
do_echo=0
while IFS= read -r line
do
  if [ $do_echo == 1 ]; then
    echo "$line" >> $output
  fi 
  if [ "x$line" = "x</profile_xml>" ]; then
    do_echo=1
  fi
done < "$input"
