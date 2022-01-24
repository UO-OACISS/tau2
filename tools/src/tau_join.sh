#!/bin/bash
input="tauprofile.xml"
key="tau_anonymized_key.xml"
output="original.xml"
declare -i lineno=0

usage() {
  echo "$0 is a tool to join a tauprofile.xml file generated using TAU_ANONYMIZE=1"
  echo "with the key file (mapping event names) to generate an output file that" 
  echo "may be viewed with original function and event names (decrypted) in paraprof."
  echo " "
  echo "Usage: $0 <tauprofile.xml> [<tau_anonymized_key.xml> -o <original.xml>]"
  exit
}

# Print help message
if [ $# = 0  -o "x$1" == "x--help" ] ; then
  usage
  exit
fi

# If only one arg is specified, use defaults
if [ $# = 1 ]; then
  input=$1; 
  echo "$0 $input $key -o $output"
fi

# If two args are specified, write to original.xml 
if [ $# = 2 ]; then 
  input=$1; 
  key=$2;
  echo "$0 $input $key -o $output"
fi

# If all 4 args are specified, then use them
if [ $# = 4 -a "x$3" == "x-o" ]; then
  input=$1; 
  key=$2;
  output=$4; 
fi

# Check if files are readable
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

# First copy over the key file to output
cp $key $output

# Read one line at a time until you reach end of header
while IFS= read -r line
do
  lineno=lineno+1;
  if [ "x$line" = "x</profile_xml>" ]; then
    break;
  fi
done < "$input"

# Start from the next line and write to output.
lineno=$lineno+1
tail -n +${lineno} ${input} >> ${output}
