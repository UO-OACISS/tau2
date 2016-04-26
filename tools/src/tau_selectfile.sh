#!/bin/bash
selectfile=$1 
filename=$2

# USAGE: 
# <tool> <select_file> <filename> 
#selectfile=select.tau
#filename=foo.f90

echoIfDebug() {
  echo -n ""
}


if [ $# == 0 ]; then
  echo "Usage: $0 <select_file> <filename_to_instrument>"
  exit 1;
fi



includeIt=1
for pattern in `sed -e 's/#.*//g' -e '/BEGIN_FILE_INCLUDE_LIST/,/END_FILE_INCLUDE_LIST/{/BEGIN_FILE_INCLUDE_LIST/{h;d};H;/END_FILE_INCLUDE_LIST/{x;/BEGIN_FILE_INCLUDE_LIST/,/END_FILE_INCLUDE_LIST/p}};d' $selectfile |  sed -e 's/BEGIN_FILE_INCLUDE_LIST//' -e 's/END_FILE_INCLUDE_LIST//' -e 's/#/\.\*/g'  -e 's/"//g' -e 's/^/"/' -e 's/$/"/' | sed -n '1h;2,$H;${g;s/\n/,/g;p}' | sed -e 's/"",//g' -e 's/,""//g' -e 's/,/ /g'  | sed -e 's/"//g' `
do
  echoIfDebug "Filename = $filename in included pattern = $pattern"
  if ([[ "$filename" == $pattern ]]); then
    echoIfDebug "$pattern matches included $filename: OK TO INSTRUMENT $filename"
    includeIt=1
    break;
  else
    includeIt=0
  fi
done

if [ $includeIt == 1 ]; then
  echoIfDebug "It is ok to instrument $filename"
fi 

excludeIt=0

for pattern in `sed -e 's/#.*//g' -e '/BEGIN_FILE_EXCLUDE_LIST/,/END_FILE_EXCLUDE_LIST/{/BEGIN_FILE_EXCLUDE_LIST/{h;d};H;/END_FILE_EXCLUDE_LIST/{x;/BEGIN_FILE_EXCLUDE_LIST/,/END_FILE_EXCLUDE_LIST/p}};d' $selectfile |  sed -e 's/BEGIN_FILE_EXCLUDE_LIST//' -e 's/END_FILE_EXCLUDE_LIST//' -e 's/#/\.\*/g'  -e 's/"//g' -e 's/^/"/' -e 's/$/"/' | sed -n '1h;2,$H;${g;s/\n/,/g;p}' | sed -e 's/"",//g' -e 's/,""//g' -e 's/,/ /g'  | sed -e 's/"//g' `
do
  echoIfDebug "Filename = $filename in excluded pattern = $pattern"
  if ([[ "$filename" == $pattern ]] ); then 
    echoIfDebug "$pattern matches excluded $filename: DO NOT INSTRUMENT $filename"
    excludeIt=1
    break;
  else
    excludeIt=0
  fi 
done

if [ $includeIt == 0 -o $excludeIt == 1 ]; then
  echo "DO NOT instrument $filename"
  exit 0
else
  echo "OK to instrument $filename"
  exit 1
fi



