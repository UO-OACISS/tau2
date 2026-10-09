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



# Strip leading whitespace, comment lines (# at the start of a line) and blank lines.
tau_si_clean() { sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^#/d' -e '/^$/d' "$1"; }

# Patterns are globs ('*' and '?'); a pattern without a '/' is matched against the basename.
# Do not let the shell expand patterns against the current directory.
set -f
matchFile() {
  local f=$1 pat=$2
  case $pat in
    */*) ;;
    *) f=${f##*/} ;;
  esac
  [[ "$f" == $pat ]]
}

includeIt=1
for pattern in `tau_si_clean $selectfile | sed -e '/BEGIN_FILE_INCLUDE_LIST/,/END_FILE_INCLUDE_LIST/{/BEGIN_FILE_INCLUDE_LIST/{h;d};H;/END_FILE_INCLUDE_LIST/{x;/BEGIN_FILE_INCLUDE_LIST/,/END_FILE_INCLUDE_LIST/p}};d' |  sed -e 's/BEGIN_FILE_INCLUDE_LIST//' -e 's/END_FILE_INCLUDE_LIST//'  -e 's/"//g' -e 's/^/"/' -e 's/$/"/' | sed -n '1h;2,$H;${g;s/\n/,/g;p}' | sed -e 's/"",//g' -e 's/,""//g' -e 's/,/ /g'  | sed -e 's/"//g' `
do
  echoIfDebug "Filename = $filename in included pattern = $pattern"
  if matchFile "$filename" "$pattern"; then
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

for pattern in `tau_si_clean $selectfile | sed -e '/BEGIN_FILE_EXCLUDE_LIST/,/END_FILE_EXCLUDE_LIST/{/BEGIN_FILE_EXCLUDE_LIST/{h;d};H;/END_FILE_EXCLUDE_LIST/{x;/BEGIN_FILE_EXCLUDE_LIST/,/END_FILE_EXCLUDE_LIST/p}};d' |  sed -e 's/BEGIN_FILE_EXCLUDE_LIST//' -e 's/END_FILE_EXCLUDE_LIST//'  -e 's/"//g' -e 's/^/"/' -e 's/$/"/' | sed -n '1h;2,$H;${g;s/\n/,/g;p}' | sed -e 's/"",//g' -e 's/,""//g' -e 's/,/ /g'  | sed -e 's/"//g' `
do
  echoIfDebug "Filename = $filename in excluded pattern = $pattern"
  if matchFile "$filename" "$pattern"; then 
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



