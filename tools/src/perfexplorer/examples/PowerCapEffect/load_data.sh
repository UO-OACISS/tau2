#!/bin/bash -e

tar -xzf profile_data.tgz
taudb_configure --create-default -c PowerExample
cd profile_data

for app in am8 co8 lu8 ; do
  if [ ${app} = "am8" ] ; then
    appname="amg2013"
  elif [ ${app} = "co8" ] ; then
    appname="CoMD-mpi"
  else
    appname="lulesh"
  fi
  for cap in {50..115} ; do 
    dir="cap${cap}_wrapped_${app}"
    if [ -d ${dir} ] ; then
	  cmd="taudb_loadtrial -a $appname -x $cap -n $dir -c PowerExample $dir"
	  echo $cmd
	  $cmd
	fi
  done
done

cd ..
rm -rf profile_data
