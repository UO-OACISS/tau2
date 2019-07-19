#!/bin/bash

#Load SOS enviromental variables
source sosd.env.sourceme\


set -x
#Execute tau with sos and use ebs to get code performance data
tau_exec -ebs -ebs_resolution=function -T mpi,pthread,sos,pdt -sos ./matmult

#Wait a bit for the data to be saved to disk
sleep 5

#Stop SOS daemon
sosd_stop

sleep 2

#Check the code related performance data
sqlite3 sosd.00000.db "SELECT frame,value_name,value_guid,value FROM viewCombined WHERE value_name LIKE '%[SAMPLE]%' ORDER by frame, value_name"
