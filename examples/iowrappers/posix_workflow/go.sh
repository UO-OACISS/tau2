#!/bin/bash -x
set -e

make

export TAU_VERBOSE=0

export PROFILEDIR=reader_profiles
./reader &

export PROFILEDIR=writer_profiles
./writer

wait
./tau_prof2json.py -o linked.json -w workflow_in.json writer_profiles reader_profiles

#export TAU_VERBOSE=1
export PROFILEDIR=reader2_profiles
tau_exec -T serial,pdt -io ./reader2 &

export PROFILEDIR=writer2_profiles
tau_exec -T serial,pdt -io ./writer2

wait
tau_prof2json.py -o preload.json writer2_profiles reader2_profiles

