#!/bin/bash -x
set -e

export TAU_VERBOSE=0

export PROFILEDIR=reader_profiles
./reader &

export PROFILEDIR=writer_profiles
./writer

wait

#export TAU_VERBOSE=1
export PROFILEDIR=reader2_profiles
tau_exec -T serial,pdt -io ./reader2 &

export PROFILEDIR=writer2_profiles
tau_exec -T serial,pdt -io ./writer2

wait

