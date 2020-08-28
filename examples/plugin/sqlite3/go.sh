#!/bin/bash -e

#make clean
#make -j

rm -f tauprofile.db
export TAU_CALLPATH=1
export TAU_CALLPATH_DEPTH=100

tau_exec -T pthread,serial -sqlite3 ./matmult
tau_exec -T pthread,serial -sqlite3 ./matmult
tau_exec -T pthread,serial -sqlite3 ./matmult

sqlite3 tauprofile.db < check.sql
./parser.py