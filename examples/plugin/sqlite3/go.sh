#!/bin/bash -e

#make clean
#make -j

tau_exec -v -T pthread,serial -sqlite3 ./matmult
tau_exec -v -T pthread,serial -sqlite3 ./matmult
tau_exec -v -T pthread,serial -sqlite3 ./matmult

sqlite3 tauprofile.db < check.sql