#****************************************************************************
#*                      TAU Performance System (R)                          *
#*                      http://tau.uoregon.edu                              *
#****************************************************************************
#*    Copyright 2012                                                       **
#*    Department of Computer and Information Science, University of Oregon **
#*    Juelich Supercomputing Center, NIC, Germany                          ** 
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
# Use gmake to build matmult

.SUFFIXES : .c
.SILENT :

CC=gcc
AR=ar
#CC=tau_cc.sh -tau_makefile=${HOME}/src/tau2/x86_64/lib/Makefile.tau-pdt -optQuiet -optPreProcess

# POSTGRESQL_INCLUDES=-I/usr/local/pgsql/include
# POSTGRESQL_LIBRARIES=-L/usr/local/pgsql/lib -lpq
POSTGRESQL_LIBRARIES=-lpq

FLAGS=-g
PERFDMF_SUPPORT=-DTAUDB_PERFDMF
#DEBUG=-DTAUDB_DEBUG -DTAUDB_DEBUG_DEBUG
DEBUG=-DTAUDB_DEBUG

