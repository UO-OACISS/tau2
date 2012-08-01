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
TEST_CONFIG=regression_taudb
TEST_TRIAL_ID=4

# POSTGRESQL_INCLUDES=-I/usr/local/pgsql/include
# POSTGRESQL_LIBRARIES=-L/usr/local/pgsql/lib -lpq
# POSTGRESQL_LIBRARIES=-lpq
# Mist
PQDIR=/home/users/khuck/src/postgresql-8.4.9/src
POSTGRESQL_INCLUDES=-I$(PQDIR)/interfaces/libpq -I$(PQDIR)/include
POSTGRESQL_LIBRARIES=-L$(PQDIR)/interfaces/libpq -lpq -Wl,-rpath=$(PQDIR)/interfaces/libpq

FLAGS=-g
# FLAGS=-ggdb -Waddress -Warray-bounds -Wchar-subscripts -Wimplicit-int -Wimplicit-function-declaration -Wcomment -Wformat -Wmain -Wmissing-braces -Wnonnull -Wparentheses -Wpointer-sign -Wreturn-type -Wsequence-point -Wstrict-aliasing -Wstrict-overflow=1 -Wswitch -Wtrigraphs -Wuninitialized -Wunknown-pragmas -Wunused-function -Wunused-label -Wunused-value -Wvolatile-register-var -Wextra
PERFDMF_SUPPORT=-DTAUDB_PERFDMF
#DEBUG=-DTAUDB_DEBUG -DTAUDB_DEBUG_DEBUG
DEBUG=-DTAUDB_DEBUG

