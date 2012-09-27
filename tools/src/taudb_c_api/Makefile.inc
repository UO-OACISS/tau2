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
#CC=tau_cc.sh -tau_makefile=${HOME}/src/tau2-taudb-branch/apple/lib/Makefile.tau-pdt -optQuiet
#TEST_CONFIG=ccsm
TEST_CONFIG=regression_taudb
#TEST_CONFIG=alcf_working
# TEST_CONFIG=facets
# TEST_TRIAL_ID=1 # alcf_working
TEST_TRIAL_ID=25 # ccsm
TEST_TRIAL_ID=5 # regression_taudb

TAUDB_DATABASE_DEFINE="-D__TAUDB_POSTGRESQL__"
# MacBook Pro
# POSTGRESQL_INCLUDES=$(TAUDB_DATABASE_DEFINE) 
# POSTGRESQL_LIBRARIES=-lpq -lz

# MacBook Air
PQDIR=/Library/PostgreSQL/9.2
POSTGRESQL_INCLUDES=$(TAUDB_DATABASE_DEFINE)  -I$(PQDIR)/include -I/usr/include/libxml2
POSTGRESQL_LIBRARIES=-L/usr/lib -L$(PQDIR)/lib -lpq -lz -lxml2

# Paratools03
# PQDIR=/usr/local/pgsql
# POSTGRESQL_INCLUDES=$(TAUDB_DATABASE_DEFINE) -I$(PQDIR)/include
# POSTGRESQL_LIBRARIES=-L$(PQDIR)/lib -lpq -lz

# Mist / Tank
# PQDIR=/home/users/khuck/src/postgresql-8.4.9/src
# POSTGRESQL_INCLUDES=$(TAUDB_DATABASE_DEFINE) -I$(PQDIR)/interfaces/libpq -I$(PQDIR)/include
# POSTGRESQL_LIBRARIES=-L$(PQDIR)/interfaces/libpq -lpq -Wl,-rpath=$(PQDIR)/interfaces/libpq -lz

FLAGS=-g -Wall
PERFDMF_SUPPORT=-DTAUDB_PERFDMF

# Debugging levels
DEBUG=
#DEBUG=-DTAUDB_DEBUG -DHASH_DEBUG=1
#DEBUG=-DTAUDB_DEBUG -DTAUDB_DEBUG_DEBUG -DHASH_DEBUG=1
