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
#TEST_CONFIG=sqlite
TEST_CONFIG=localpsql
#TEST_CONFIG=regression_taudb
#TEST_CONFIG=alcf_working
#TEST_CONFIG=facets
TEST_TRIAL_ID=1 # alcf_working
#TEST_TRIAL_ID=25 # ccsm
#TEST_TRIAL_ID=5 # regression_taudb

################# POSTGRESQL ####################
TAUDB_DATABASE_DEFINE="-D__TAUDB_POSTGRESQL__"

# MacBook Air
PQDIR=/Library/PostgreSQL/9.2
# Paratools03
# PQDIR=/usr/local/pgsql
# Mist / Tank
# PQDIR=/home/users/khuck/src/postgresql-8.4.9/src

DB_INCLUDES=$(TAUDB_DATABASE_DEFINE) -I$(PQDIR)/include
DB_LIBRARIES=-L$(PQDIR)/lib -lpq
# Mist / Tank
# DB_INCLUDES=$(TAUDB_DATABASE_DEFINE) -I$(PQDIR)/interfaces/libpq -I$(PQDIR)/include
# DB_LIBRARIES=-L$(PQDIR)/interfaces/libpq -lpq -Wl,-rpath=$(PQDIR)/interfaces/libpq -lz

################# SQLITE ########################
# TAUDB_DATABASE_DEFINE="-D__TAUDB_SQLITE__"
# SQLDIR=/Users/khuck/install/sqlite
# DB_INCLUDES=$(TAUDB_DATABASE_DEFINE) -I$(SQLDIR)/include
# DB_LIBRARIES=-L$(SQLDIR)/lib -lsqlite3
#################################################

XML_INCLUDES=-I/usr/include/libxml2
XML_LIBRARIES=-L/opt/local/lib -lxml2
ZLIB_INCLUDES=
ZLIB_LIBRARIES=-lz

FLAGS=-g -Wall

# Debugging levels
#DEBUG=
#DEBUG=-DTAUDB_DEBUG -DHASH_DEBUG=1
DEBUG=-DTAUDB_DEBUG -DTAUDB_DEBUG_DEBUG -DHASH_DEBUG=1

INCLUDE=$(XML_INCLUDES) $(ZLIB_INCLUDES) $(DB_INCLUDES)
LIBS=$(XML_LIBRARIES) $(ZLIB_LIBRARIES) $(DB_LIBRARIES)
