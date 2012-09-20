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
TEST_CONFIG=alcf_working
# TEST_CONFIG=facets
TEST_TRIAL_ID=1
# TEST_TRIAL_ID=34

TAUDB_DATABASE_DEFINE="-D__TAUDB_POSTGRESQL__"
# MacBook Pro
# POSTGRESQL_INCLUDES=$(TAUDB_DATABASE_DEFINE) 
# POSTGRESQL_LIBRARIES=-lpq

# MacBook Air
# PQDIR=/Library/PostgreSQL/9.2
# POSTGRESQL_INCLUDES=$(TAUDB_DATABASE_DEFINE)  -I$(PQDIR)/include
# POSTGRESQL_LIBRARIES=-L$(PQDIR)/lib -lpq

# Paratools03
PQDIR=/usr/local/pgsql
POSTGRESQL_INCLUDES=$(TAUDB_DATABASE_DEFINE) -I$(PQDIR)/include
POSTGRESQL_LIBRARIES=-L$(PQDIR)/lib -lpq

# Mist / Tank
# PQDIR=/home/users/khuck/src/postgresql-8.4.9/src
# POSTGRESQL_INCLUDES=$(TAUDB_DATABASE_DEFINE) -I$(PQDIR)/interfaces/libpq -I$(PQDIR)/include
# POSTGRESQL_LIBRARIES=-L$(PQDIR)/interfaces/libpq -lpq -Wl,-rpath=$(PQDIR)/interfaces/libpq

FLAGS=-g -Wall
# FLAGS=-ggdb -Waddress -Warray-bounds -Wchar-subscripts -Wimplicit-int -Wimplicit-function-declaration -Wcomment -Wformat -Wmain -Wmissing-braces -Wnonnull -Wparentheses -Wpointer-sign -Wreturn-type -Wsequence-point -Wstrict-aliasing -Wstrict-overflow=1 -Wswitch -Wtrigraphs -Wuninitialized -Wunknown-pragmas -Wunused-function -Wunused-label -Wunused-value -Wvolatile-register-var -Wextra
PERFDMF_SUPPORT=-DTAUDB_PERFDMF

# Debugging levels
#DEBUG=
DEBUG=-DTAUDB_DEBUG -DHASH_DEBUG=1
#DEBUG=-DTAUDB_DEBUG -DTAUDB_DEBUG_DEBUG -DHASH_DEBUG=1
