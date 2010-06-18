# Set by TAU configure
TAU_MPI_INC=
TAU_MPILIB_DIR=

# To be set by user. These are standard MPI C/C++ codes.
INSTALL_DIR = ${LUSTRE_HOME}/software/bin
CXX = CC
CC = cc

# MAY need to be set by user. We take whatever TAU defines for locating
#   MPI, but if none is found, the user may have to set them.
INCLUDE = ${TAU_MPI_INC}
LIB = -L${TAU_MPILIB_DIR} -lmpich

# Development definitions for ToM support software
BINARIES = probe probeDiff

all: ${BINARIES}

probe:	probe.cpp
	${CXX} -o probe probe.cpp ${INCLUDE} ${LIB}

probeDiff: probeDiff.cpp
	${CXX} -o probeDiff probeDiff.cpp ${INCLUDE} ${LIB}

install: ${BINARIES}
	cp probe ${INSTALL_DIR}
	cp probeDiff ${INSTALL_DIR}

clean:	uninstall
	rm -f ${BINARIES} *.o *~

uninstall:
	rm -f ${INSTALL_DIR}/probe ${INSTALL_DIR}/probeDiff