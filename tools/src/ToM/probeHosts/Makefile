# To be set by user. These are standard MPI C/C++ codes.
INSTALL_DIR = ${HOME}/software/bin
CXX = mpiCC
CC = mpicc
INCLUDE = -I/usr/local/packages/openmpi-1.2.6/gnu-4.1.2-tm/include
LIB = -L/usr/local/packages/openmpi-1.2.6/gnu-4.1.2-tm/lib

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