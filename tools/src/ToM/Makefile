# To be set by TAU configure
TAUROOT=
MRNET_ROOT=
MRNET_INCLUDE=
MRNET_LIBS=

# To be set by users. CXX has to be set separately from the defaults in TAU
#   because on some platforms like craycnl, the MRNet filters have to be
#   built using g++ instead of CC. Consequently, because CXX is divorced
#   from TAU's, CXXFLAGS cannot also make use of TAU's defintions.
INSTALL_ROOT = ${HOME}/software
CXX = g++
CXXFLAGS = -O2 -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -Dcompiler_gnu -fno-default-inline $(INCLUDE)

# NOT TO BE MODIFIED. These definitions depend strictly on the above
#   TAU settings, no more, no less.
INCLUDE = ${MRNET_INCLUDE} -I${TAUROOT}/include
LD_FLAGS = ${MRNET_LIBS} -lpthread -Wl,-E 
SOFLAGS = -fPIC -shared -rdynamic

# Development definitions for ToM to build correctly
BINARIES = ToM_FE
TOM_FILTERS = ToM_Sync_Filter.so ToM_Histogram_Filter.so
EXPR_FILTERS = ToM_Test_Filter.so ToM_StatsSingle_Filter.so
FILTERS = ${TOM_FILTERS} ${EXPR_FILTERS}

all: $(BINARIES) $(FILTERS)

ToM_FE:	ToM_FE.o
	$(CXX) -o ToM_FE ToM_FE.o $(LD_FLAGS)

ToM_FE.o:	ToM_FE.cpp
	$(CXX) $(CXXFLAGS) -c ToM_FE.cpp

ToM_Test_Filter.so:	ToM_Test_Filter.cpp
	$(CXX) $(CXXFLAGS) $(SOFLAGS) -o $@ ToM_Test_Filter.cpp ${INCLUDE}

ToM_StatsSingle_Filter.so:	ToM_StatsSingle_Filter.cpp
	$(CXX) $(CXXFLAGS) $(SOFLAGS) -o $@ ToM_StatsSingle_Filter.cpp ${INCLUDE}

ToM_Stats_Filter.so:	ToM_Stats_Filter.cpp
	$(CXX) $(CXXFLAGS) $(SOFLAGS) -o $@ ToM_Stats_Filter.cpp ${INCLUDE}

ToM_Sync_Filter.so:	ToM_Sync_Filter.cpp
	$(CXX) $(CXXFLAGS) $(SOFLAGS) -o $@ ToM_Sync_Filter.cpp ${INCLUDE}

ToM_Histogram_Filter.so:	ToM_Histogram_Filter.cpp
	$(CXX) $(CXXFLAGS) $(SOFLAGS) -o $@ ToM_Histogram_Filter.cpp ${INCLUDE}

install: ${BINARIES} ${FILTERS}
	cp ${BINARIES} ${INSTALL_ROOT}/bin/
	cp ${FILTERS} ${INSTALL_ROOT}/lib/

clean: uninstall
	rm -f ${BINARIES} ${FILTERS} *.o *~

uninstall:
	pushd .; cd ${INSTALL_ROOT}/bin; rm -f ${BINARIES}; cd ${INSTALL_ROOT}/lib; rm -f ${FILTERS}; popd

