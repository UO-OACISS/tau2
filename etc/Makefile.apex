include apex.mk

build/CMakeCache.txt:
	cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=$(CC) -DCMAKE_CXX_COMPILER=$(CXX) -DAPEX_BUILD_TESTS=FALSE -DAPEX_BUILD_EXAMPLES=FALSE -DAPEX_WITH_KOKKOS=TRUE $(BFDOPT) -DCMAKE_INSTALL_PREFIX=$(PREFIX)/$(TAUARCH) $(TAUOMPT) $(TAUOPENCL) $(TAUCFLAGS) $(TAUCXXFLAGS) $(TAUCUDA) $(TAUHIP) $(TAUSYCL)

all : build/CMakeCache.txt
	$(MAKE) -C build all

install: all
	$(MAKE) -C build install

clean:
	rm -rf build

