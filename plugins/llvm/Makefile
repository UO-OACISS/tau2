include ../../include/Makefile

PLUGININSTALLDIR = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/shared$(TAU_CONFIG)/plugins
BUILDDIR = build

all: install
	@echo

install: 
	bash -c "if [ ! -d $(BUILDDIR) ] ; then mkdir $(BUILDDIR) ; fi"
	cd $(BUILDDIR) && CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"  $(TAU_CMAKE) .. -DLLVM_DIR=$(TAU_LLVM_SRC_DIR) -G 'Unix Makefiles' -DCMAKE_INSTALL_PREFIX=$(PLUGININSTALLDIR) -DCMAKE_CXX_COMPILER=$(TAU_LLVM_CXX) -DCMAKE_C_COMPILER=$(TAU_LLVM_CC) -DCMAKE_BUILD_TYPE=Debug && make -j install
	cd ..

clean:
	rm -Rf $(BUILDDIR)

