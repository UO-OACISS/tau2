include ../../include/Makefile
include ../include/Makefile

CC     = $(TAU_CC)
CXX		 = $(TAU_CXX)
CXX		 = $(TAU_CXX)


RM      = /bin/rm -f

INSTALLDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin

all: $(TAU_ROCM_AVAIL)
install: $(INSTALLTAU_ROCM_AVAIL)

$(TAU_ROCM_AVAIL): tau_rocm_avail.o $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/libTAUsh$(TAU_CONFIG).so
		$(CXX) -o $@ $< $(TAU_ROCPROFILER_LINKER_OPTS)  $(ROCPROFILER_FLAGS) $(CXXFLAGS) -I$(TAUROOT)/include

tau_rocm_avail.o: tau_rocm_avail.cpp
		$(CXX) $(CFLAGS)-c $< -I$(TAUROOT)/include $(CXXFLAGS) $(ROCPROFILER_FLAGS)

$(INSTALLTAU_ROCM_AVAIL): $(TAU_ROCM_AVAIL)
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $(TAU_ROCM_AVAIL) $(INSTALLDEST)


clean:
	$(RM) tau_ROCM_avail.o $(TAU_ROCM_AVAIL)