include ../../include/Makefile
include ../include/Makefile

CC     = $(TAU_CC)
CXX		 = $(TAU_CXX)
CXX		 = $(TAU_CXX)


RM      = /bin/rm -f

INSTALLDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin

all: $(TAU_L0_AVAIL)
install: $(INSTALLTAU_L0_AVAIL)

$(TAU_L0_AVAIL): tau_l0_avail.o $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/libTAUsh$(TAU_CONFIG).so
		$(CXX) -o $@ $< $(TAU_LEVEL_ZERO_LINKER_OPTS) $(CXXFLAGS) -I$(TAUROOT)/include

tau_l0_avail.o: tau_l0_avail.cpp
		$(CXX) $(CFLAGS)-c $< -I$(TAUROOT)/include $(CXXFLAGS)

$(INSTALLTAU_L0_AVAIL): $(TAU_L0_AVAIL)
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $(TAU_L0_AVAIL) $(INSTALLDEST)


clean:
	$(RM) tau_l0_avail.o $(TAU_L0_AVAIL)