include ../../include/Makefile

CC     = $(TAU_CC)
CXX		 = $(TAU_CXX)


RM      = /bin/rm -f

INSTALLDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin

#CUPTI#INSTALLTAU_CUPTI_AVAIL = $(INSTALLDEST)/tau_cupti_avail #ENDIF#
#CUPTI#TAU_CUPTI_AVAIL = tau_cupti_avail #ENDIF#

all: $(TAU_CUPTI_AVAIL)
install: $(INSTALLTAU_CUPTI_AVAIL)

$(TAU_CUPTI_AVAIL): tau_cupti_avail.o
		$(CXX) -L$(TAU_CUPTI_INC)/../lib -lcupti -o $@ $< $(TAU_LIBS)

tau_cupti_avail.o: tau_cupti_avail.cpp
		$(CXX) $(CFLAGS) -I$(TAU_CUDA_INC) -I$(TAU_CUPTI_INC) -I$(TAU_INC_DIR) -c $<

$(INSTALLTAU_CUPTI_AVAIL): $(TAU_CUPTI_AVAIL)
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $(TAU_CUPTI_AVAIL) $(INSTALLDEST)


clean:
	$(RM) tau_cupti_avail.o $(TAU_CUPTI_AVAIL)

