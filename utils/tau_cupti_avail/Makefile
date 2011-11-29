include ../../include/Makefile

CC     = $(TAU_CC)
CXX		 = $(TAU_CUDA_CXX)


RM      = /bin/rm -f

INSTALLDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin

#CUPTI#INSTALLTAU_CUPTI_AVAIL = $(INSTALLDEST)/tau_cupti_avail #ENDIF#
#CUPTI#TAU_CUPTI_AVAIL = tau_cupti_avail #ENDIF#

all: $(TAU_CUPTI_AVAIL)
install: $(INSTALLTAU_CUPTI_AVAIL)

$(TAU_CUPTI_AVAIL): tau_cupti_avail.o $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/libTAUsh$(TAU_CONFIG).so
		$(CXX) -L$(TAU_CUPTI_INC)/../lib -lcupti -L$(TAU_CUDA_INC)/../lib \
		-L$(TAU_CUDA_INC)/../lib64 -lcuda -lcudart -o $@ $< $(TAU_LIBS)

tau_cupti_avail.o: tau_cupti_avail.cpp
		$(TAU_CUDA_CXX) $(CFLAGS) -I$(TAU_CUDA_INC) -I$(TAU_CUPTI_INC) -I$(TAU_INC_DIR) -c $<

$(INSTALLTAU_CUPTI_AVAIL): $(TAU_CUPTI_AVAIL)
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $(TAU_CUPTI_AVAIL) $(INSTALLDEST)


clean:
	$(RM) tau_cupti_avail.o $(TAU_CUPTI_AVAIL) 

