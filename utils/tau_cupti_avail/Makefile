include ../../include/Makefile
include ../Makefile

CC     = $(TAU_CC)
CXX		 = $(TAU_CXX)


RM      = /bin/rm -f

INSTALLDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin

all: $(TAU_CUPTI_AVAIL)
install: $(INSTALLTAU_CUPTI_AVAIL)

$(TAU_CUPTI_AVAIL): tau_cupti_avail.o $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/libTAUsh$(TAU_CONFIG).so
		$(CXX) -L$(TAU_CUPTI_INC)/../$(TAU_LIBEX) -lcupti -L$(TAU_CUDA_INC)/../$(TAU_LIBEX) \
		-lcuda -lcudart \
		$(TAU_CUDA_LIBRARY) -o $@ $< -L$(TAU_LIB_DIR) -lTAUsh$(TAU_CONFIG) $(TAU_LDFLAGS)

tau_cupti_avail.o: tau_cupti_avail.cpp
		$(TAU_CUDA_CXX) $(CFLAGS) -I$(TAU_CUDA_INC) -I$(TAU_CUPTI_INC) -I$(TAU_INC_DIR) -c $<

$(INSTALLTAU_CUPTI_AVAIL): $(TAU_CUPTI_AVAIL)
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $(TAU_CUPTI_AVAIL) $(INSTALLDEST)


clean:
	$(RM) tau_cupti_avail.o $(TAU_CUPTI_AVAIL) 

