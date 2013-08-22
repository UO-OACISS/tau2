include ../../include/Makefile
include ../include/Makefile

CC     = $(TAU_CC)
CXX		 = $(TAU_CXX)


RM      = /bin/rm -f

INSTALLDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin

all: $(TAU_CUPTI_AVAIL)
install: $(INSTALLTAU_CUPTI_AVAIL)

<<<<<<< HEAD
$(TAU_CUPTI_AVAIL): tau_cupti_avail.o $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/libtau$(TAU_CONFIG).a
		$(CXX) -dynamic -L$(TAU_CUPTI_INC)/../lib -lcupti -L$(TAU_CUDA_INC)/../lib \
		-L$(TAU_CUDA_INC)/../lib64 -lcuda -lcudart \
		$(TAU_CUDA_LIBRARY) -o $@ $< $(TAU_LINK)
=======
$(TAU_CUPTI_AVAIL): tau_cupti_avail.o $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/libTAUsh$(TAU_CONFIG).so
		$(CXX) -L$(TAU_CUPTI_INC)/../$(TAU_LIBEX) -lcupti -L$(TAU_CUDA_INC)/../$(TAU_LIBEX) \
		-lcuda -lcudart \
		$(TAU_CUDA_LIBRARY) -o $@ $< -L$(TAU_LIB_DIR) -lTAUsh$(TAU_CONFIG) $(TAU_LDFLAGS)
>>>>>>> bd6d8c4fdca0be1743de687d06b5846d0b89dc96

tau_cupti_avail.o: tau_cupti_avail.cpp
		$(TAU_CUDA_CXX) $(CFLAGS) -I$(TAU_CUDA_INC) -I$(TAU_CUPTI_INC) -I$(TAU_INC_DIR) -c $<

$(INSTALLTAU_CUPTI_AVAIL): $(TAU_CUPTI_AVAIL)
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $(TAU_CUPTI_AVAIL) $(INSTALLDEST)


clean:
	$(RM) tau_cupti_avail.o $(TAU_CUPTI_AVAIL) 

