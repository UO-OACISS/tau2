##############################################################
#
#                   TAU Performance System  
#
##############################################################

# 
include ../../include/Makefile
#PIN_ROOT=/usr/local/packages/pin/pin-3.2-81205-gcc-linux

CONFIG_ROOT := $(PIN_ROOT)/source/tools/Config
include $(CONFIG_ROOT)/makefile.config
include $(TOOLS_ROOT)/Config/makefile.default.rules

obj-intel64: 
	mkdir -p obj-intel64 

install: installpin

installpin: obj-intel64 obj-intel64/LinuxTauPin.so 
	cp obj-intel64/LinuxTauPin.so $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/shared$(TAU_CONFIG)/libTAU-pin.so 


#clean: 
#	/bin/rm -rf obj-intel64

# default rules build LinuxTauPin.so 
