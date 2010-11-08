# Set by TAU configure script
CONFIG_ARCH=default
TAU_PREFIX_INSTALL_DIR=
OTFINC=
OTFLIB=

# Fixed Variables, do not modify
TAU_LIB_DIR=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib
EBS2OTF_DIR=$(TAU_PREFIX_INSTALL_DIR)/utils/ebs2otf
OTF_1_8_INCLUDES=OTF_CopyHandler.h OTF_FileManager.h OTF_inttypes.h \
	OTF_Reader.h OTF_Writer.h OTF_Definitions.h OTF_Filenames.h \
	OTF_inttypes_unix.h OTF_RStream.h OTF_WStream.h OTF_Errno.h \
	otf.h OTF_MasterControl.h OTF_Version.h OTF_File.h \
	OTF_HandlerArray.h OTF_RBuffer.h OTF_WBuffer.h OTF_KeyValue.h

all: ebs2otf.so

ebs2otf.so: reswig.sh $(OTF_1_8_INCLUDES) ebs2otf.i
	./reswig.sh $(OTFLIB)

%.h:
	cp $(OTFINC)/$@ $(EBS2OTF_DIR)

# *CWL* cleanup is meant to be called by utils/Makefile in the event
#   configure is called again, with the possibility of new OTF locations.
#   configure is expected touch ebs2otf.so utils/Makefile can be triggered.
cleanup:
	rm -f $(OTF_1_8_INCLUDES) ebs2otf.pm ebs2otf_wrap.c *.o *~ 

# Meant for standalone installation, not to be used by utils/Makefile
install: ebs2otf.so
	cp ebs2otf.so $(TAU_LIB_DIR)
	cp ebs2otf.pm $(TAU_LIB_DIR)

uninstall: clean
	rm -f $(TAU_LIB_DIR)/ebs2otf.so $(TAU_LIB_DIR)/ebs2otf.pm

clean: cleanup
	rm -f ebs2otf.so

