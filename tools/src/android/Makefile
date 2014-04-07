include ../../../include/Makefile

SUBDIRS = BinaryXML dexInjector

.PHONY: subdirs $(SUBDIRS)

install: subdirs
	@sed -e 's,@JDKDIR@,'$(JDKDIR)',' \
             -e 's,@ANDROID_SDK@,'$(ANDROID_SDK)',' \
             -e 's,@TAU_BIN_DIR@,'$(TAU_BIN_DIR)',' \
             -e 's,@TAU_LIB_DIR@,'$(TAU_LIB_DIR)',' \
             tau_apk.sh.skel > tau_apk.sh
	chmod +x tau_apk.sh
	$(TAU_INSTALL) tau_apk.sh $(TAU_BIN_DIR)

subdirs: $(SUBDIRS)

$(SUBDIRS):
	make -C $@ install
