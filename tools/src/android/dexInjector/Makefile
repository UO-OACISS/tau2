include ../../../../include/Makefile

INSTALLDEST = $(TAU_LIB_DIR)

PATH  := $(JDKBINDIR):$(ANDROID_SDK)/build-tools/android-$(ANDROID_VERSION):$(ANDROID_SDK)/platform-tools:$(PATH)
JAVAC := javac -bootclasspath $(ANDROID_SDK)/platforms/android-$(ANDROID_PLATFORM)/android.jar 

dalvik_flags := -Xrunjdwp:server=y,transport=dt_android_adb

pkg_prefix := edu/uoregon/TAU

injector += $(pkg_prefix)/dexInjector/ProfileDump.class
injector += $(pkg_prefix)/dexInjector/ProfilerDump.class
injector += $(pkg_prefix)/dexInjector/TypeDescriptor.class
injector += $(pkg_prefix)/dexInjector/MethodDescriptor.class
injector += $(pkg_prefix)/dexInjector/Filter.class
injector += $(pkg_prefix)/dexInjector/MethodAdapter.class
injector += $(pkg_prefix)/dexInjector/ClassAdapter.class
injector += $(pkg_prefix)/dexInjector/ApplicationAdapter.class
injector += $(pkg_prefix)/dexInjector/DexInjector.class

install: $(injector)
	jar cf DexInjector.jar $(injector) 'edu/uoregon/TAU/dexInjector/MethodAdapter$$DelegatedReturn.class'
	@echo Installing DexInjector.jar in $(INSTALLDEST)
	if [ ! -d $(INSTALLDEST) ]; then mkdir -p $(INSTALLDEST); fi
	$(TAU_INSTALL) DexInjector.jar $(INSTALLDEST)

$(pkg_prefix)/dexInjector/ProfileDump.java: $(pkg_prefix)/Profile.class
	dx --dex --output $(pkg_prefix)/Profile.dex $^
	echo 'package edu.uoregon.TAU.dexInjector;' > $@
	java -cp $(TAU_LIB_DIR)/asmdex.jar org.ow2.asmdex.util.AsmDexifierApplicationVisitor $(pkg_prefix)/Profile.dex >> $@

$(pkg_prefix)/dexInjector/ProfilerDump.java: $(pkg_prefix)/Profiler.class
	dx --dex --output $(pkg_prefix)/Profiler.dex $^
	echo 'package edu.uoregon.TAU.dexInjector;' > $@
	java -cp $(TAU_LIB_DIR)/asmdex.jar org.ow2.asmdex.util.AsmDexifierApplicationVisitor $(pkg_prefix)/Profiler.dex >> $@

clean:
	rm -f $(injector) DexInjector.jar
	rm -f $(pkg_prefix)/dexInjector/ProfileDump.java
	rm -f $(pkg_prefix)/dexInjector/ProfilerDump.java
	find . -name '*.dex' -exec rm -f '{}' ';'
	find . -name '*.class' -exec rm -f '{}' ';'

%.class: %.java
	javac -cp $(TAU_LIB_DIR)/asmdex.jar:. $^
