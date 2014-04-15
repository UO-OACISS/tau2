include ../../../../include/Makefile

INSTALLDEST = $(TAU_LIB_DIR)

PATH  := $(JDKBINDIR):$(ANDROID_SDK)/build-tools/android-4.4:$(ANDROID_SDK)/platform-tools:$(PATH)
JAVAC := javac -bootclasspath $(ANDROID_SDK)/platforms/android-19/android.jar 

dalvik_flags := -Xrunjdwp:server=y,transport=dt_android_adb

pkg_prefix := edu/uoregon/TAU

injector += $(pkg_prefix)/dexInjector/TypeDescriptor.class
injector += $(pkg_prefix)/dexInjector/MethodDescriptor.class
injector += $(pkg_prefix)/dexInjector/Filter.class
injector += $(pkg_prefix)/dexInjector/MethodAdapter.class
injector += $(pkg_prefix)/dexInjector/ClassAdapter.class
injector += $(pkg_prefix)/dexInjector/ApplicationAdapter.class
injector += $(pkg_prefix)/dexInjector/DexInjector.class

tau_env = LD_LIBRARY_PATH=/mnt/obb TAU_VERBOSE=1 PROFILEDIR=/mnt/obb

install: $(injector)
	jar cf DexInjector.jar $(injector) 'edu/uoregon/TAU/dexInjector/MethodAdapter$$DelegatedReturn.class'
	@echo Installing asmdex.jar in $(INSTALLDEST)
	@echo Installing DexInjector.jar in $(INSTALLDEST)
	if [ -d $(INSTALLDEST) ] ; then true; \
	else mkdir $(INSTALLDEST) ;fi
	$(TAU_INSTALL) asmdex.jar $(INSTALLDEST)
	$(TAU_INSTALL) DexInjector.jar $(INSTALLDEST)


stub: $(pkg_prefix)/Profile.class $(pkg_prefix)/Profiler.class
	dx --dex --output $(pkg_prefix)/Profile.dex $(pkg_prefix)/Profile.class
	dx --dex --output $(pkg_prefix)/Profiler.dex $(pkg_prefix)/Profiler.class
	java -cp asmdex.jar org.ow2.asmdex.util.AsmDexifierApplicationVisitor $(pkg_prefix)/Profile.dex > $(pkg_prefix)/Profile.stub
	java -cp asmdex.jar org.ow2.asmdex.util.AsmDexifierApplicationVisitor $(pkg_prefix)/Profiler.dex > $(pkg_prefix)/Profiler.stub

repack:
	java -cp asmdex.jar:DexInjector.jar edu.uoregon.TAU.dexInjector.DexInjector data/tomdroid/classes.dex data/tom/classes.dex >/dev/null
	rm -f data/tom/*.apk
	rm -rf data/tom/META-INF
	cp /home/ender/Downloads/tau/prefix/tau/arm_android/lib/libTAU.so data/tom/lib/armeabi/
	cd data/tom && zip -r tom.apk . >/dev/null
	jarsigner -keystore /home/ender/.android/debug.keystore -storepass android -keypass android data/tom/tom.apk androiddebugkey
	zipalign 4 data/tom/tom.apk data/tom/tom-a.apk
	adb start-server
	adb connect 128.223.202.220:5555
	adb -s 128.223.202.220:5555 wait-for-device
	adb -s 128.223.202.220:5555 usb
	adb -s CB5A1KE2U4           wait-for-device
	adb -s CB5A1KE2U4           shell pm uninstall org.tomdroid
	adb -s CB5A1KE2U4           install data/tom/tom-a.apk 
	adb -s CB5A1KE2U4           tcpip 5555
	adb kill-server

clean:
	rm -f $(injector) DexInjector.jar
	find . -name '*.dex' -exec rm -f '{}' ';'
	find . -name '*.class' -exec rm -f '{}' ';'
	find . -name '*.stub' -exec rm -f '{}' ';'

%.class: %.java
	javac -cp asmdex.jar:. $^
