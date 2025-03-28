TAU_MAKEFILE ?=../../../include/Makefile
include $(TAU_MAKEFILE)

ifndef $(TAUROOT)
TOOLS_SRC=..
else
TOOLS_SRC=$(TAUROOT)/tools/src
endif


SOURCES = $(wildcard src/*.java)


bin/vis.jar: $(SOURCES) classes
	javac -target 1.8 -source 1.8 -d ./classes -classpath $(TOOLS_SRC)/contrib/jogl/jogl-all.jar:$(TOOLS_SRC)/contrib/jogl/gluegen-rt.jar:$(TOOLS_SRC)/common/bin/tau-common.jar $(SOURCES)
	jar -cvf bin/vis-jogl2.jar -C ./classes edu
	cp bin/vis-jogl2.jar $(TAUROOT)/$(CONFIG_ARCH)/lib

javadoc: doc
	javadoc -d doc $(SOURCES)

classes:
	mkdir classes

doc:
	mkdir doc

clean:
	rm -rf bin/vis-jogl2.jar classes doc
