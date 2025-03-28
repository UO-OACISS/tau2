TAU_MAKEFILE ?=../../../include/Makefile

include $(TAU_MAKEFILE)


ifndef $(TAUROOT)
TOOLS_SRC=..
else
TOOLS_SRC=$(TAUROOT)/tools/src
endif


SOURCES = $(wildcard src/*.java) $(wildcard src/tar/*.java) $(wildcard src/treetable/*.java)

CLASSPATH = $(TOOLS_SRC)/contrib/batik-combined.jar:$(TOOLS_SRC)/contrib/jython.jar:$(TOOLS_SRC)/contrib/jfreechart-1.0.12.jar:$(TOOLS_SRC)/contrib/jcommon-1.0.15.jar

bin/tau-common.jar: $(SOURCES) classes
	javac -target 1.8 -source 1.8 -d ./classes -classpath $(CLASSPATH) $(SOURCES)
	cp -r ./resources ./classes/edu/uoregon/tau/common
	jar -cvf bin/tau-common.jar -C ./classes edu
	cp bin/tau-common.jar $(TAUROOT)/$(CONFIG_ARCH)/lib

javadoc: javadoc
	javadoc -d doc $(SOURCES)

classes:
	mkdir classes

doc:
	mkdir doc

clean:
	rm -rf bin/tau-common.jar classes doc
