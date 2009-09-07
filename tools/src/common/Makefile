CONFIG_ARCH=x86_64
TAUROOT=/home/khuck/tau2
TOOLS_SRC=$(TAUROOT)/tools/src



SOURCES = $(wildcard src/*.java) $(wildcard src/tar/*.java) $(wildcard src/treetable/*.java)

CLASSPATH = $(TAUROOT)/$(CONFIG_ARCH)/lib/batik-combined.jar:$(TOOLS_SRC)/contrib/jython.jar:$(TOOLS_SRC)/contrib/jfreechart-1.0.12.jar:$(TOOLS_SRC)/contrib/jcommon-1.0.15.jar

bin/tau-common.jar: $(SOURCES) classes
	javac -target 1.4 -source 1.4 -d ./classes -classpath $(CLASSPATH) $(SOURCES)
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
