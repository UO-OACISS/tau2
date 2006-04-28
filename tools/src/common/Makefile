
CONFIG_ARCH=i386_linux
TAUROOT=/home/users/khuck/tau2
TOOLS_SRC=$(TAUROOT)/tools/src



SOURCES = $(wildcard src/*.java) $(wildcard src/tar/*.java)

CLASSPATH = $(TAUROOT)/$(CONFIG_ARCH)/lib/batik-combined.jar:$(TOOLS_SRC)/contrib/jython.jar

bin/tau-common.jar: $(SOURCES) classes
	javac -d ./classes -classpath $(CLASSPATH) $(SOURCES)
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
