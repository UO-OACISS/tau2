
CONFIG_ARCH=default
TAUROOT=
TOOLS_SRC=$(TAUROOT)/tools/src



SOURCES = $(wildcard src/*.java)


bin/vis.jar: $(SOURCES) classes bin
	javac -d ./classes -classpath $(TOOLS_SRC)/contrib/jogl.jar $(SOURCES)
	jar -cvf bin/vis.jar -C ./classes edu
	cp bin/vis.jar $(TAUROOT)/$(CONFIG_ARCH)/lib

javadoc: doc
	javadoc -d doc $(SOURCES)

classes:
	mkdir classes

bin:
	mkdir bin

doc:
	mkdir doc

clean:
	rm -rf bin classes doc
