
CONFIG_ARCH=x86_64
TAUROOT=/home/wspear/bin/tau2
TOOLS_SRC=$(TAUROOT)/tools/src



SOURCES = $(wildcard src/*.java)


bin/vis.jar: $(SOURCES) classes
	javac -d ./classes -classpath $(TOOLS_SRC)/contrib/jogl/jogl.jar:$(TOOLS_SRC)/common/bin/tau-common.jar $(SOURCES)
	jar -cvf bin/vis.jar -C ./classes edu
	cp bin/vis.jar $(TAUROOT)/$(CONFIG_ARCH)/lib

javadoc: doc
	javadoc -d doc $(SOURCES)

classes:
	mkdir classes

doc:
	mkdir doc

clean:
	rm -rf bin/vis.jar classes doc
