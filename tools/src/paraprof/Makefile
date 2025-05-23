# Makefile for ParaProf
TAU_MAKEFILE ?=../../../include/Makefile
include $(TAU_MAKEFILE)

BASE_SRC = $(wildcard src/enums/*.java) $(wildcard src/interfaces/*.java) $(wildcard src/treetable/*.java) $(wildcard src/barchart/*.java) $(wildcard src/util/*.java) $(wildcard src/script/*.java) $(wildcard src/sourceview/*.java) $(wildcard src/tablemodel/*.java) $(wildcard src/graph/*.java) $(wildcard src/*.java)

# Remove ParaProf.java from the list since we will used the `sed` replaced one
SOURCES = $(filter-out src/ParaProf.java, $(BASE_SRC))

NOW = $(shell date)

CUBE_JAVA_READER=../contrib/CubeReader.jar
ifndef $(TAUROOT)
TOOLS_SRC=..
else
TOOLS_SRC=$(TAUROOT)/tools/src
endif
CLASSPATH = $(TOOLS_SRC)/contrib/jogl/jogl-all.jar:$(TOOLS_SRC)/contrib/jogl/gluegen-rt.jar:$(TOOLS_SRC)/contrib/jogl-all/jogl.jar:$(TOOLS_SRC)/perfdmf/bin/perfdmf.jar:$(TOOLS_SRC)/contrib/jargs.jar:$(TOOLS_SRC)/contrib/batik-combined.jar:$(TOOLS_SRC)/common/bin/tau-common.jar:$(TOOLS_SRC)/contrib/jfreechart-1.0.12.jar:$(TOOLS_SRC)/contrib/jcommon-1.0.15.jar:$(TOOLS_SRC)/contrib/jython.jar:$(TOOLS_SRC)/contrib/jgraph.jar:$(TOOLS_SRC)/vis-jogl2/bin/vis-jogl2.jar:$(TOOLS_SRC)/contrib/mesp.jar:$(CUBE_JAVA_READER)

TARGET_JAR=$(TOOLS_SRC)/paraprof/bin/paraprof.jar

# We only want to use Java 1.8
JAVA_FOUR=$(shell java -version 2>&1 | grep version | /usr/bin/awk '{ print $$3; }'| sed -e s/\"//g | sed -e s/1\.8\..*// | wc -c)
ifeq ($(JAVA_FOUR),1)
  TARGET=build
else
  TARGET=badversion
endif

default: override

all : $(TARGET_JAR)

build : $(TARGET_JAR)

$(TARGET_JAR): $(SOURCES)
	rm -rf classes	
	mkdir classes	
	sed "s/XXXXX/$(NOW)/" src/ParaProf.java > wrapper/ParaProf.java
	javac -source 1.8 -target 1.8 -d ./classes -classpath $(CLASSPATH) $(SOURCES) wrapper/ParaProf.java
	jar -cvf bin/paraprof.jar -C ./classes edu
	rm -f $(TAUROOT)/$(CONFIG_ARCH)/lib/paraprof.jar
	cp $(TOOLS_SRC)/paraprof/bin/paraprof.jar $(TAUROOT)/$(CONFIG_ARCH)/lib
	echo $(JAVAC_VERSION)

badversion:
	@echo "Error, ParaProf must be compiled with Java 1.8, use 'make override' to proceed anyway"

override: build

clean:
	rm -rf classes wrapper/ParaProf.java bin/paraprof.jar
