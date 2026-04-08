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

# Java release target: default to 8 for HPC legacy compatibility.
# Override with: make JAVA_RELEASE=17  (or 21, 25, etc.)
# Set JAVA_RELEASE=none to skip release flags and compile at the JDK default.
JAVA_RELEASE ?= 8

# --release requires JDK 9+; JDK 8 and older use -source/-target instead.
_JAVAC_MAJ := $(shell javac -version 2>&1 | sed 's/javac \([0-9]*\).*/\1/')
ifneq ($(JAVA_RELEASE),none)
  ifeq ($(_JAVAC_MAJ),1)
    JAVAC_RELEASE_FLAGS := -source $(JAVA_RELEASE) -target $(JAVA_RELEASE)
  else
    JAVAC_RELEASE_FLAGS := --release $(JAVA_RELEASE)
  endif
else
  JAVAC_RELEASE_FLAGS :=
endif

# Detect major Java version for user guidance (works with both "1.8.0" and "25.0.2" formats)
JAVA_MAJOR=$(shell java -version 2>&1 | awk -F'[".]+' '/version/ {if($$2 == "1") print $$3; else print $$2}')
JAVA_OK=$(shell [ $(JAVA_MAJOR) -ge 8 ] 2>/dev/null && echo yes || echo no)

ifeq ($(JAVA_OK),yes)
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
	javac $(JAVAC_RELEASE_FLAGS) -d ./classes -classpath $(CLASSPATH) $(SOURCES) wrapper/ParaProf.java
	jar -cvf bin/paraprof.jar -C ./classes edu
	rm -f $(TAUROOT)/$(CONFIG_ARCH)/lib/paraprof.jar
	cp $(TOOLS_SRC)/paraprof/bin/paraprof.jar $(TAUROOT)/$(CONFIG_ARCH)/lib
	echo $(JAVAC_VERSION)

badversion:
	@echo "Error: Java >= 8 required. Detected Java $(JAVA_MAJOR). Use 'make override' to proceed anyway."

override: build

clean:
	rm -rf classes wrapper/ParaProf.java bin/paraprof.jar
