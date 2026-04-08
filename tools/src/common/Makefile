TAU_MAKEFILE ?=../../../include/Makefile

include $(TAU_MAKEFILE)


ifndef $(TAUROOT)
TOOLS_SRC=..
else
TOOLS_SRC=$(TAUROOT)/tools/src
endif

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

SOURCES = $(wildcard src/*.java) $(wildcard src/tar/*.java) $(wildcard src/treetable/*.java)

CLASSPATH = $(TOOLS_SRC)/contrib/batik-combined.jar:$(TOOLS_SRC)/contrib/jython.jar:$(TOOLS_SRC)/contrib/jfreechart-1.0.12.jar:$(TOOLS_SRC)/contrib/jcommon-1.0.15.jar

bin/tau-common.jar: $(SOURCES) classes
	javac $(JAVAC_RELEASE_FLAGS) -d ./classes -classpath $(CLASSPATH) $(SOURCES)
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
