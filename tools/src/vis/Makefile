TAU_MAKEFILE ?=../../../include/Makefile
include $(TAU_MAKEFILE)

ifndef $(TAUROOT)
TOOLS_SRC=..
else
TOOLS_SRC=$(TAUROOT)/tools/src
endif


SOURCES = $(wildcard src/*.java)


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

bin/vis.jar: $(SOURCES) classes
	javac $(JAVAC_RELEASE_FLAGS) -d ./classes -classpath $(TOOLS_SRC)/contrib/jogl/jogl.jar:$(TOOLS_SRC)/common/bin/tau-common.jar $(SOURCES)
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
