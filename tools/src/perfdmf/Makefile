# settings set automatically by the configure script.
CONFIG_ARCH=i386_linux
TAUROOT=/home/khuck/tau2

# these are classpaths for the build.  The second classpath is for the 
# loader directory.  The first is for everything else.
CLASSPATH=./classes
LOADER_CLASSPATH=$(CLASSPATH):../contrib/jargs.jar:../contrib/xerces.jar:../common/bin/tau-common.jar

# These are class and source targets.
DOC_PATH=./doc
SRC_PATH=./src
DATABASE_SOURCES=$(wildcard $(SRC_PATH)/database/*.java)
LOADER_SOURCES=$(wildcard $(SRC_PATH)/loader/*.java)
DSS_SOURCES=$(wildcard $(SRC_PATH)/*.java)
ANALYSIS_SOURCES=$(wildcard $(SRC_PATH)/analysis/*.java)
EXAMPLE_CLASSES=./src/examples/*.class
EXAMPLE_SOURCES=./src/examples/*.java


SOURCES = $(DATABASE_SOURCES) $(DSS_SOURCES) $(LOADER_SOURCES) $(ANALYSIS_SOURCES)
CLASSES = $(subst ./src/, ./classes/, $(patsubst %.java, %.class, $(SOURCES)))



# We only want to use Java 1.4
JAVA_FOUR=$(shell java -version 2>&1 | grep version | /usr/bin/awk '{ print $$3; }'| sed -e s/\"//g | sed -e s/1\.4\..*// | wc -c)
ifeq ($(JAVA_FOUR),1)
  TARGET=build
else
  TARGET=badversion
endif


all: $(TARGET)

build: bin/perfdmf.jar 


#all: perfdmf.jar javadoc rebuilddb buildtest simple



bin/perfdmf.jar: $(SOURCES) classes
	javac -d $(CLASSPATH) -classpath $(LOADER_CLASSPATH) $(SOURCES)
	jar -cvf bin/perfdmf.jar -C classes .
	cp bin/perfdmf.jar $(TAUROOT)/$(CONFIG_ARCH)/lib/perfdmf.jar



$(EXAMPLE_CLASSES): perfdmf.jar $(EXAMPLE_SOURCES)
	javac -d $(CLASSPATH) -classpath perfdmf.jar $(EXAMPLE_SOURCES)

BOTTOM = '<font size="-1">****************************************************************************<br>Copyright 1997-2009<br> Department of Computer and Information Science, University of Oregon<br> Advanced Computing Laboratory, Los Alamos National Laboratory<br> Juelich Supercomputing Centre, Research Center Juelich, Germany<br><a href="http://tau.uoregon.edu" target="new">http://tau.uoregon.edu</a><br>****************************************************************************<br>Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without fee is hereby granted, provided that the above copyright notice appear in all copies and that both that copyright notice and this permission notice appear in supporting documentation, and that the name of University of Oregon (UO) and Los Alamos National Laboratory (LANL) not be used in advertising or publicity pertaining to distribution of the software without specific, written prior permission.  The University of Oregon and LANL makes no representations about the suitability of this software for any purpose.  It is provided "as is" with out express or implied warranty.<br><br>THE UNIVERSITY OF OREGON AND LANL DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL THE UNIVERSITY OF OREGON OR LANL BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.<font>'

javadoc: bin/perfdmf.jar $(SRC_PATH)/package.html
	javadoc -d ./javadoc \
	-classpath $(CLASSPATH) $(DSS_SOURCES) $(ANALYSIS_SOURCES) \
	-sourcepath src \
    -use \
    -windowtitle "PerfDMF" \
    -doctitle "PerfDMF" \
    -header "<b>PerfDMF</b>" \
    -bottom ${BOTTOM} \
    -public \
    -source 1.4 \
    -quiet \
    -version \
    -author \
    -noqualifier all \
    -nodeprecated \
	edu.uoregon.tau.perfdmf

# these are maintenance targets.

clean:
	rm -rf bin/perfdmf.jar ./classes ./javadoc

remake: clean all

# These targets are all for testing purposes.

buildtest: $(EXAMPLE_CLASSES)
	./src/examples/testBuild

simple: $(EXAMPLE_CLASSES)
	./src/examples/testSimpleExample

scalability: $(EXAMPLE_CLASSES)
	./src/examples/testScalabilityExample

copytest: $(EXAMPLE_CLASSES)
	./src/examples/testCopy

simple.db2: $(EXAMPLE_CLASSES)
	./src/examples/testSimpleExample.db2

buildtest.db2: $(EXAMPLE_CLASSES)
	./src/examples/testBuild.db2

testdelete: $(EXAMPLE_CLASSES)
	./src/examples/testDelete

testsave: $(EXAMPLE_CLASSES)
	./src/examples/testSave

analysis: $(EXAMPLE_CLASSES)
	./src/examples/testAnalysis

rebuilddb: dms.jar
	dropdb perfdmf_buildtest
	createdb perfdmf_buildtest
	perfdmf_loadschema
	perfdmf_loadapp -x data/App_Info.xml
	perfdmf_loadexp -x data/Exp_Info.xml -a 1
	perfdmf_loadtrial -e 1 -f pprof -s data/pprof.dat -n Test

rebuilddb_and_test: rebuilddb simple buildtest

classes:
	mkdir classes


badversion:
	@echo "Error, ParaProf must be compiled with Java 1.4, use 'make override' to proceed anyway"

override: build


