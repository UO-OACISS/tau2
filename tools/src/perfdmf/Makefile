# settings set automatically by the configure script.
CONFIG_ARCH=apple
TAUROOT=/Users/khuck/src/tau2

JAVA_JVM_VERSION=1.4

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



default: bin/perfdmf.jar 


all: perfdmf.jar javadoc rebuilddb buildtest simple

#./classes/%.class : ./src/%.java
#	javac -d $(CLASSPATH) -classpath $(CLASSPATH) $<

bin/perfdmf.jar: $(SOURCES) classes
	javac -d $(CLASSPATH) -classpath $(LOADER_CLASSPATH) $(SOURCES)
	jar -cvf bin/perfdmf.jar -C classes .
	cp bin/perfdmf.jar $(TAUROOT)/$(CONFIG_ARCH)/lib/perfdmf.jar

#dms.jar: $(DATABASE_CLASSES) $(DSS_CLASSES) $(LOADER_CLASSES) $(ANALYSIS_CLASSES)
#	jar -cvf dms.jar -C classes .


# $(DATABASE_CLASSES): $(DATABASE_SOURCES)
# 	javac -d $(CLASSPATH) -classpath $(CLASSPATH) $(DATABASE_SOURCES)

# $(LOADER_CLASSES): $(LOADER_SOURCES) $(DSS_CLASSES)
# 	javac -d $(CLASSPATH) -classpath $(LOADER_CLASSPATH) $(LOADER_SOURCES)

# $(DSS_CLASSES): $(DSS_SOURCES) $(DATABASE_CLASSES)
# 	javac -d $(CLASSPATH) -classpath $(CLASSPATH) $(DSS_SOURCES)

# $(ANALYSIS_CLASSES): $(ANALYSIS_SOURCES) $(DSS_CLASSES)
# 	javac -d $(CLASSPATH) -classpath $(CLASSPATH) $(ANALYSIS_SOURCES)


$(EXAMPLE_CLASSES): perfdmf.jar $(EXAMPLE_SOURCES)
	javac -d $(CLASSPATH) -classpath perfdmf.jar $(EXAMPLE_SOURCES)

javadoc: bin/perfdmf.jar $(SRC_PATH)/package.html
	javadoc -d ./javadoc -classpath $(CLASSPATH) $(DSS_SOURCES) $(ANALYSIS_SOURCES)

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
