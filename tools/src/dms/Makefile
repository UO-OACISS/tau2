# these are classpaths for the build.  The second classpath is for the 
# src/edu/uoregon/tau/dms/loader directory.  The first is for everything else.
CLASSPATH=./classes
LOADER_CLASSPATH=$(CLASSPATH):../contrib/jargs.jar:../contrib/xerces.jar

# These are class and source targets.
DOC_PATH=./doc
SRC_PATH=./src/edu/uoregon/tau
DATABASE_SOURCES=$(SRC_PATH)/dms/database/*.java
DATABASE_CLASSES=classes/edu/uoregon/tau/dms/database/*.class
LOADER_CLASSES=classes/edu/uoregon/tau/dms/loader/*.class
LOADER_SOURCES=$(SRC_PATH)/dms/loader/*.java
DSS_CLASSES=classes/edu/uoregon/tau/dms/dss/*.class
DSS_SOURCES=$(SRC_PATH)/dms/dss/*.java
ANALYSIS_CLASSES=classes/edu/uoregon/tau/dms/analysis/*.class
ANALYSIS_SOURCES=$(SRC_PATH)/dms/analysis/*.java
EXAMPLE_CLASSES=./src/examples/*.class
EXAMPLE_SOURCES=./src/examples/*.java

default: dms.jar 

all: dms.jar javadoc rebuilddb buildtest simple

dms.jar: $(DATABASE_CLASSES) $(DSS_CLASSES) $(LOADER_CLASSES) $(ANALYSIS_CLASSES)
	jar -cvf dms.jar -C classes .

$(DATABASE_CLASSES): $(DATABASE_SOURCES)
	javac -d $(CLASSPATH) -classpath $(CLASSPATH) $(DATABASE_SOURCES)

$(LOADER_CLASSES): $(LOADER_SOURCES) $(DSS_CLASSES)
	javac -d $(CLASSPATH) -classpath $(LOADER_CLASSPATH) $(LOADER_SOURCES)

$(DSS_CLASSES): $(DSS_SOURCES) $(DATABASE_CLASSES)
	javac -d $(CLASSPATH) -classpath $(CLASSPATH) $(DSS_SOURCES)

$(ANALYSIS_CLASSES): $(ANALYSIS_SOURCES) $(DSS_CLASSES)
	javac -d $(CLASSPATH) -classpath $(CLASSPATH) $(ANALYSIS_SOURCES)

$(EXAMPLE_CLASSES): dms.jar $(EXAMPLE_SOURCES)
	javac -d $(CLASSPATH) -classpath dms.jar $(EXAMPLE_SOURCES)

javadoc: dms.jar $(SRC_PATH)/dms/dss/package.html
	javadoc -d $(DOC_PATH) -classpath $(CLASSPATH) $(DSS_SOURCES)

# these are maintenance targets.

clean:
	rm -rf dms.jar ./classes/edu ./classes/examples ./doc/[a-z]*

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

rebuilddb: dms.jar
	dropdb perfdmf
	createdb perfdmf
	perfdmf_loadschema
	perfdmf_loadapp -x data/App_Info.xml
	perfdmf_loadexp -x data/Exp_Info.xml -a 1
	perfdmf_loadtrial -e 1 -f pprof -s data/pprof.dat -n Test
