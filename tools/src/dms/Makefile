default: dms.jar 

all: dms.jar javadoc .buildtest .simple

dms.jar: classes/dms/dss/*.class
	jar -cvf dms.jar -C classes .

classes/dms/dss/*.class: dss/*.java
	javac -d ./classes -classpath ./classes:./../perfdb/jars/perfdb.jar ./dss/*.java	

./examples/TestPerfDBSession.class: dms.jar ./examples/TestPerfDBSession.java
	javac -d ./examples -classpath ./classes:./dms.jar ./examples/TestPerfDBSession.java

./examples/SimpleExample.class: dms.jar ./examples/SimpleExample.java
	javac -d ./examples -classpath ./classes:./dms.jar ./examples/SimpleExample.java

.buildtest: examples/TestPerfDBSession.class
	examples/testBuild >& .buildtest
	tail .buildtest

clean:
	rm -rf dms.jar classes/dms examples/*.class .buildtest .simple doc/*.html doc/*.css doc/package-list doc/dms doc/resources

.simple: examples/SimpleExample.class
	examples/testSimpleExample >& .simple
	tail .simple

javadoc: ./dss/*.java ./dss/package.html
	javadoc -d ./doc -classpath ./classes:./../perfdb/jars/perfdb.jar ./dss/*.java	

remake: clean all
