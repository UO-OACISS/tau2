default: dms.jar 

all: dms.jar javadoc buildtest simple copytest

dms.jar: classes/dms/perfdb/*.class classes/dms/dss/*.class
	jar -cvf dms.jar -C classes .

classes/dms/perfdb/*.class: ./src/dms/perfdb/*.java
	javac -d ./classes -classpath ./classes:../contrib/postgresql.jar:../contrib/jargs.jar:../contrib/xerces.jar ./src/dms/perfdb/*.java ./src/dms/dss/*.java	

classes/dms/dss/*.class: ./src/dms/dss/*.java
	javac -d ./classes -classpath ./classes ./src/dms/dss/*.java	

./src/examples/TestPerfDBSession.class: dms.jar ./src/examples/TestPerfDBSession.java
	javac -d ./src/examples -classpath ./dms.jar ./src/examples/TestPerfDBSession.java

./src/examples/SimpleExample.class: dms.jar ./src/examples/SimpleExample.java
	javac -d ./src/examples -classpath ./dms.jar ./src/examples/SimpleExample.java

./src/examples/CopyTest.class: dms.jar ./src/examples/CopyTest.java
	javac -d ./src/examples -classpath ./dms.jar ./src/examples/CopyTest.java

buildtest: src/examples/TestPerfDBSession.class
	src/examples/testBuild

clean:
	rm -rf dms.jar ./classes/dms ./src/examples/*.class ./doc/[a-z]*

simple: ./src/examples/SimpleExample.class
	./src/examples/testSimpleExample

copytest: ./src/examples/CopyTest.class
	./src/examples/testCopy

javadoc: ./src/dms/dss/*.java ./src/dms/dss/package.html
	javadoc -d ./doc -classpath ./classes ./src/dms/dss/*.java	

remake: clean all
