all: .buildtest .simple

dms.jar: classes/dms/dss/*.class
	jar -cvf dms.jar -C classes .

classes/dms/dss/*.class: dss/*.java
	javac -d ./classes -classpath ./classes:./../perfdb/jars/perfdb.jar ./dss/*.java	

./examples/TestPerfDBSession.class: dms.jar
	javac -d ./examples -classpath ./classes:./dms.jar ./examples/TestPerfDBSession.java

./examples/SimpleExample.class: dms.jar ./examples/SimpleExample.java
	javac -d ./examples -classpath ./classes:./dms.jar ./examples/SimpleExample.java

.buildtest: examples/TestPerfDBSession.class
	examples/testBuild
	touch .buildtest

clean:
	rm -rf dms.jar classes/* examples/*.class .buildtest .simple

.simple: examples/SimpleExample.class
	examples/testSimpleExample
	touch .simple

javadoc:
	javadoc -d ./doc -classpath ./classes:./../perfdb/jars/perfdb.jar ./dss/*.java	

