all: buildtest.txt simple.txt

dms.jar: classes/dms/dss/PerfDBSession.class
	jar -cvf dms.jar -C classes .

classes/dms/dss/PerfDBSession.class: dss/PerfDBSession.java
	javac -d ./classes -classpath ./classes:./../perfdb/jars/perfdb.jar ./dss/*.java	

./examples/TestPerfDBSession.class: dms.jar
	javac -d ./examples -classpath ./classes:./dms.jar ./examples/TestPerfDBSession.java

./examples/SimpleExample.class: dms.jar ./examples/SimpleExample.java
	javac -d ./examples -classpath ./classes:./dms.jar ./examples/SimpleExample.java

buildtest.txt: examples/TestPerfDBSession.class
	examples/testBuild
	touch buildtest.txt

clean:
	rm -rf dms.jar classes/* examples/*.class buildtest.txt

simple.txt: examples/SimpleExample.class
	examples/testSimpleExample
	touch simple.txt


