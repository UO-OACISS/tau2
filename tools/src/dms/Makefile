all: buildtest.txt

dms.jar: classes/dms/dss/PerfDBSession.class
	jar -cvf dms.jar -C classes .

classes/dms/dss/PerfDBSession.class: dss/PerfDBSession.java
	javac -d ./classes -classpath ./classes:./../perfdb/jars/perfdb.jar ./dss/*.java	

./examples/TestPerfDBSession.class: dms.jar
	javac -d ./examples -classpath ./classes:./dms.jar ./examples/*.java

buildtest.txt: examples/TestPerfDBSession.class
	examples/testBuild >& buildtest.txt
	# Build test results are in buildtest.txt

clean:
	rm -rf dms.jar classes/* examples/*.class buildtest.txt
