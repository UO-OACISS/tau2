all: dms.jar

dms.jar: classes/dms/dss/PerfDBSession.class
	jar -cmvf manifest.txt dms.jar classes

classes/dms/dss/PerfDBSession.class: dss/PerfDBSession.java
	javac -d ./classes -classpath ./classes ./dss/Application.java  
	javac -d ./classes -classpath ./classes ./dss/Experiment.java          
	javac -d ./classes -classpath ./classes ./dss/Trial.java
	javac -d ./classes -classpath ./classes ./dss/FunctionDataObject.java  
	javac -d ./classes -classpath ./classes ./dss/Function.java            
	javac -d ./classes -classpath ./classes ./dss/DataSession.java  
	javac -d ./classes -classpath ./classes:./../perfdb/jars/perfdb.jar ./dss/PerfDBSession.java


clean:
	rm -rf dms.jar classes/*
