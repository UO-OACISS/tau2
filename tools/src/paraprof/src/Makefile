##########################
#Author: Robert Bell

#Makefile creates the ParaProf jar file.
##########################

default: ParaProf.jar

ParaProf.jar:
	javac -classpath ../../dms/dms.jar *.java
	mv *.class ./paraprof
	jar -cvf ParaProf.jar paraprof
	mv ParaProf.jar ../bin

clean:
	rm -f paraprof/*.class
