##########################
#Author: Robert Bell

#Makefile creates the ParaProf jar file.
##########################

default: ParaProf.jar

ParaProf.jar: ../../dms.jar
	javac -classpath ../../dms/dms.jar *.java
	mv *.class ./paraprof
	jar -cvf ParaProf.jar paraprof
	mv ParaProf.jar ../bin

../../dms.jar:
	cd ../../dms/;make;cd ../paraprof/src

clean:
	rm -f paraprof/*.class
