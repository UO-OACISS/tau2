##########################
#Author: Robert Bell

#Makefile creates the ParaProf jar file.
##########################

default: ParaProf.jar

ParaProf.jar:
	javac *.java
	mv *.class ./paraprof
	jar -cmvf manifest.txt ParaProf.jar paraprof
	mv ParaProf.jar ../bin

clean:
	rm -f paraprof/*.class
