all: jRacy.jar

jRacy.jar: jRacy/jRacy.class
	jar -cmvf manifest.txt jRacy.jar jRacy

jRacy/jRacy.class: jRacy.class
	mv *.class jRacy

jRacy.class: jRacy.java
	javac *.java

clean:
	rm -f jRacy.jar jRacy/*.class
