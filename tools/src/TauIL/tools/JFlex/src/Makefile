# you need JFlex 1.2.2 and CUP 0.10j or later for this Makefile to work

JFLEX     = jflex -skel skeleton.nested
CUP       = java java_cup.Main 

JAVAC     = javac -target 1.1 -bootclasspath /usr/lib/jdk1.1/lib/classes.zip
#JAVAC     = jikes   
JAR       = jar

DEST      = ../lib

# jdk 1.2 
CLPATH    = -classpath /usr/lib/java/jre/lib/rt.jar:.

# jdk 1.1
#CLPATH    = -classpath /usr/lib/java/lib/classes.zip:.

# ------------------------------------------------------------

JAVAC_FLAGS = 
JARFILE     = JFlex.jar


all: jflex jar

build: clean jflex jar

install:
	cd ../bin; ln -s `pwd`/jflex /usr/bin/jflex

clean: jarclean classclean genclean
	rm -f `find . -name "*~"`

genclean:
	rm -f JFlex/sym.java
	rm -f JFlex/LexParse.java
	rm -f JFlex/LexScan.java

classclean:
	rm -rf $(DEST)/java_cup $(DEST)/JFlex
	rm -f `find . -name "*.class"`

jarclean:
	rm -f $(DEST)/$(JARFILE)

parser: JFlex/LexParse.java 

scanner: JFlex/LexScan.java

JFlex/LexScan.java: JFlex/LexScan.flex
	$(JFLEX) JFlex/LexScan.flex

JFlex/LexParse.java: JFlex/LexParse.cup
	cd JFlex; $(CUP) -interface -parser LexParse < LexParse.cup

jflex: scanner parser
	$(JAVAC) $(CLPATH) $(JAVAC_FLAGS) -d $(DEST) JFlex/Main.java JFlex/anttask/JFlexTask.java

jar: jarclean
	echo "Main-Class: JFlex.Main" > $(DEST)/manifest
	cd $(DEST); $(JAR) -c0mf manifest $(JARFILE) JFlex java_cup
	rm -rf $(DEST)/java_cup $(DEST)/JFlex $(DEST)/manifest

