# settings

TARGET = DrawImage

CONFIG_FILE=/home/huck2/.ParaProf/perfdmf.cfg
TAUROOT=/home/huck2/tau2
CONFIG_ARCH=i386_linux
PERFDMF_HOME=${TAUROOT}/tools/src/dms

XMLSAX_JAR=${TAUROOT}/${CONFIG_ARCH}/lib/xerces.jar
JDBC_JAR=${TAUROOT}/${CONFIG_ARCH}/lib/postgresql.jar
DMS_JAR=${PERFDMF_HOME}/dms.jar

CLASSPATH = .:${DMS_JAR}:${XMLSAX_JAR}:${JDBC_JAR}

CLASS_TARGETS = DrawImage.class PlotDrawingWindow.class

# build instructions

default: run

all: $(TARGET).class

$(TARGET).class: $(TARGET).java
	javac -classpath $(CLASSPATH) *.java

run: all
	java -classpath $(CLASSPATH) $(TARGET) $(CONFIG_FILE) 55 0

clean:
	rm -f *.class
