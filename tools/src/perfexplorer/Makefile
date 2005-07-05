TAUROOT = /home/amorris/tau2
CONFIG_ARCH = i386_linux
CONFIGFILE = /home/amorris/.ParaProf/perfdmf.cfg
R_HOME = /usr/lib/R
OMEGAHAT_HOME = /home/amorris/PerfExplorer/SJava/inst
OMEGAHAT_JARPATH = $(OMEGAHAT_HOME)/org/omegahat/Jars
SERVER_HOSTNAME=utonium.cs.uoregon.edu
# SERVER_HOSTNAME=localhost
SERVER_PORT=2001

PERFDMF_JAR = ${TAUROOT}/${CONFIG_ARCH}/lib/dms.jar
XMLSAX_JAR = ${TAUROOT}/${CONFIG_ARCH}/lib/xerces.jar
JARGS_JAR = ${TAUROOT}/${CONFIG_ARCH}/lib/jargs.jar
JDBC_JAR = ${TAUROOT}/${CONFIG_ARCH}/lib/postgresql.jar
JOGL_JAR =${TAUROOT}/${CONFIG_ARCH}/lib/jogl.jar:${TAUROOT}/${CONFIG_ARCH}/lib/ParaProf.jar
OMEGAHAT_JARS = $(OMEGAHAT_HOME):$(OMEGAHAT_JARPATH)/Environment.jar:$(OMEGAHAT_JARPATH)/antlr.jar:$(OMEGAHAT_JARPATH)/jas.jar:$(OMEGAHAT_JARPATH)/jhall.jar
JFREECHART_JAR = ./JFreeChart/jfreechart-0.9.21.jar
JCOMMON_JAR = ./JFreeChart/jcommon-0.9.6.jar
WEKA_JAR = /home/amorris/PerfExplorer/weka.jar

CLIENT_DYLD_LIBRARY_PATH=${TAUROOT}/${CONFIG_ARCH}/lib:/usr/lib/R/library/SJava/libs
SERVER_DYLD_LIBRARY_PATH=$(OMEGAHAT_HOME)/libs:$(R_HOME)/lib:$(R_HOME)/bin:$(R_HOME)/library/SJava/libs:/usr/lib/R/library/SJava/libs

BUILD_CLASSPATH = classes:$(PERFDMF_JAR):$(JOGL_JAR):$(PARAPROF_JAR)
CLIENT_BUILD_CLASSPATH = classes:$(PERFDMF_JAR):$(JFREECHART_JAR):$(JCOMMON_JAR):$(JOGL_JAR):$(PARAPROF_JAR)
CLUSTERING_BUILD_CLASSPATH = classes:$(PERFDMF_JAR):$(WEKA_JAR)
SERVER_BUILD_CLASSPATH = classes:$(PERFDMF_JAR):$(OMEGAHAT_JARS):$(WEKA_JAR):$(JFREECHART_JAR):$(JCOMMON_JAR)
CLIENT_RUN_CLASSPATH = classes:$(PERFDMF_JAR):$(XMLSAX_JAR):$(JDBC_JAR):$(JFREECHART_JAR):$(JCOMMON_JAR):$(JOGL_JAR):$(PARAPROF_JAR)
SERVER_RUN_CLASSPATH = classes:$(PERFDMF_JAR):$(XMLSAX_JAR):$(JDBC_JAR):$(OMEGAHAT_JARS):$(WEKA_JAR):$(JFREECHART_JAR):$(JCOMMON_JAR)
STANDALONE_RUN_CLASSPATH = classes:$(PERFDMF_JAR):$(XMLSAX_JAR):$(JDBC_JAR):$(OMEGAHAT_JARS):$(JFREECHART_JAR):$(JCOMMON_JAR):$(WEKA_JAR):$(JOGL_JAR):$(PARAPROF_JAR)

COMMON_SOURCES = $(wildcard src/common/*.java)
CLUSTERING_SOURCES = $(wildcard src/clustering/*.java)
CLIENT_SOURCES = $(wildcard src/client/*.java)
SERVER_SOURCES = $(wildcard src/server/*.java)
SOURCES = $(COMMON_SOURCES) $(SERVER_SOURCES) $(CLIENT_SOURCES) $(CLUSTERING_SOURCES)

COMMON_CLASSES = $(subst src/, classes/, $(patsubst %.java, %.class, $(COMMON_SOURCES))) 
CLUSTERING_CLASSES = $(subst src/, classes/, $(patsubst %.java, %.class, $(CLUSTERING_SOURCES))) 
SERVER_CLASSES = $(subst src/, classes/, $(patsubst %.java, %.class, $(SERVER_SOURCES))) 
CLIENT_CLASSES = $(subst src/, classes/, $(patsubst %.java, %.class, $(CLIENT_SOURCES))) 
CLASSES = $(COMMON_CLASSES) $(SERVER_CLASSES) $(CLIENT_CLASSES) $(CLUSTERING_CLASSES)
RMIC_CLASSES = classes/server/PerfExplorerServer_Skel.class

default: all

###
### THESE COMMANDS ARE FOR COMPILING
###

all: common clustering server client perfexplorer.jar

common: $(COMMON_CLASSES)

clustering: $(CLUSTERING_CLASSES)

server: common clustering $(SERVER_CLASSES) $(RMIC_CLASSES)

client: common clustering $(CLIENT_CLASSES)

standalone: common clustering server client

perfexplorer.jar: standalone
	jar -cvf perfexplorer.jar -C classes .

$(COMMON_CLASSES): $(COMMON_SOURCES)
	javac -source 1.4 -classpath $(BUILD_CLASSPATH) -d classes $(COMMON_SOURCES)

$(CLUSTERING_CLASSES): $(CLUSTERING_SOURCES)
	javac -source 1.4 -classpath $(CLUSTERING_BUILD_CLASSPATH) -d classes $(CLUSTERING_SOURCES)

$(SERVER_CLASSES): $(SERVER_SOURCES) $(COMMON_CLASSES)
	javac -source 1.4 -classpath $(SERVER_BUILD_CLASSPATH) -d classes $(SERVER_SOURCES)

$(CLIENT_CLASSES): $(CLIENT_SOURCES) $(SERVER_CLASSES)
	javac -source 1.4 -classpath $(CLIENT_BUILD_CLASSPATH) -d classes $(CLIENT_SOURCES)
	cp src/client/resources/*.gif classes/client/.

$(RMIC_CLASSES): $(SERVER_CLASSES)
	rmic -classpath $(SERVER_BUILD_CLASSPATH) -d classes server.PerfExplorerServer

###
### THESE COMMANDS ARE FOR CLEANING
###

cleanclient:
	rm -rf classes/client
	# rm -f $(CLIENT_CLASSES) classes/client/*.gif classes/client/*.class

cleanserver:
	rm -rf classes/server
	# rm -f $(SERVER_CLASSES)

cleancommon:
	rm -rf classes/common
	# rm -f $(COMMON_CLASSES)

cleanclustering:
	rm -rf classes/clustering
	# rm -f $(CLUSTERING_CLASSES)

clean: cleanclient cleanserver cleancommon cleanclustering
	rm -rf PerfExplorer-0.1/*.jar PerfExplorer-0.1/java.policy
	rm -f PerfExplorer-0.1.tgz

remake: clean all

###
### THESE COMMANDS ARE FOR RUNNING/TESTING
###

testserver: server
	java -ea -cp $(SERVER_RUN_CLASSPATH) server.TestServer $(CONFIGFILE)

runsupport:
	# start the RMI registry
	rmiregistry &
	# start the web server
	# java -cp web examples.classServer.ClassFileServer $(SERVER_PORT) /Users/khuck/PerfExplorer/classes &

runserver: server
	# start the PerfExplorer server
	java -ea -Xmx1800m -cp $(SERVER_RUN_CLASSPATH) \
	-Djava.rmi.server.codebase=http://$(SERVER_HOSTNAME):$(SERVER_PORT)/ \
	-Djava.security.policy=data/java.policy \
	-Djava.library.path=${SERVER_DYLD_LIBRARY_PATH} \
	server.PerfExplorerServer $(SERVER_PORT) \
	/Users/khuck/PerfExplorer/classes $(CONFIGFILE) weka

runclient: client
	java -ea -Xmx500m -cp $(CLIENT_RUN_CLASSPATH) \
	-Djava.rmi.server.codebase=http://$(SERVER_HOSTNAME):$(SERVER_PORT)/ \
	-Djava.security.policy=data/java.policy \
	-Djava.library.path=${CLIENT_DYLD_LIBRARY_PATH} \
	client.PerfExplorerClient $(SERVER_HOSTNAME)

runstandalone: standalone
	java -ea -Xmx500m -cp $(STANDALONE_RUN_CLASSPATH) \
	-Djava.security.policy=data/java.policy \
	-Djava.library.path=${CLIENT_DYLD_LIBRARY_PATH}:${SERVER_DYLD_LIBRARY_PATH} \
	client.PerfExplorerClient $(SERVER_HOSTNAME) \
	--standalone $(CONFIGFILE) weka

install: perfexplorer.jar
	rm -rf PerfExplorer-0.1/*.jar PerfExplorer-0.1/java.policy
	rm -f PerfExplorer-0.1.tgz
	cp perfexplorer.jar PerfExplorer-0.1/.
	cp $(PERFDMF_JAR) PerfExplorer-0.1/.
	cp $(XMLSAX_JAR) PerfExplorer-0.1/.
	cp $(JARGS_JAR) PerfExplorer-0.1/.
	cp $(JDBC_JAR) PerfExplorer-0.1/.
	cp $(JOGL_JAR) PerfExplorer-0.1/.
	cp $(PARAPROF_JAR) PerfExplorer-0.1/.
	cp $(JFREECHART_JAR) PerfExplorer-0.1/.
	cp $(JCOMMON_JAR) PerfExplorer-0.1/.
	cp $(WEKA_JAR) PerfExplorer-0.1/.
	#cp $(OMEGAHAT_HOME) PerfExplorer-0.1/.
	cp $(OMEGAHAT_JARPATH)/Environment.jar PerfExplorer-0.1/.
	cp $(OMEGAHAT_JARPATH)/antlr.jar PerfExplorer-0.1/.
	cp $(OMEGAHAT_JARPATH)/jas.jar PerfExplorer-0.1/.
	cp $(OMEGAHAT_JARPATH)/jhall.jar PerfExplorer-0.1/.
	cp ./JFreeChart/jfreechart-0.9.21.jar PerfExplorer-0.1/.
	cp ./JFreeChart/jcommon.jar PerfExplorer-0.1/.
	cp data/java.policy PerfExplorer-0.1/.
	chmod 444 PerfExplorer-0.1/*.jar
	chmod 775 PerfExplorer-0.1/pe.sh
	tar -cvf PerfExplorer-0.1.tar PerfExplorer-0.1
	# cp PerfExplorer-0.1.tgz ~/tmp/.
	scp PerfExplorer-0.1.tar ix.cs.uoregon.edu:public_html/PerfExplorer-0.1.tar

webstart: perfexplorer.jar
	cp perfexplorer.jar /Library/WebServer/Documents/perfexplorer/.
	cp $(PERFDMF_JAR) /Library/WebServer/Documents/perfexplorer/.
	cp $(XMLSAX_JAR) /Library/WebServer/Documents/perfexplorer/.
	cp $(JARGS_JAR) /Library/WebServer/Documents/perfexplorer/.
	cp $(JDBC_JAR) /Library/WebServer/Documents/perfexplorer/.
	cp $(JOGL_JAR) /Library/WebServer/Documents/perfexplorer/.
	cp $(PARAPROF_JAR) /Library/WebServer/Documents/perfexplorer/.
	cp $(JFREECHART_JAR) /Library/WebServer/Documents/perfexplorer/.
	cp $(JCOMMON_JAR) /Library/WebServer/Documents/perfexplorer/.
	cp $(WEKA_JAR) /Library/WebServer/Documents/perfexplorer/.
	#cp $(OMEGAHAT_HOME) /Library/WebServer/Documents/perfexplorer/.
	cp $(OMEGAHAT_JARPATH)/Environment.jar /Library/WebServer/Documents/perfexplorer/.
	cp $(OMEGAHAT_JARPATH)/antlr.jar /Library/WebServer/Documents/perfexplorer/.
	cp $(OMEGAHAT_JARPATH)/jas.jar /Library/WebServer/Documents/perfexplorer/.
	cp $(OMEGAHAT_JARPATH)/jhall.jar /Library/WebServer/Documents/perfexplorer/.
	cp ./JFreeChart/jfreechart-0.9.21.jar /Library/WebServer/Documents/perfexplorer/.
	cp ./JFreeChart/jcommon.jar /Library/WebServer/Documents/perfexplorer/.
	cp data/java.policy /Library/WebServer/Documents/perfexplorer/.
	chmod 777 /Library/WebServer/Documents/perfexplorer/*.jar

###
### END
###
