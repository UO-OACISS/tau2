TAU_MAKEFILE=../../include/Makefile
include $(TAU_MAKEFILE)

DOC_PATH=$(TAUROOT)/documentation

JAVA_LOCATIONS=common vis perfdmf paraprof perfexplorer

JAVA_SRCS=$(shell find -L common/edu vis/edu perfdmf/edu paraprof/edu perfexplorer/edu -name '*.java' 2>/dev/null)
#JAVA_SRCS=$(call rwildcard perfdmf/src/edu/ *.java)

PERFEXPLORER_JAR=$(TAUROOT)/$(CONFIG_ARCH)/lib/perfexplorer.jar
PARAPROF_JAR=$(TAUROOT)/$((CONFIG_ARCH)/lib/paraprof.jar
PERFDMF_JAR=$(TAUROOT)/$(CONFIG_ARCH)/lib/perfdmf.jar
JARGS_JAR=$(TAUROOT)/$(CONFIG_ARCH)/lib/jargs.jar
GSON_JAR=$(TAUROOT)/$(CONFIG_ARCH)/lib/gson-2.1.jar
JYTHON_JAR=$(TAUROOT)/$(CONFIG_ARCH)/lib/jython.jar
WEKA_JAR=$(JAR_HOME)/weka-3-6-1.jar
PERFEXPLORER_JAR=$(TAUROOT)/$(CONFIG_ARCH)/lib/perfexplorer.jar
JFREECHART_JAR=$(TAUROOT)/$(CONFIG_ARCH)/lib/jfreechart-1.0.12.jar
JCOMMON_JAR=$(TAUROOT)/$(CONFIG_ARCH)/lib/jcommon-1.0.15.jar
JOGL_JAR=$(TAUROOT)/$(CONFIG_ARCH)/lib/jogl.jar
VIS_JAR=$(TAUROOT)/$(CONFIG_ARCH)/lib/vis.jar
TAUCOMMON_JAR=$(TAUROOT)/$(CONFIG_ARCH)/lib/tau-common.jar
BATIK_JAR=$(TAUROOT)/$(CONFIG_ARCH)/lib/batik-combined.jar
XERCES_JAR=$(TAUROOT)/$(CONFIG_ARCH)/lib/xerces.jar
JUNIT_JAR=$(TAUROOT)/$(CONFIG_ARCH)/junit-3.8.1.jar

CLASSPATH=$(PERFEXPLORER_JAR):$(PARAPROF_JAR):$(PERFDMF_JAR):$(JDBC_JAR):$(JFREECHART_JAR):$(JCOMMON_JAR):$(WEKA_JAR):$(JOGL_JAR):$(VIS_JAR):$(JARGS_JAR):$(GSON_JAR):$(JYTHON_JAR):$(TAUCOMMON_JAR):$(BATIK_JAR):$(XERCES_JAR):$(JUNIT_JAR)

javadoc: $(DOC_PATH)/index.html

$(DOC_PATH)/index.html:
	javadoc -d $(DOC_PATH) \
	-classpath $(CLASSPATH) \
	-sourcepath $(JAVA_SRCS) \
    -use \
    -windowtitle "PerfDMF" \
    -doctitle "PerfDMF" \
    -header "<b>PerfDMF</b>" \
    -bottom ${BOTTOM} \
    -public \
    -source 1.5 \
    -quiet \
    -version \
    -author \
    -noqualifier all \
    -nodeprecated \
		> /dev/null 2>1

