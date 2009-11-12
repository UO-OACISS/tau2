::
:: set TAU_ROOT below and make sure java is in your path
::
@echo off
set TAU_ROOT=..


set JAR_ROOT=%TAU_ROOT%/bin
set ETC=%TAU_ROOT%\etc
set CONFIG_FILE=%USERPROFILE%/.ParaProf/perfdmf.cfg
set PERFDMF_JAR=%JAR_ROOT%/perfdmf.jar
set JARGS_JAR=%JAR_ROOT%/jargs.jar
set JDBC_JAR=%JAR_ROOT%/postgresql.jar;%JAR_ROOT%/mysql.jar;%JAR_ROOT%/oracle.jar;%JAR_ROOT%/derby.jar
set COMMON_JAR=%JAR_ROOT%/tau-common.jar
set JARS=%JAR_ROOT%/paraprof.jar;%JAR_ROOT%/vis.jar;%PERFDMF_JAR%;%JAR_ROOT%/jogl.jar;%JAR_ROOT%/jgraph.jar;%JDBC_JAR%;%JAR_ROOT%/jargs.jar;%JAR_ROOT%/epsgraphics.jar;%JAR_ROOT%/batik-combined.jar;%JAR_ROOT%/tau-common.jar;%JAR_ROOT%/jfreechart-1.0.12.jar;%JAR_ROOT%/jcommon-1.0.15.jar

set CLASSPATH=%JARS%;%JAR_ROOT%/perfexplorer.jar;%JAR_ROOT%/weka.jar;%JAR_ROOT%/antlr-2.7.6.jar;%JAR_ROOT%/commons-lang-2.1.jar;%JAR_ROOT%/drools-core-3.0.6.jar;%JAR_ROOT%/junit-3.8.1.jar;%JAR_ROOT%/antlr-3.0ea8.jar;%JAR_ROOT%/commons-logging-api-1.0.4.jar;%JAR_ROOT%/drools-decisiontables-3.0.6.jar;%JAR_ROOT%/jxl-2.4.2.jar;%JAR_ROOT%/commons-jci-core-1.0-406301.jar;%JAR_ROOT%/core-3.2.0.666.jar;%JAR_ROOT%/drools-jsr94-3.0.6.jar;%JAR_ROOT%/stringtemplate-2.3b6.jar;%JAR_ROOT%/commons-jci-eclipse-3.2.0.666.jar;%JAR_ROOT%/drools-compiler-3.0.6.jar;%JAR_ROOT%/jsr94-1.1.jar


java -Xmx500m -classpath %CLASSPATH% edu.uoregon.tau.perfexplorer.client.PerfExplorerClient -t %JAR_ROOT% -a %ETC% -w -s
