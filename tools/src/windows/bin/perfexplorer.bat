::
:: set TAU_ROOT below and make sure java is in your path
::
@echo off
set TAU_ROOT=..


set JAR_ROOT=%TAU_ROOT%/bin
set CONFIG_FILE=%USERPROFILE%/.ParaProf/perfdmf.cfg
set PERFDMF_JAR=%JAR_ROOT%/perfdmf.jar
set JARGS_JAR=%JAR_ROOT%/jargs.jar
set JDBC_JAR=%JAR_ROOT%/postgresql.jar;%JAR_ROOT%/mysql.jar;%JAR_ROOT%/oracle.jar;%JAR_ROOT%/derby.jar
set COMMON_JAR=%JAR_ROOT%/tau-common.jar
set JARS=%JAR_ROOT%/paraprof.jar;%JAR_ROOT%/vis.jar;%PERFDMF_JAR%;%JAR_ROOT%/jogl.jar;%JAR_ROOT%/jgraph.jar;%JDBC_JAR%;%JAR_ROOT%/jargs.jar;%JAR_ROOT%/epsgraphics.jar;%JAR_ROOT%/batik-combined.jar;%JAR_ROOT%/tau-common.jar;%JAR_ROOT%/jfreechart-0.9.21.jar;%JAR_ROOT%/jcommon-0.9.6.jar

set CLASSPATH=%JARS%;%JAR_ROOT%/perfexplorer.jar;%JAR_ROOT%/weka.jar

java -Xmx500m -classpath %CLASSPATH% client.PerfExplorerClient -w -s -e weka
