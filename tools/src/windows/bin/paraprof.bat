::
:: set TAU_ROOT below and make sure java is in your path
::
@echo off
set TAU_ROOT=..


set JAR_ROOT=%TAU_ROOT%/bin
set PERFDMF_JAR=%JAR_ROOT%/perfdmf.jar
set JARGS_JAR=%JAR_ROOT%/jargs.jar
set JDBC_JAR=%JAR_ROOT%/postgresql.jar;%JAR_ROOT%/mysql.jar;%JAR_ROOT%/oracle.jar;%JAR_ROOT%/derby.jar
set COMMON_JAR=%JAR_ROOT%/tau-common.jar
set JARS=%JAR_ROOT%/paraprof.jar;%JAR_ROOT%/vis.jar;%PERFDMF_JAR%;%JAR_ROOT%/jogl.jar;%JAR_ROOT%/jgraph.jar;%JDBC_JAR%;%JAR_ROOT%/jargs.jar;%JAR_ROOT%/epsgraphics.jar;%JAR_ROOT%/batik-combined.jar;%JAR_ROOT%/tau-common.jar;%JAR_ROOT%/jfreechart-0.9.21.jar;%JAR_ROOT%/jcommon-0.9.6.jar;%JAR_ROOT%/xerces.jar

echo.
java -Xmx500m -Djava.library.path=%JAR_ROOT% -Dderby.system.home="%HOMEPATH%/.ParaProf" -cp %JARS% edu/uoregon/tau/paraprof/ParaProf %1 %2 %3 %4 %5

:handleError
if not errorlevel 1 goto finalActions
echo.
echo Failed to execute ParaProf... 
echo Please check to make sure Sun Java is installed and is in the path
echo. 

pause
:finalActions
