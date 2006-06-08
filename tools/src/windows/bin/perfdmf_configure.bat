::
:: set TAU_ROOT below and make sure java is in your path
::
@echo off

set TAU_ROOT=..
set JAR_ROOT=%TAU_ROOT%/bin
set CONFIG_FILE=%HOME%/.ParaProf/perfdmf.cfg
set PERFDMF_JAR=%JAR_ROOT%/perfdmf.jar
set JARGS_JAR=%JAR_ROOT%/jargs.jar
set JDBC_JAR=%JAR_ROOT%/postgresql.jar;%JAR_ROOT%/mysql.jar;%JAR_ROOT%/oracle.jar;%JAR_ROOT%/derby.jar
set COMMON_JAR=%JAR_ROOT%/tau-common.jar

java -Xmx500m -Djava.library.path=%TAU_ROOT%\bin -cp %PERFDMF_JAR%;%JARGS_JAR%;%JDBC_JAR%;%COMMON_JAR% edu/uoregon/tau/perfdmf/loader/Configure -t %TAU_ROOT% %1 %2 %3 %4 %5


java -Xmx500m -Djava.library.path=%TAU_ROOT%\bin -cp %PERFDMF_JAR%;%JARGS_JAR%;%JDBC_JAR%;%COMMON_JAR% edu/uoregon/tau/perfdmf/loader/ConfigureTest -t %TAU_ROOT% %1 %2 %3 %4 %5
