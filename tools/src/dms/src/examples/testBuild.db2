#!/bin/sh
TAUROOT=/home/users/khuck/tau2
PERFDMF_HOME=${TAUROOT}/tools/src/dms
XMLSAX_JAR=${TAUROOT}/tools/src/contrib/xerces.jar
# JDBC_JAR=${TAUROOT}/tools/src/contrib/postgresql.jar
JDBC_JAR=/home/users/khuck/db2/db2java.jar:/home/db2inst/sqllib/java/db2jcc.jar:/home/db2inst/sqllib/java/db2jcc_license_cu.jar:/home/db2inst/jndi/lib/jndi.jar:/home/db2inst/jndi/lib/providerutil.jar:/home/db2inst/jndi/lib/fscontext.jar
DMS_JAR=${PERFDMF_HOME}/dms.jar

java -cp ${PERFDMF_HOME}/src/examples:${DMS_JAR}:${XMLSAX_JAR}:${JDBC_JAR} examples.TestPerfDMFSession ${PERFDMF_HOME}/data/perfdmf.cfg
