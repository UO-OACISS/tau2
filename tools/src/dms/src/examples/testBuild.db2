#!/bin/sh
TAUROOT=/home/users/khuck/tau2
PERFDB_HOME=${TAUROOT}/tools/src/dms
XMLSAX_JAR=${TAUROOT}/tools/src/contrib/xerces.jar
# JDBC_JAR=${TAUROOT}/tools/src/contrib/postgresql.jar
JDBC_JAR=/home/users/khuck/db2/db2java.jar:/home/db2inst/sqllib/java/db2jcc.jar:/home/db2inst/sqllib/java/db2jcc_license_cu.jar:/home/db2inst/jndi/lib/jndi.jar:/home/db2inst/jndi/lib/providerutil.jar:/home/db2inst/jndi/lib/fscontext.jar
DMS_JAR=${PERFDB_HOME}/dms.jar

java -cp ${PERFDB_HOME}/src/examples:${DMS_JAR}:${XMLSAX_JAR}:${JDBC_JAR} examples.TestPerfDBSession ${PERFDB_HOME}/data/perfdb.cfg
