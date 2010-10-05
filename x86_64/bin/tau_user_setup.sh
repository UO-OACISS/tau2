#!/bin/sh

# $Id: tau_user_setup.sh.skel,v 1.1 2009/12/01 21:41:25 amorris Exp $

TAUROOT=/home/scottb/tau2
MACHINE=x86_64
PARAPROF_HOME=${HOME}/.ParaProf
JYTHON_HOME=${HOME}/.ParaProf/jython

if [ ! -d ${PARAPROF_HOME} ] ; then
    mkdir -p ${PARAPROF_HOME}
fi

if [ ! -d ${JYTHON_HOME} ] ; then
    mkdir -p ${JYTHON_HOME}
fi

cat ${TAUROOT}/etc/derby.properties.skel | sed -e 's,@HOME@,'${HOME}',' > ${PARAPROF_HOME}/derby.properties

cp ${TAUROOT}/etc/jython.registry ${JYTHON_HOME}/registry



