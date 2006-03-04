#!/bin/sh

cd /c/tau-windows/bin
rm -f *.exe *.dll *.jar
rm -f ../lib/*
rm -f ../JavaDLL/*
rm -rf ../include

cd /c 
rm -rf tau2
export CVS_RSH=$HOME/bin/cvs_rsh
CVSROOT=amorris@ix.cs.uoregon.edu:/research/paraducks2/cvs-src/master cvs co tau2

gunzip /c/tau2/tools/src/contrib/windows/jogl.dll.gz
cp -r /c/tau2/include /c/tau-windows
rm -rf /c/tau-windows/include/makefiles
rm /c/tau-windows/include/Makefile
cd /c/tau-windows/include
find . -name "CVS" | xargs rm -rf
