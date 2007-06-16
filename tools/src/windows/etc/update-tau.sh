#!/bin/sh

ROOT=/cygdrive/c/tau

cd $ROOT/tau-windows/bin
rm -f *.exe *.dll *.jar
rm -f ../lib/*
rm -f ../JavaDLL/*
rm -rf ../include

cd $ROOT 
rm -rf tau2
export CVS_RSH=$HOME/bin/cvs_rsh
CVSROOT=amorris@ix.cs.uoregon.edu:/research/paraducks2/cvs-src/master cvs co tau2
find $ROOT/tau2 -name "CVS" | xargs rm -rf

gunzip $ROOT/tau2/tools/src/contrib/jogl/windows/jogl.dll.gz
gunzip $ROOT/tau2/tools/src/contrib/jogl/windows/jogl_awt.dll.gz
gunzip $ROOT/tau2/tools/src/contrib/jogl/windows/jogl_cg.dll.gz
cp -r $ROOT/tau2/include /c/tau/tau-windows
rm -rf $ROOT/tau-windows/include/makefiles
rm $ROOT/tau-windows/include/Makefile
cd $ROOT/tau-windows
