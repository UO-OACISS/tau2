#!/bin/sh

REPO="http://www.nic.uoregon.edu/~amorris/regression"

ROOT=/cygdrive/c/tau


# export CVS_RSH=$HOME/bin/cvs_rsh
# CVSROOT=amorris@ix.cs.uoregon.edu:/research/paraducks2/cvs-src/master cvs co tau2
# find $ROOT/tau2 -name "CVS" | xargs rm -rf

cd $ROOT 
rm -rf tau2
rm tau2.tar.gz
wget $REPO/checkouts/tau2.tar.gz
tar -xzvf tau2.tar.gz


# set up for windows
gunzip $ROOT/tau2/tools/src/contrib/jogl/windows/jogl.dll.gz
gunzip $ROOT/tau2/tools/src/contrib/jogl/windows/jogl_awt.dll.gz
gunzip $ROOT/tau2/tools/src/contrib/jogl/windows/jogl_cg.dll.gz
cp $ROOT/tau2/tools/src/windows/trace_impl.h $ROOT/tau2/utils/slogconverter
cp $ROOT/tau2/tools/src/windows/etc/tau_config.h $ROOT/tau2/include
cp $ROOT/tau2/tools/src/windows/etc/tauarch.h $ROOT/tau2/include

