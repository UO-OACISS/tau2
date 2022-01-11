#!/bin/bash -x

#REPO="http://www.nic.uoregon.edu/~amorris/regression"
REPO="http://www.cs.uoregon.edu/research/paracomp/tau/tauprofile/dist"

ROOT=/cygdrive/c/tau


# export CVS_RSH=$HOME/bin/cvs_rsh
# CVSROOT=amorris@ix.cs.uoregon.edu:/research/paraducks2/cvs-src/master cvs co tau2
# find $ROOT/tau2 -name "CVS" | xargs rm -rf

cd $ROOT 
#rm -rf tau2
rm tau.tgz tau-$1.tar.gz
##wget $REPO/tau-$1.tar.gz
##cp tau-$1-win.tar.gz tau-$1.tar.gz
##tar -xzvf tau-$1.tar.gz
#set -e

##
##Do not enable this until the public archive of TAU has the latest windows jogl support
##

#wget -q tau.uoregon.edu/tau.tgz
#tar -zxf tau.tgz
#ls -lh
#mv ./tau-$1 ./tau2

## set up for windows

#gunzip $ROOT/tau2/tools/src/contrib/jogl/windows64/jogl.dll.gz
#gunzip $ROOT/tau2/tools/src/contrib/jogl/windows64/jogl_awt.dll.gz
#gunzip $ROOT/tau2/tools/src/contrib/jogl/windows64/jogl_cg.dll.gz
#gunzip $ROOT/tau2/tools/src/contrib/jogl/windows64/gluegen-rt.dll.gz
#
#gunzip $ROOT/tau2/tools/src/contrib/jogl/windows/jogl.dll.gz
#gunzip $ROOT/tau2/tools/src/contrib/jogl/windows/jogl_awt.dll.gz
#gunzip $ROOT/tau2/tools/src/contrib/jogl/windows/jogl_cg.dll.gz

tar -zxf $ROOT/tau2/tools/src/contrib/jogl/windows-all.tgz 

cp $ROOT/tau2/tools/src/windows/trace_impl.h $ROOT/tau2/utils/slogconverter
cp $ROOT/tau2/tools/src/windows/etc/tau_config.h $ROOT/tau2/include
cp $ROOT/tau2/tools/src/windows/etc/tauarch.h $ROOT/tau2/include

