#!/bin/sh

ROOT=/cygdrive/c/tau


# below is like a 'make install' style step
rm -rf $ROOT/tau-windows
mkdir -p $ROOT/tau-windows/bin
mkdir -p $ROOT/tau-windows/include
mkdir -p $ROOT/tau-windows/lib
mkdir -p $ROOT/tau-windows/examples
mkdir -p $ROOT/tau-windows/JavaDLL
cp -r $ROOT/tau2/include /c/tau/tau-windows


cd $ROOT/tau2
rm $ROOT/tau2/win32/bin/TraceInput.exp
rm $ROOT/tau2/win32/bin/TraceInput.lib

cp $ROOT/tau2/utils/taupin/tau_pin.exe $ROOT/tau-windows/bin
cp $ROOT/tau2/utils/taupin/*.dll $ROOT/tau-windows/bin
cp $ROOT/tau2/utils/taupin/README-PIN.txt $ROOT/tau-windows
cp $ROOT/pin/*.* $ROOT/tau-windows/bin

cp $ROOT/tau2/tools/src/windows/bin/* $ROOT/tau-windows/bin
cp $ROOT/tau2/src/Profile/TAU.jar $ROOT/tau-windows/bin
cp -r $ROOT/tau2/tools/src/windows/examples $ROOT/tau-windows
mkdir $ROOT/tau-windows/examples/pin
cp $ROOT/pin/examples/** $ROOT/tau-windows/examples/pin
mkdir -p $ROOT/tau-windows/tools/src/perfdmf/etc
mkdir -p $ROOT/tau-windows/etc
mkdir -p $ROOT/tau-windows/contrib
cp tools/src/perfexplorer/etc/* $ROOT/tau-windows/etc
cp $ROOT/weka.jar $ROOT/tau-windows/bin
cp -r $ROOT/tau2/tools/src/perfdmf/etc $ROOT/tau-windows/tools/src/perfdmf
cp win32/bin/*.* $ROOT/tau-windows/bin
cp tools/src/contrib/*.jar $ROOT/tau-windows/bin
cp tools/src/perfdmf/bin/perfdmf.jar $ROOT/tau-windows/bin
cp tools/src/perfexplorer/perfexplorer.jar $ROOT/tau-windows/bin
cp tools/src/common/bin/tau-common.jar $ROOT/tau-windows/bin
cp tools/src/paraprof/bin/*.jar $ROOT/tau-windows/bin
cp tools/src/vis/bin/*.jar $ROOT/tau-windows/bin
cp tools/src/contrib/jogl/jogl.jar $ROOT/tau-windows/bin
cp tools/src/contrib/jogl/windows/jogl.dll $ROOT/tau-windows/bin
cp tools/src/contrib/jogl/windows/jogl_awt.dll $ROOT/tau-windows/bin
cp tools/src/contrib/jogl/windows/jogl_cg.dll $ROOT/tau-windows/bin
cp tools/src/contrib/slog2sdk/lib/jumpshot.jar $ROOT/tau-windows/bin
cp win32/lib/*.* $ROOT/tau-windows/lib
cp win32/java/*.* $ROOT/tau-windows/javadll
cp LICENSE $ROOT/tau-windows
cp tools/src/windows/etc/Readme.txt $ROOT/tau-windows
cp tools/src/contrib/LICENSE-* $ROOT/tau-windows/contrib
cp tools/src/contrib/jogl/*.txt $ROOT/tau-windows/contrib

cd $ROOT/tau-windows
find . -name "CVS" | xargs rm -rf
cd $ROOT

