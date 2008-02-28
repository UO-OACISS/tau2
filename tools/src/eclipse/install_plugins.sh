#!/bin/sh
if [ $# != 1 ] ; then
  echo "Usage: $0 <path/to/eclipse/root>"
  echo ""
  echo "Please enter the location of your eclipse installation."
  echo "Note that you must have the CDT and PTP plugins installed"
  echo "for the TAU plugins to function properly."
  exit 1
fi

if [ ! -d "$1" ] ; then
    echo "Invalid eclipse directory"
    exit 1
fi

if [ ! -d "$1"/plugins/ ] ; then
    echo "Warning: No plugins directory in eclipse root.  Creating directory"
    mkdir $1/plugins/
fi

CURRENT_DIR=`pwd`

cd $1/plugins/
PLUG_DIR=`pwd`
cd $CURRENT_DIR

cd `dirname $0`
echo "Installing to $1/plugins"
echo "..."
cp ./plugins/*.jar $PLUG_DIR
cp -r ./plugins/org.eclipse.ptp.perf.tau.jars_1.0.0/ $PLUG_DIR

cp ../contrib/batik-combined.jar $PLUG_DIR/org.eclipse.ptp.perf.tau.jars_1.0.0/
cp ../contrib/jargs.jar $PLUG_DIR/org.eclipse.ptp.perf.tau.jars_1.0.0/
cp ../contrib/jcommon-0.9.6.jar $PLUG_DIR/org.eclipse.ptp.perf.tau.jars_1.0.0/
cp ../contrib/jfreechart-0.9.21.jar $PLUG_DIR/org.eclipse.ptp.perf.tau.jars_1.0.0/
cp ../contrib/jgraph.jar $PLUG_DIR/org.eclipse.ptp.perf.tau.jars_1.0.0/
cp ../contrib/jogl/jogl.jar $PLUG_DIR/org.eclipse.ptp.perf.tau.jars_1.0.0/
cp ../contrib/jython.jar $PLUG_DIR/org.eclipse.ptp.perf.tau.jars_1.0.0/
cp ../contrib/xerces.jar $PLUG_DIR/org.eclipse.ptp.perf.tau.jars_1.0.0/

cp ../paraprof/bin/paraprof.jar $PLUG_DIR/org.eclipse.ptp.perf.tau.jars_1.0.0/
cp ../perfdmf/bin/perfdmf.jar $PLUG_DIR/org.eclipse.ptp.perf.tau.jars_1.0.0/
cp ../common/bin/tau-common.jar $PLUG_DIR/org.eclipse.ptp.perf.tau.jars_1.0.0/
cp ../vis/bin/vis.jar $PLUG_DIR/org.eclipse.ptp.perf.tau.jars_1.0.0/
cd $CURRENT_DIR
echo "Eclipse plugins installed!"
