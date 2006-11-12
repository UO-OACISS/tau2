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

echo "Installing to $1/plugins"
echo "..."
cp ./plugins/*.jar $1/plugins/
cp -r ./plugins/org.eclipse.ptp.tau.perfdmf_1.0.0/ $1/plugins/

cp ../contrib/batik-combined.jar $1/plugins/org.eclipse.ptp.tau.perfdmf_1.0.0/
cp ../contrib/jargs.jar $1/plugins/org.eclipse.ptp.tau.perfdmf_1.0.0/
cp ../contrib/jcommon-0.9.6.jar $1/plugins/org.eclipse.ptp.tau.perfdmf_1.0.0/
cp ../contrib/jfreechart-0.9.21.jar $1/plugins/org.eclipse.ptp.tau.perfdmf_1.0.0/
cp ../contrib/jgraph.jar $1/plugins/org.eclipse.ptp.tau.perfdmf_1.0.0/
cp ../contrib/jogl.jar $1/plugins/org.eclipse.ptp.tau.perfdmf_1.0.0/
cp ../contrib/jython.jar $1/plugins/org.eclipse.ptp.tau.perfdmf_1.0.0/

cp ../paraprof/bin/paraprof.jar $1/plugins/org.eclipse.ptp.tau.perfdmf_1.0.0/
cp ../perfdmf/bin/perfdmf.jar $1/plugins/org.eclipse.ptp.tau.perfdmf_1.0.0/
cp ../common/bin/tau-common.jar $1/plugins/org.eclipse.ptp.tau.perfdmf_1.0.0/
cp ../vis/bin/vis.jar $1/plugins/org.eclipse.ptp.tau.perfdmf_1.0.0/

echo "Eclipse plugins installed!"
