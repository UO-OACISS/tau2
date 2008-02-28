#!/bin/bash
if [ $# != 1 ] ; then
  echo "Usage: $0 <path/to/tau/root>"
  echo ""
  echo "Please enter the location of your tau installation."
  exit 1
fi

if [ ! -d "$1"/tools/src/ ] ; then
    echo "Invalid tau directory"
    exit 1
fi

cp "$1"/tools/src/contrib/batik-combined.jar .
cp "$1"/tools/src/contrib/jcommon-0.9.6.jar .
cp "$1"/tools/src/contrib/jargs.jar .
cp "$1"/tools/src/contrib/jfreechart-0.9.21.jar .
cp "$1"/tools/src/contrib/jgraph.jar .
cp "$1"/tools/src/contrib/jython.jar .
cp "$1"/tools/src/contrib/jogl/jogl.jar .
cp "$1"/tools/src/contrib/xerces.jar .
cp "$1"/tools/src/paraprof/bin/paraprof.jar .
cp "$1"/tools/src/perfdmf/bin/perfdmf.jar .
cp "$1"/tools/src/common/bin/tau-common.jar .
cp "$1"/tools/src/vis/bin/vis.jar .

echo "Tau Jars Copied"
