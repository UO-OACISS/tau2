#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
paraprof_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT HUP INT TERM

javac -source 8 -target 8 -Xlint:-options -d "$build_dir" \
  "$paraprof_dir/src/SlurmTopologyLayout.java" \
  "$script_dir/SlurmTopologyLayoutTest.java"
java -cp "$build_dir" edu.uoregon.tau.paraprof.SlurmTopologyLayoutTest

# Hover-pick math test; needs the JOGL jars to load TopoPlot (no GL context is created).
tools_src=$(CDPATH= cd -- "$paraprof_dir/.." && pwd)
vis_cp="$tools_src/contrib/jogl/jogl-all.jar:$tools_src/contrib/jogl/gluegen-rt.jar:$tools_src/common/bin/tau-common.jar"
javac -source 8 -target 8 -Xlint:-options -nowarn -cp "$vis_cp" -d "$build_dir" \
  "$tools_src"/vis-jogl2/src/*.java "$script_dir/TopoPlotPickTest.java"
java -cp "$build_dir:$vis_cp" TopoPlotPickTest
