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
