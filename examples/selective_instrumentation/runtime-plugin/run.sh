#!/bin/bash
# Builds the runtime filter plugin source (only its SI code is exercised; TAU runtime symbols are left unresolved)
# into a driver and runs the common cases.
here=$(cd "$(dirname "$0")" && pwd)
root=$here/../../..
CXX=${CXX:-g++}
work=${SI_WORKDIR:-$(mktemp -d)}
$CXX -w -std=c++14 -DTAU_ENABLED -DPROFILING_ON -DTAU_STDCXXLIB -DTAU_DOT_H_LESS_HEADERS -I"$root/include" \
  -c "$root/plugins/examples/Tau_plugin_example_disable_instrumentation_runtime.cpp" -o "$work/plugin.o" 2>&1 | tail -5
[ -f "$work/plugin.o" ] || { echo "FAIL runtime-plugin: build"; exit 1; }
$CXX -w "$here/driver.cpp" "$work/plugin.o" -Wl,--unresolved-symbols=ignore-all -o "$work/plugin_driver" || { echo "FAIL runtime-plugin: link"; exit 1; }
status=0
for sel in "$here"/../common/*.tau; do
  base=${sel%.tau}
  if out=$("$work/plugin_driver" "$sel" "$base.expect" 2>&1); then echo "PASS runtime-plugin $(basename "$base")"
  else echo "FAIL runtime-plugin $(basename "$base")"; echo "$out"; status=1; fi
done
exit $status
