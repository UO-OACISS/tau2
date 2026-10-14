#!/bin/bash
# Builds utils/tau_selective.dyn.cpp (the tau_run / Dyninst-side implementation) into a driver and runs the common cases.
here=$(cd "$(dirname "$0")" && pwd)
utils=$here/../../../utils
CXX=${PDT_CXX:-${CXX:-g++}}
if [ -z "$PDTDIR" ]; then
    echo "FAIL: PDTDIR must be set for this test to work"
    exit 1
fi
if [ ! -f "$PDTDIR/include/pdbAll.h" ]; then echo "SKIP dyninst: set PDTDIR to a PDT installation"; exit 77; fi
work=${SI_WORKDIR:-$(mktemp -d)}
$CXX -w -DTAU_DYNINST -DTAU_DYNINST41PLUS -I"$utils" -I"$PDTDIR/include" "$here/driver.cpp" "$utils/tau_selective.dyn.cpp" $(ls "$PDTDIR"/*/lib/libpdb.a | head -1) -o "$work/dyn_driver" 2>&1 | tail -5; [ -x "$work/dyn_driver" ] || { echo "FAIL dyninst: build"; exit 1; }
status=0
for sel in "$here"/../common/*.tau; do
  base=${sel%.tau}
  if out=$("$work/dyn_driver" "$sel" "$base.expect" 2>&1); then echo "PASS dyninst $(basename "$base")"
  else echo "FAIL dyninst $(basename "$base")"; echo "$out"; status=1; fi
done
exit $status
