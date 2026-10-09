#!/bin/bash
# Builds a small driver around plugins/llvm/src/Instrument.cpp and runs every common case through it.
# Requires clang++ and llvm-config (override with LLVM_CONFIG / LLVM_CXX).
here=$(cd "$(dirname "$0")" && pwd)
LLVM_CONFIG=${LLVM_CONFIG:-$(command -v llvm-config llvm-config-20 | head -1)}
LLVM_CXX=${LLVM_CXX:-$(command -v clang++ | head -1)}
if [ -z "$LLVM_CONFIG" ] || [ -z "$LLVM_CXX" ]; then echo "SKIP llvm: llvm-config or clang++ not found"; exit 77; fi
work=${SI_WORKDIR:-$(mktemp -d)}
$LLVM_CXX -w -std=c++17 -DTAU_PROF_CXX -I$($LLVM_CONFIG --includedir) -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS \
  "$here/driver.cpp" $($LLVM_CONFIG --ldflags --libs core support analysis) -o "$work/llvm_driver" || { echo "FAIL llvm: build"; exit 1; }
status=0
for sel in "$here"/../common/*.tau; do
  base=${sel%.tau}
  # The parser must not complain about any of the (valid) common select files.
  if "$work/llvm_driver" "$sel" "$base.expect" > "$work/out.txt" 2>"$work/err.txt" && [ ! -s "$work/err.txt" ]; then echo "PASS llvm $(basename "$base")"
  else echo "FAIL llvm $(basename "$base")"; cat "$work/out.txt" "$work/err.txt"; status=1; fi
done
exit $status
